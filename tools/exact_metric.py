"""The harness's surface metric (`verify.surface_metric`: points on the B-rep faces, probes along the normals into the
exact swept-volume workpiece, `cam_agent.services.exact`) of a saved session, and optionally the whole part after
machining as a closed STL. Recomputes the metric without re-running CAM: single- and multi-setup sessions (built, not
dropped setups), the points from the session's STEP, the blank from `stock.mesh_raw` or the primary setup's stock box.
Sessions of older harness versions (dexel metric) get the current metric this way.

  FREECADCMD=$CONDA_PREFIX/bin/freecadcmd python tools/exact_metric.py <session_dir> <out_dir> [--rapids cut|skip]
      [--threads N] [--mesh 0.15] [--project <make_project.py run of a setup> ...]

Writes <out_dir>/surface.json (key "sim"; "agent" — the session's own numbers, "diff" — sim - agent), gouge_points.npz,
surface_final.npz (all points and verdicts, original frame) and, with --mesh, exact.stl (original part frame).
--project: the setup index and the operation labels a ready project (the sim line's make_project.py) comments out — cut
only its operations, setups without one are not cut.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cam_agent.services import exact as E  # noqa: E402
from cam_agent.services.freecad_runner import FreeCADRunner  # noqa: E402
from cam_agent.services.stock import box_mesh, load_mesh  # noqa: E402
from cam_agent.services.verify import brep_samples, save_surface, surface_metric  # noqa: E402
from cam_agent.services.voxel import Frame  # noqa: E402

KEYS = ("surface_within_tol_pct", "band_pct", "excess_cm3", "gouge_cm3", "excess_area_pct", "gouge_area_pct", "max_excess_mm", "max_gouge_mm")


def project_ops(run: Path) -> tuple[int, set]:
    """A ready project (make_project.py output): its setup index and the labels it comments out (`--ops`, taps …)."""
    man = json.loads((run / "manifest.json").read_text())
    return (man.get("setup") or {}).get("index") or 1, {o["label"] for o in man["operations"] if o["skipped"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rapids", choices=("cut", "skip"), default="cut", help="G0 removes material (the harness) or not")
    ap.add_argument("--arc-tol", type=float, default=E.ARC_TOL, help="mm, arc linearisation (the harness's metric: %(default)s)")
    ap.add_argument("--cell", type=float, default=2.0, help="mm, bucket of the move index")
    ap.add_argument("--samples", type=int, default=600000)
    ap.add_argument("--threads", type=int, help="numba threads")
    ap.add_argument("--mesh", type=float, metavar="STEP",
                    help="also write the whole part after machining as a closed STL (<out>/exact.stl, original part frame), grid step, mm")
    ap.add_argument("--mesh-bisect", type=int, default=10, help="bisections of a crossed edge (vertex error step / 2^N)")
    ap.add_argument("--mesh-method", choices=("nets", "tets", "nets-dense"), default="nets",
                    help="nets: manifold surface nets (a vertex per sheet of a cell), sparse (blocks proven uniform "
                         "skipped); tets: marching tetrahedra, sparse, ~4x the faces; nets-dense: the first surface nets "
                         "on the dense grid (a vertex per cell: not a manifold at walls thinner than a cell)")
    ap.add_argument("--mesh-block", type=int, default=16, help="nets/tets: top block, cells (power of 2)")
    ap.add_argument("--mesh-leaf", type=int, default=4, help="nets/tets: leaf block whose nodes are evaluated, cells")
    ap.add_argument("--project", type=Path, action="append", default=[],
                    help="ready project of a setup (make_project.py output): cut only its operations, setups without one "
                         "are not cut; repeat per setup")
    a = ap.parse_args()
    projects = dict(project_ops(p) for p in a.project)
    if a.threads:
        E.nb.set_num_threads(a.threads)
    t0 = time.time()
    a.out.mkdir(parents=True, exist_ok=True)
    s = json.loads((a.session / "session.json").read_text())
    tol = float(s.get("tol_mm") or 0.1)
    multi = bool(s.get("setups"))
    cam = s["setups"][0]["cam"] if multi else s["cam"]
    req = cam["request"]
    step_path = a.session / Path(req["path"]).name
    if not step_path.exists():
        step_path = ROOT / req["path"]
    samples = brep_samples(step_path, a.samples, a.out / "fc", FreeCADRunner())

    e1 = s["setups"][0] if multi else {"up": req.get("up", "+Z"), "translation": req.get("translation")}
    frame0 = Frame(e1["up"], e1["translation"])
    mesh_raw = (s.get("stock") or {}).get("mesh_raw")
    if mesh_raw and (ROOT / mesh_raw).exists():
        stock_raw = load_mesh(ROOT / mesh_raw)
    elif mesh_raw and Path(mesh_raw).exists():
        stock_raw = load_mesh(mesh_raw)
    else:  # no stock mesh: the primary setup's stock box, back in the original frame
        b = cam["verification"]["stock_bbox"]
        corners = np.array([[x, y, z] for x in (b[0], b[3]) for y in (b[1], b[4]) for z in (b[2], b[5])], float)
        o = frame0.to_original(corners)
        stock_raw = box_mesh(o.min(0).tolist(), o.max(0).tolist())
    t_prep = time.time() - t0

    setups = [(e["index"], e) for e in s["setups"] if e.get("cam") and not e.get("dropped")] if multi else [(1, None)]
    if projects:
        if set(projects) - {k for k, _ in setups}:
            raise SystemExit(f"projects of setups {sorted(projects)}, the session has {[k for k, _ in setups]}")
        setups = [(k, e) for k, e in setups if k in projects]
    programs = []
    for k, e in setups:
        c = e["cam"] if e else cam
        g = Path(c["gcode_file"])
        g = a.session / (g.parent.name if e else "") / g.name
        programs.append((Frame(e["up"], e["translation"]) if e else frame0, g, c["request"]["operations"], c["request"]["tools"]))
    wp, per = E.workpiece(stock_raw, programs, a.rapids == "cut", a.arc_tol, a.cell, [projects.get(k) for k, _ in setups])
    for p, (k, _) in zip(per, setups):
        p["setup"] = k
    t_metric = time.time()
    sd = surface_metric(samples, wp.inside, tol)
    t_metric = time.time() - t_metric
    save_surface(a.out / "surface_final.npz", sd, "original")
    for p, st in zip(per, wp.setups):
        p["queries"] = st.queries

    gs = sd["dev"] <= -tol + 1e-9
    Q = sd["points"][gs] - tol * sd["normals"][gs]
    cut = np.stack([~st.inside(Q) for st in wp.setups]) if gs.any() else np.zeros((len(wp.setups), 0), bool)
    for p, m in zip(per, cut):
        p["gouge_removed_mm2"] = round(float(sd["w"][gs][m].sum()), 1)
    np.savez_compressed(a.out / "gouge_points.npz", P=sd["points"][gs], N=sd["normals"][gs], dev=sd["dev"][gs], cut_by=cut)

    mesh = None
    if a.mesh:
        import trimesh
        tm = time.time()
        V0 = np.asarray(stock_raw["vertices"], float)
        if a.mesh_method == "nets-dense":
            V, F, mesh = E.mesh_of(wp.inside, V0.min(0), V0.max(0), a.mesh, a.mesh_bisect)
            mesh["method"] = "nets-dense"
        else:
            V, F, mesh = (E.mesh_nets if a.mesh_method == "nets" else E.mesh_tets)(
                wp, V0.min(0), V0.max(0), a.mesh, a.mesh_bisect, a.mesh_block, a.mesh_leaf)
        t_build = time.time() - tm
        trimesh.Trimesh(V, F, process=False).export(a.out / "exact.stl")
        chk = E.mesh_check(V, F)  # not trimesh's is_watertight: its edge hashing needs ~1 GB per 10 M faces
        vol = float(np.einsum("ij,ij->i", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])).sum()) / 6 if len(F) else 0.0
        mesh.update({"stl": str((a.out / "exact.stl").resolve()), "vertices": len(V), "faces": len(F),
                     "watertight": chk["edges_not_2"] == 0 and chk["directed_edges_repeated"] == 0,
                     "volume_cm3": round(vol / 1000, 3), "seconds": round(t_build, 1), "check": chk})

    agent = (s["multi_setup"].get("final") or {}) if s.get("multi_setup") else ((cam.get("verification") or {}).get("deviation") or {}).get("stats")
    out = {"part": s.get("part_name"), "session": str(a.session.resolve()), "simulator": "exact", "rapids": a.rapids,
           "arc_tol_mm": a.arc_tol, "projects": [str(p.resolve()) for p in a.project], "tol_mm": tol,
           "blank": "box" if isinstance(wp, E.BoxWorkpiece) else "mesh", "sim": sd["stats"], "agent": agent, "setups": per,
           "diff": {k: round(sd["stats"][k] - agent[k], 3) for k in KEYS if agent and agent.get(k) is not None},
           **({"mesh": mesh} if mesh else {}),
           "seconds": {"total": round(time.time() - t0, 1), "prep": round(t_prep, 1), "metric": round(t_metric, 1),
                       "threads": E.nb.get_num_threads()}}
    (a.out / "surface.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("part", "blank", "diff", "seconds")}, indent=1))
    print("exact ", {k: sd["stats"][k] for k in KEYS})
    print("agent ", {k: (agent or {}).get(k) for k in KEYS})


if __name__ == "__main__":
    main()
