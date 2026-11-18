"""Setup Planning Agent: indexed (3+2 / multi-setup) machining.

Features are recognised from all six sides. A feature belongs to a setup only if the void it stands for is
(a) reachable from that side - visible (no target material above it) AND within the reach of a cutter of the library
whose holder clears the part around it - and (b) not already removed by the setups chosen so far.
Setups are chosen greedily by the volume of new material they can reach. The covered volume is a monotone submodular
function of the set of sides (a union of fixed sets), so the greedy choice is within 1 - 1/e of the best choice of
the same number of setups; the plan reports that bound with the volume no choice of sides can reach.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..services import FreeCADRunner
from ..services.voxel import Frame, VoxelGrid

SIDES = ["+Z", "-Z", "+X", "-X", "+Y", "-Y"]
MACHINED = ("hole", "pocket", "slot", "step")
EDGE = ("planar_face", "outer_profile", "chamfer", "fillet", "boss", "sloped_face")


def _boundary_links(target: np.ndarray) -> np.ndarray:
    """Voxel target surface: count all six target/void links at each void cell.

    Each link has area grid.res**2. This is the existing voxel representation,
    not exact B-Rep area or a claim of cutter fit/manufacturing feasibility.
    """
    links = np.zeros(target.shape, np.uint8)
    for axis in range(3):
        a, b = [slice(None)] * 3, [slice(None)] * 3
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        links[a] += ~target[a] & target[b]
        links[b] += ~target[b] & target[a]
    return links


def _area_admitted(gain: float, new_links: int, total_links: int) -> bool:
    """Inclusive 1% on one grid; require positive new material evidence."""
    return gain > 0 and total_links > 0 and 100 * new_links >= total_links


class SetupPlanningAgent:
    name = "setup_planning"

    def __init__(self, freecad: FreeCADRunner | None = None, max_setups: int = 4, min_gain_mm3: float = 30.0, stock_mesh: dict | None = None,
                 reach: list | None = None, cutters: list | None = None, holder_d: float = 40.0) -> None:
        """reach: [(holder overhang radius, reach)] of the tool library (cam_generation.reach_profile); None = visibility only.
        cutters: [(diameter, reach)] of the library (cam_generation.cutter_reach). Given, a voxel counts as reachable only
        if one cutter both fits the opening at that height and is long enough, not the longest cutter of any size."""
        self.freecad = freecad or FreeCADRunner()
        self.max_setups, self.min_gain = max_setups, min_gain_mm3
        self.reach = [tuple(map(float, p)) for p in reach] if reach else None
        self.cutters = [tuple(map(float, c)) for c in cutters] if cutters else None
        self.holder_d = float(holder_d)
        self.stock_mesh = stock_mesh  # user's blank in the part's original frame (None = default box: the part's bounding box is
        # what has to be removed; the few millimetres of box margin do not justify a setup of their own)

    def run(self, cad_path: str | Path, workdir: Path, primary: str = "auto", log=lambda m: None, groups: list | None = None) -> dict:
        """groups: face groups of a learned recogniser (AAGNet) for the original STEP, used on every side."""
        cad_path = str(Path(cad_path).resolve())
        req = {"path": cad_path, "ups": SIDES}
        if groups:
            req["groups"] = groups
        by_up = self.freecad.run("extract_features.py", req, workdir)["by_up"]
        mesh = self.freecad.run("tessellate.py", {"path": cad_path, "deflection": 0.05, "raw": True}, workdir)
        sm = self.stock_mesh
        if sm is None:
            grid = VoxelGrid(mesh["bbox"])
        else:
            from ..services.stock import union_bbox
            grid = VoxelGrid(union_bbox(mesh["bbox"], sm["bbox"]))
        target = grid.voxelize(mesh["vertices"], mesh["faces"])
        void = ~target
        frames = {u: Frame(u, by_up[u]["translation"]) for u in SIDES}
        visible = {u: grid.visible(target, frames[u]) for u in SIDES}
        # reachable share of the part per side (same meaning as services.access.accessibility)
        part_vol = float(target.sum())
        access = {}
        for u in SIDES:
            shadow = ~visible[u]  # everything a tool from u cannot reach = the part plus its undercuts
            access[u] = round(part_vol / max(float(shadow.sum()), 1.0), 3)
        cols = {u: grid.columns(frames[u]) for u in SIDES}
        # what a tool from u really gets to: visible and not deeper than the reach below the part around it (holder)
        if self.cutters:
            reach = {u: self._reachable_fit(grid, target, frames[u], cols[u], visible[u]) for u in SIDES}
        elif self.reach:
            reach = {u: self._reachable(grid, target, frames[u], cols[u], visible[u]) for u in SIDES}
        else:
            reach = visible

        def feature_void(u: str, f: dict) -> np.ndarray:
            col, zk, (ncy, ncx), lo, hi = cols[u]
            if f["type"] == "hole":
                r = f["diameter"] / 2
                x0, x1, y0, y1 = f["top"][0] - r, f["top"][0] + r, f["top"][1] - r, f["top"][1] + r
                z0, z1 = f["top_z"] - f["depth"], f["top_z"] + grid.res
            else:
                x0, y0, x1, y1 = f["bbox"]
                z0, z1 = f["z"], hi[2]
            jj, ii = col % ncx, col // ncx
            xs, ys = lo[0] + (jj + 0.5) * grid.res, lo[1] + (ii + 0.5) * grid.res
            m = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1) & (zk >= z0 - 1e-6) & (zk <= z1)
            if f["type"] == "hole":
                m &= (xs - f["top"][0]) ** 2 + (ys - f["top"][1]) ** 2 <= (r + grid.res / 2) ** 2
            return m.reshape(grid.shape) & void

        def candidates(u: str, removed: np.ndarray) -> tuple[list[dict], float]:
            keep, gain = [], 0.0
            for f in by_up[u]["features"]:
                if f["type"] not in MACHINED or (f["type"] == "hole" and not f.get("vertical")):
                    continue
                v = feature_void(u, f)
                n = float(v.sum())
                if n * grid.cell_volume < 8.0:
                    continue
                get = visible[u] if f["type"] == "hole" else reach[u]  # holes: drills are long, tool selection checks their length
                vis = float((v & get).sum()) / n
                new = float((v & get & ~removed).sum())
                if vis >= 0.8 and new >= 0.5 * n:
                    keep.append({**f, "void_mm3": round(n * grid.cell_volume, 1), "new_mm3": round(new * grid.cell_volume, 1), "visible": round(vis, 2)})
                    gain += new * grid.cell_volume
            return keep, gain

        x, y, z = grid.axes()
        bb = mesh["bbox"]
        if sm is None:
            inside = ((z >= bb[2]) & (z <= bb[5]))[:, None, None] & ((y >= bb[1]) & (y <= bb[4]))[None, :, None] & ((x >= bb[0]) & (x <= bb[3]))[None, None, :]
        else:  # the material to remove is the blank minus the part, not the part's bounding box
            inside = grid.voxelize(sm["vertices"], sm["faces"]) | target

        def side_gain(u: str, removed: np.ndarray) -> float:
            """New material this side can remove inside the stock (part's bounding box or the blank), recognised as a feature or not."""
            return float((reach[u] & void & inside & ~removed).sum()) * grid.cell_volume

        none = np.zeros(grid.shape, bool)
        if primary == "auto":
            # primary setup = the side from which the most material can be removed (the modelled +Z wins near-ties)
            gains = {u: side_gain(u, none) for u in SIDES}
            primary = max(gains, key=gains.get)
            if gains["+Z"] >= 0.9 * gains[primary]:
                primary = "+Z"
        min_gain = max(self.min_gain, 0.003 * part_vol * grid.cell_volume)
        if os.environ.get("CAM_REL_THRESHOLDS", "0") == "1":
            # a part of a few tens of mm3 has less left for its second side than the absolute floor (evolution A04)
            min_gain = max(min(self.min_gain, 0.1 * part_vol * grid.cell_volume), 0.003 * part_vol * grid.cell_volume)
        area_enabled = os.environ.get("CAM_SETUP_AREA_GAIN", "0") == "1"
        links = _boundary_links(target) if area_enabled else None
        total_links = int(links.sum()) if area_enabled else 0
        area_admissions = []
        setups, removed = [], none.copy()
        order = [primary] + [u for u in SIDES if u != primary]
        feats, _ = candidates(primary, removed)
        setups.append({"up": primary, "features": feats, "gain_mm3": round(side_gain(primary, removed), 1), "before": removed.copy()})
        removed |= reach[primary] & void
        while len(setups) < self.max_setups:
            best = None
            for u in order[1:]:
                if any(s["up"] == u for s in setups):
                    continue
                feats, _ = candidates(u, removed)
                gain = side_gain(u, removed)
                new_links = int(links[reach[u] & void & inside & ~removed].sum()) if area_enabled else 0
                admissible = gain >= min_gain or (area_enabled and _area_admitted(gain, new_links, total_links))
                if admissible and (best is None or gain > best[2]):
                    best = (u, feats, gain)
            if best is None:
                break
            if area_enabled:
                n = int(links[reach[best[0]] & void & inside & ~removed].sum())
                area_admissions.append({"up": best[0], "new_boundary_links": n,
                                        "new_area_fraction": n / total_links if total_links else 0.0,
                                        "volume_admitted": best[2] >= min_gain})
            setups.append({"up": best[0], "features": best[1], "gain_mm3": round(best[2], 1), "before": removed.copy()})
            removed |= reach[best[0]] & void
        if sm is not None and os.environ.get("CAM_ALLOWANCE_FLIP", "1") == "1":
            # A blank face standing a few tenths above a flat face of the part: thinner than a voxel, so no gain is
            # seen, yet it is the whole face out of tolerance. Flip to face it from the side opposite a chosen one.
            pb, sb = mesh["bbox"], sm["bbox"]
            allow = {"+Z": sb[5] - pb[5], "-Z": pb[2] - sb[2], "+X": sb[3] - pb[3], "-X": pb[0] - sb[0], "+Y": sb[4] - pb[4], "-Y": pb[1] - sb[1]}
            flip = {"+Z": "-Z", "-Z": "+Z", "+X": "-X", "-X": "+X", "+Y": "-Y", "-Y": "+Y"}
            for s in list(setups):
                u = flip[s["up"]]
                if len(setups) >= self.max_setups or any(x["up"] == u for x in setups) or allow[u] <= 0.05:
                    continue
                if any(f["type"] == "planar_face" and f.get("role") == "top" for f in by_up[u]["features"]):
                    setups.append({"up": u, "features": [], "gain_mm3": 0.0, "before": removed.copy(), "allowance_mm": round(float(allow[u]), 3)})
                    log(f"[setup_planning] side {u}: {allow[u]:.2f} mm of blank over a flat face of the part: facing setup added")
        unreachable = void & inside & ~np.any([visible[s["up"]] for s in setups], axis=0)
        never = void & inside & ~np.any([visible[u] for u in SIDES], axis=0)
        # enclosed void that is part of the stock box but hidden from every chosen side
        part_cm3 = float(target.sum()) * grid.cell_volume / 1000
        out = {"primary": primary, "access_by_side": access, "grid_res_mm": round(grid.res, 3),
               "unreachable_cm3": round(float(unreachable.sum()) * grid.cell_volume / 1000, 3),
               "unreachable_from_any_axis_cm3": round(float(never.sum()) * grid.cell_volume / 1000, 3),
               # stock that stays attached no matter how the part is clamped, as a share of the finished part:
               # the machined result is wrong by this much, whatever the toolpaths do
               "unreachable_vs_part": round(float(unreachable.sum()) * grid.cell_volume / 1000 / max(part_cm3, 1e-9), 4),
               "part_cm3": round(part_cm3, 2), "setups": []}
        if area_enabled:
            out["area_admission"] = {"representation": "voxel target/void boundary, existing reach and stock masks",
                                     "total_boundary_links": total_links, "boundary_cell_mm2": grid.res ** 2,
                                     "threshold": 0.01, "secondary": area_admissions}
        if self.reach:
            cm3 = grid.cell_volume / 1000
            vis_chosen = np.any([visible[s["up"]] for s in setups], axis=0)
            got = np.any([reach[s["up"]] for s in setups], axis=0)
            any_reach = np.any([reach[u] for u in SIDES], axis=0)
            todo = void & inside
            # visible from a chosen side but deeper than any cutter reaches there (longer tools / extended holders / other blank)
            out["beyond_reach_cm3"] = round(float((todo & vis_chosen & ~got).sum()) * cm3, 3)
            out["beyond_reach_any_side_cm3"] = round(float((todo & ~any_reach & np.any([visible[u] for u in SIDES], axis=0)).sum()) * cm3, 3)
            total, best_all = float(todo.sum()) * cm3, float((todo & any_reach).sum()) * cm3
            out["reach"] = {"profile": [list(p) for p in self.reach], "removable_cm3": round(total, 3), "reachable_all_sides_cm3": round(best_all, 3),
                            "reached_cm3": round(float((todo & got).sum()) * cm3, 3),
                            "first_gain_by_side_cm3": {u: round(float((reach[u] & todo).sum()) * cm3, 3) for u in SIDES},
                            # greedy coverage of a monotone submodular function: >= (1 - 1/e) of the best plan with as many setups
                            "greedy_bound": "reached >= 0.632 * optimum(same number of setups); optimum <= reachable_all_sides"}
        if sm is not None:
            out["stock_cm3"] = round(float(inside.sum()) * grid.cell_volume / 1000, 2)
        for k, s in enumerate(setups, start=1):
            r = by_up[s["up"]]
            ids = {f["id"] for f in s["features"]}
            feats = []
            for f in r["features"]:
                keep = f["id"] in ids
                facing_only = False
                if k == 1 and f["type"] in EDGE:
                    keep = True  # facing, outer contour and edge features are done in the primary setup
                elif k > 1 and f["type"] == "planar_face" and f.get("role") == "top" and os.environ.get("CAM_SETUP_FACE_ALL", "1") == "1":
                    facing_only = not keep
                    keep = True  # this side's own top face: facing removes the blank's allowance there (evolution A03)
                if keep:
                    g = next((x for x in s["features"] if x["id"] == f["id"]), f)
                    if facing_only:
                        g = {**g, "facing_only": True}  # do not introduce new edge breaking on this restored face
                    if s.get("allowance_mm") and f["type"] == "planar_face" and f.get("role") == "top":
                        g = {**g, "allowance_mm": s["allowance_mm"], "facing_only": True}
                    feats.append({**g, "id": f"S{k}{f['id']}" if len(setups) > 1 else f["id"], "setup": k})
            surface = None
            if k > 1:  # what is still there before this setup, seen from its side (ideal result of the earlier setups)
                left = target | (void & ~s["before"] & (inside if sm is not None else True))
                _, _, _, lo, _ = cols[s["up"]]
                h = grid.top_view(left, frames[s["up"]], empty=float(lo[2]))
                f_npy = Path(workdir) / f"stock_surface_setup{k}.npy"
                Path(workdir).mkdir(parents=True, exist_ok=True)
                np.save(f_npy, h)
                surface = {"file": str(f_npy), "x0": float(lo[0]), "y0": float(lo[1]), "res": float(grid.res)}
            out["setups"].append({"index": k, "up": s["up"], "translation": r["translation"], "part": r["part"], "features": feats, "stock_surface": surface,
                                  "hole_patterns": r.get("hole_patterns", []), "gain_cm3": round(s["gain_mm3"] / 1000, 2),
                                  "allowance_mm": s.get("allowance_mm"),
                                  "summary": {t: sum(1 for f in feats if f["type"] == t) for t in MACHINED}})
            log(f"[setup_planning] setup {k}: side {s['up']}  removes {s['gain_mm3'] / 1000:.1f} cm3 new material  features {out['setups'][-1]['summary']}")
        log(f"[setup_planning] {len(setups)} setup(s); void not reachable from the chosen sides: {out['unreachable_cm3']} cm3 "
            f"(from no axis direction at all: {out['unreachable_from_any_axis_cm3']} cm3)")
        if self.reach:
            log(f"[setup_planning] reach (holder overhang, reach) {self.reach}: reached {out['reach']['reached_cm3']} of {out['reach']['removable_cm3']} cm3; "
                f"visible but beyond reach: {out['beyond_reach_cm3']} cm3 from the chosen sides, {out['beyond_reach_any_side_cm3']} cm3 from every side")
        if os.environ.get("CAM_TWO_SIDED_DRILL", "0") == "1":
            # Read-only recognised inputs for the post-baseline A34 coordinator.
            # No extra ideal setup is present while baseline CAM is generated.
            out["a34_recognition"] = by_up
            out["a34_max_setups"] = self.max_setups
        return out

    def _reachable_fit(self, grid: VoxelGrid, target: np.ndarray, frame: Frame, cinfo, vis: np.ndarray, band: float = 2.0) -> np.ndarray:
        """Visible voxels that one cutter of the library gets to: it fits the opening at that height (a disc of its radius
        inside the columns with no part above that height covers the voxel) and its reach clears the part under its
        holder. Heights are taken in `band` mm layers, the lower edge of each layer (conservative)."""
        from scipy import ndimage
        col, zk, (ncy, ncx), _, _ = cinfo
        res = grid.res
        top = grid.top_view(target, frame, empty=-1e9)
        flat = vis.ravel()
        idx = np.flatnonzero(flat)
        ok = np.zeros(col.shape, bool)
        if not idx.size:
            return vis & ok.reshape(grid.shape)
        z0 = float(zk[idx].min())
        layer = np.floor((zk - z0) / band).astype(np.int32)
        f = max(1, int(round(2.0 / res)))
        py, px = -ncy % f, -ncx % f
        coarse = np.pad(top, ((0, py), (0, px)), constant_values=-1e9).reshape((ncy + py) // f, f, (ncx + px) // f, f).max(axis=(1, 3))
        cres = res * f
        clear: dict[int, np.ndarray] = {}
        pad = int(np.ceil(max(c[0] for c in self.cutters) / res)) + 2
        # with CAM_NO_REACH the cutters' reach is unlimited and only the fit in the opening counts
        aspect = 0.0 if os.environ.get("CAM_NO_REACH", "1") == "1" else float(os.environ.get("CAM_REACH_ASPECT", "5"))
        for d, L in sorted(self.cutters, key=lambda c: -c[0]):
            # a long thin cutter does not hold a wall deeper than a few diameters, whatever its overall length
            L = min(L, aspect * d) if aspect > 0 else L
            rho = max(0.0, self.holder_d / 2 - d / 2)
            r = int(np.ceil(rho / cres))
            if r > 0:
                yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
                obst = ndimage.maximum_filter(coarse, footprint=(xx * xx + yy * yy) * cres * cres <= rho * rho + 1e-9, mode="constant", cval=-1e9)
            else:
                obst = coarse
            full = np.repeat(np.repeat(obst, f, axis=0), f, axis=1)[:ncy, :ncx].ravel()
            todo = idx[(~ok[idx]) & (full[col[idx]] - zk[idx] <= L)]
            if not todo.size:
                continue
            R = d / 2
            for k in np.unique(layer[todo]):
                if k not in clear:
                    # beyond the lattice is air: pad it so that a large cutter fits along the outer edge too
                    free = np.pad(top <= z0 + k * band + 1e-6, pad, constant_values=True)
                    clear[k] = ndimage.distance_transform_edt(free) * res
                centres = clear[k] >= R - 1e-9
                if not centres.any():
                    continue
                cover = (ndimage.distance_transform_edt(~centres) * res <= R + 1e-9)[pad:-pad, pad:-pad]
                sel = todo[layer[todo] == k]
                ok[sel] = cover.ravel()[col[sel]]
        return vis & ok.reshape(grid.shape)

    def _reachable(self, grid: VoxelGrid, target: np.ndarray, frame: Frame, cinfo, vis: np.ndarray) -> np.ndarray:
        """Visible voxels that some cutter of the reach profile gets to: the part within its holder overhang around the
        column is at most its reach above the voxel. The holder test runs on a ~2 mm lattice of max-pooled part tops
        (conservative by at most one coarse cell)."""
        from scipy import ndimage
        col, zk, (ncy, ncx), _, _ = cinfo
        top = grid.top_view(target, frame, empty=-1e9)
        f = max(1, int(round(2.0 / grid.res)))
        py, px = -ncy % f, -ncx % f
        c = np.pad(top, ((0, py), (0, px)), constant_values=-1e9).reshape((ncy + py) // f, f, (ncx + px) // f, f).max(axis=(1, 3))
        cres = grid.res * f
        ok = np.zeros(col.shape, bool)
        for rho, L in self.reach:
            r = int(np.ceil(rho / cres))
            if r > 0:
                yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
                obst = ndimage.maximum_filter(c, footprint=(xx * xx + yy * yy) * cres * cres <= rho * rho + 1e-9, mode="constant", cval=-1e9)
            else:
                obst = c
            full = np.repeat(np.repeat(obst, f, axis=0), f, axis=1)[:ncy, :ncx]
            ok |= full.ravel()[col] - zk <= L
        return vis & ok.reshape(grid.shape)
