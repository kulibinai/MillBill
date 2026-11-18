"""Headless FreeCAD: area-weighted random points ON THE B-REP faces of a part, with exact outward normals (the samples
of the surface metric, services/verify.py).

Every face gets round(n * its area / total area) points (largest remainders). A point is placed uniformly on a random
triangle of the face's tessellation (picked by area), then projected onto the face's surface (Surface.parameter ->
Surface.value; points on a plane are already on it); the normal is the face's normal there (Face.normalAt), turned
outward per face (a vote of Shape.isInside a micron along it). Weight of a point = its face's area / its face's points.
Points of a face whose projection fails keep the triangle point and the triangle normal (counted in "fallback"); a face
without triangles gets no points ("dropped": its area is not in the metric).

request: {"path", "n", "seed", "deflection", "raw": true | "up", "translation", "out": <prefix>}
writes <out>.P.bin, <out>.N.bin (float64 x 3), <out>.w.bin (float64), <out>.face.bin (int32); returns counts and times.
"""
import array
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_oriented, main  # noqa: E402

import FreeCAD  # noqa: E402,F401
import Part  # noqa: E402,F401


def _dump(path, typecode, vals):
    with open(path, "wb") as f:
        array.array(typecode, vals).tofile(f)


def run(req):
    t0 = time.time()
    if req.get("raw"):
        shape, translation = load_oriented(req["path"], "+Z", [0.0, 0.0, 0.0])
    else:
        shape, translation = load_oriented(req["path"], req.get("up", "+Z"), req.get("translation"))
    n, seed, defl = int(req.get("n", 600000)), int(req.get("seed", 0)), float(req.get("deflection", 0.025))
    faces = shape.Faces
    areas = [f.Area for f in faces]
    total = sum(areas)
    coverage = None
    if os.environ.get("CAM_BREP_COVERAGE_WARN", "0") == "1":
        coverage = {"shape_valid": bool(shape.isValid()), "shape_closed": bool(shape.isClosed()),
                    "invalid_face_ids": [i + 1 for i, f in enumerate(faces) if not f.isValid()],
                    "face_id_base": 1, "zero_triangle_faces": [], "unsampled_faces": [], "diagnostic_errors": [], "total_area_mm2": total}
    raw = [n * a / total for a in areas]
    cnt = [int(math.floor(x)) for x in raw]
    for i in sorted(range(len(faces)), key=lambda i: raw[i] - cnt[i], reverse=True)[: n - sum(cnt)]:
        cnt[i] += 1
    P, N, W, F = array.array("d"), array.array("d"), array.array("d"), array.array("i")
    fallback, dropped, flipped, by_type = 0, 0, 0, {}
    for fi, (f, k) in enumerate(zip(faces, cnt)):
        if k == 0:
            if coverage is not None:
                row = {"face_id": fi + 1, "area_mm2": areas[fi]}
                coverage["unsampled_faces"].append(row)
                try:
                    _, tris = f.tessellate(defl)
                    if not tris:
                        coverage["zero_triangle_faces"].append(row)
                except Exception as e:  # diagnostics for unallocated faces must not fail sampling
                    coverage["diagnostic_errors"].append({"face_id": fi + 1, "error": type(e).__name__})
            continue
        kind = f.Surface.__class__.__name__
        tf = time.time()
        pts, tris = f.tessellate(defl)
        ta = [((pts[b] - pts[a]).cross(pts[c] - pts[a])).Length / 2 for a, b, c in tris]
        if not tris or sum(ta) <= 0:
            dropped += k  # a degenerate face (no triangles): its points are not placed
            if coverage is not None:
                row = {"face_id": fi + 1, "area_mm2": areas[fi]}
                coverage["unsampled_faces"].append(row)
                if not tris:
                    coverage["zero_triangle_faces"].append(row)
            continue
        rng = random.Random(seed * 100003 + fi)
        pick = rng.choices(range(len(tris)), weights=ta, k=k)
        planar = kind == "Plane"
        n_plane = None
        fp, fn = [], []
        for t in pick:
            a, b, c = (pts[i] for i in tris[t])
            s1, r2 = math.sqrt(rng.random()), rng.random()
            p = a * (1 - s1) + b * (s1 * (1 - r2)) + c * (s1 * r2)
            try:
                if planar:
                    if n_plane is None:
                        u, v = f.Surface.parameter(p)
                        n_plane = f.normalAt(u, v)
                    nv = n_plane
                else:
                    u, v = f.Surface.parameter(p)
                    p = f.Surface.value(u, v)
                    nv = f.normalAt(u, v)
            except Exception:  # noqa: BLE001  (projection failed: the triangle point and normal)
                nv = (b - a).cross(c - a)
                nv = nv * (1.0 / max(nv.Length, 1e-12))
                fallback += 1
            fp.append(p)
            fn.append(nv)
        if os.environ.get("CAM_NORMAL_VOTE2", "0") == "1":
            # both sides, 0.01 mm off the face: a 1 um probe sits inside the STEP's own tolerance and reads "inside" on
            # either side of a face; only a probe that is inside ahead and outside behind says the normal points in (evolution A06)
            e = 0.01
            sides = [(shape.isInside(fp[i] + fn[i] * e, 1e-6, False), shape.isInside(fp[i] - fn[i] * e, 1e-6, False)) for i in range(min(9, k))]
            flip = sum(s == (True, False) for s in sides) > sum(s == (False, True) for s in sides)
        else:
            votes = [shape.isInside(fp[i] + fn[i] * 1e-3, 1e-9, False) for i in range(min(9, k))]
            flip = sum(votes) * 2 > len(votes)  # the normal points into the solid: outward is the opposite
        flipped += bool(flip)
        sgn = -1.0 if flip else 1.0
        for p, nv in zip(fp, fn):
            P.extend((p.x, p.y, p.z))
            N.extend((sgn * nv.x, sgn * nv.y, sgn * nv.z))
            F.append(fi)
        W.extend([areas[fi] / k] * k)
        bt = by_type.setdefault(kind, {"faces": 0, "points": 0, "area": 0.0, "seconds": 0.0})
        bt["faces"] += 1
        bt["points"] += k
        bt["area"] += areas[fi]
        bt["seconds"] += time.time() - tf
    out = req["out"]
    for suffix, code, vals in (("P", "d", P), ("N", "d", N), ("w", "d", W), ("face", "i", F)):
        _dump(f"{out}.{suffix}.bin", code, vals)
    info = {"out": out, "samples": len(W), "faces": len(faces), "area": total, "fallback": fallback, "dropped": dropped, "faces_flipped": flipped,
            "by_type": {k: {**v, "area": round(v["area"], 1), "seconds": round(v["seconds"], 1)} for k, v in by_type.items()},
            "translation": translation, "seconds": round(time.time() - t0, 1)}
    if coverage is not None:
        missing = sum(r["area_mm2"] for r in coverage["unsampled_faces"])
        coverage.update(unsampled_area_mm2=missing, represented_area_mm2=total - missing,
                        incomplete=missing > 0, warnings=[])
        if not coverage["shape_valid"]:
            coverage["warnings"].append("Target B-Rep is invalid; surface samples do not establish shape validity.")
        if missing > 0:
            coverage["warnings"].append("Incomplete surface sample coverage: unsampled face area is excluded from the metric; score describes represented surface only.")
        if coverage["diagnostic_errors"]:
            coverage["warnings"].append("Tessellation diagnostics failed for an unallocated face; zero-triangle diagnostics are incomplete.")
        info["brep_coverage"] = coverage
    return info


main(run)
