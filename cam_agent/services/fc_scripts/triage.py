"""Headless FreeCAD: quick geometric triage of a STEP file (is it a candidate for 3-axis, single-setup milling?)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import main  # noqa: E402

import FreeCAD  # noqa: E402,F401
import Part  # noqa: E402


def run(req):
    shape = Part.Shape()
    shape.read(req["path"])
    solids = shape.Solids
    if solids:
        main_solid = max(solids, key=lambda s: s.Volume)
    else:
        main_solid = shape
    bb = main_solid.optimalBoundingBox()  # BoundBox of an untriangulated B-rep is loose (tori, splines); from brep2cam@savrasov
    kinds, area = {}, {}
    for f in main_solid.Faces:
        k = f.Surface.__class__.__name__
        kinds[k] = kinds.get(k, 0) + 1
        area[k] = area.get(k, 0.0) + f.Area
    total = sum(area.values()) or 1.0
    diag = (bb.XLength ** 2 + bb.YLength ** 2 + bb.ZLength ** 2) ** 0.5
    defl = max(0.05, diag / 1500.0)
    verts, faces = main_solid.tessellate(defl)
    return {"solids": len(solids), "solid_volumes": sorted([round(s.Volume, 1) for s in solids], reverse=True)[:6], "valid": bool(main_solid.isValid()),
            "bbox": [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax], "size": [round(bb.XLength, 2), round(bb.YLength, 2), round(bb.ZLength, 2)],
            "volume": main_solid.Volume if solids else 0.0, "faces": len(main_solid.Faces), "face_kinds": kinds,
            "area_share": {k: round(v / total, 3) for k, v in sorted(area.items(), key=lambda kv: -kv[1])},
            "vertices": [[round(v.x, 4), round(v.y, 4), round(v.z, 4)] for v in verts], "tri": [list(f) for f in faces], "deflection": defl}


main(run)
