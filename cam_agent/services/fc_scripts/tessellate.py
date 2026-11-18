"""Headless FreeCAD: tessellate a STEP/FCStd part into a triangle mesh (for rendering / verification).

"frame_of": <part path> places this shape (a user stock) with the part's translation instead of its own bounding box.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_oriented, main  # noqa: E402

import FreeCAD  # noqa: E402,F401
import Part  # noqa: E402


def run(req):
    # raw = original model coordinates (the voxel model and the setup frames are defined relative to them)
    if req.get("raw"):
        shape, translation = load_oriented(req["path"], "+Z", [0.0, 0.0, 0.0])
    else:
        up, translation = req.get("up", "+Z"), req.get("translation")
        if translation is None and req.get("frame_of"):
            _, translation = load_oriented(req["frame_of"], up, None)
        shape, translation = load_oriented(req["path"], up, translation)
    verts, faces = shape.tessellate(float(req.get("deflection", 0.1)))
    bb = shape.BoundBox
    return {"vertices": [[round(v.x, 4), round(v.y, 4), round(v.z, 4)] for v in verts], "faces": [list(f) for f in faces],
            "bbox": [bb.XMin, bb.YMin, bb.ZMin, bb.XMax, bb.YMax, bb.ZMax], "volume": shape.Volume, "translation": translation}


main(run)
