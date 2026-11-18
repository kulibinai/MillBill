"""Headless FreeCAD: rotate a part so that the chosen machining direction becomes +Z, move it to the origin, export STEP.

"stock" / "stock_out": the user's blank gets the same rotation and the PART's translation, so the two stay registered.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import main  # noqa: E402

import FreeCAD  # noqa: E402
import Part  # noqa: E402

ROT = {"+Z": None, "-Z": ((1, 0, 0), 180), "+X": ((0, 1, 0), -90), "-X": ((0, 1, 0), 90), "+Y": ((1, 0, 0), 90), "-Y": ((1, 0, 0), -90)}


def _read_rotated(path, r):
    shape = Part.Shape()
    shape.read(path)
    if shape.Solids:
        shape = max(shape.Solids, key=lambda s: s.Volume)
    if r:
        shape = shape.copy()
        shape.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(*r[0]), r[1])
    return shape


def run(req):
    r = ROT[req["up"]]
    shape = _read_rotated(req["path"], r)
    bb = shape.optimalBoundingBox()  # BoundBox of an untriangulated B-rep is loose (tori, splines); from brep2cam@savrasov
    move = FreeCAD.Vector(-bb.XMin, -bb.YMin, -bb.ZMin)
    shape.translate(move)
    shape.exportStep(req["out"])
    bb = shape.optimalBoundingBox()
    out = {"out": req["out"], "size": [round(bb.XLength, 3), round(bb.YLength, 3), round(bb.ZLength, 3)], "solids": len(shape.Solids), "volume": shape.Volume}
    if req.get("stock"):
        stock = _read_rotated(req["stock"], r)
        stock.translate(move)
        stock.exportStep(req["stock_out"])
        out["stock_out"] = req["stock_out"]
    return out


main(run)
