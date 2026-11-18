"""Headless FreeCAD: index, centroid and area of every face in the part's ORIGINAL coordinates (to match the faces
predicted by a learned recogniser that read the same STEP)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_oriented, main  # noqa: E402


def run(req):
    shape, _ = load_oriented(req["path"], "+Z", [0.0, 0.0, 0.0])
    return {"faces": [{"index": i, "centroid": [f.CenterOfMass.x, f.CenterOfMass.y, f.CenterOfMass.z], "area": f.Area}
                      for i, f in enumerate(shape.Faces, start=1)]}


main(run)
