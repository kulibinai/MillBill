"""Headless FreeCAD: write an axis-aligned box solid (the default stock) as STEP: {"lo": [x,y,z], "hi": [x,y,z], "out": path}."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import main  # noqa: E402

import FreeCAD  # noqa: E402
import Part  # noqa: E402


def run(req):
    lo, hi = req["lo"], req["hi"]
    box = Part.makeBox(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], FreeCAD.Vector(*lo))
    box.exportStep(req["out"])
    return {"out": req["out"], "volume": box.Volume}


main(run)
