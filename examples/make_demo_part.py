"""Headless FreeCAD script: builds a demo bracket (plate with a closed pocket, 4 through holes, 2 blind holes,
a slot and chamfered top edges) and writes STEP + FCStd to $CAM_DEMO_OUT (cam_agent demo: <out>/demo_part),
or next to this file when it is not set.

Run:  freecadcmd examples/make_demo_part.py
"""
import os

import FreeCAD
import Part

out_dir = os.environ.get("CAM_DEMO_OUT") or os.path.dirname(os.path.abspath(__file__))
os.makedirs(out_dir, exist_ok=True)
doc = FreeCAD.newDocument("demo")
L, W, H = 120.0, 80.0, 20.0
shape = Part.makeBox(L, W, H)
# closed pocket 50x30x8 with 5 mm corner radii, centred
pocket = Part.makeBox(50, 30, 8, FreeCAD.Vector(35, 25, H - 8))
pocket = pocket.makeFillet(5, [e for e in pocket.Edges if abs(e.tangentAt(e.FirstParameter).z) > 0.99])
shape = shape.cut(pocket)
# 4 through holes 6.6 mm (M6 clearance)
for x, y in [(10, 10), (110, 10), (10, 70), (110, 70)]:
    shape = shape.cut(Part.makeCylinder(3.3, H + 2, FreeCAD.Vector(x, y, -1)))
# 2 blind holes 5.0 mm x 12 deep (M6 tap drill) on the top face
for x, y in [(25, 40), (95, 40)]:
    shape = shape.cut(Part.makeCylinder(2.5, 12, FreeCAD.Vector(x, y, H - 12)))
# open slot 8 wide x 6 deep from the +X edge
shape = shape.cut(Part.makeBox(25, 8, 6, FreeCAD.Vector(L - 25, 36, H - 6)))
# 1 mm chamfer on the top outer edges
top_edges = [e for e in shape.Edges if abs(e.BoundBox.ZMin - H) < 1e-6 and abs(e.BoundBox.ZMax - H) < 1e-6
             and (abs(e.BoundBox.XLength - L) < 1e-6 or abs(e.BoundBox.YLength - W) < 1e-6)]
shape = shape.makeChamfer(1.0, top_edges)
obj = doc.addObject("Part::Feature", "Bracket")
obj.Shape = shape
doc.recompute()
shape.exportStep(os.path.join(out_dir, "demo_bracket.step"))
doc.saveAs(os.path.join(out_dir, "demo_bracket.FCStd"))
print("volume", round(shape.Volume, 1), "faces", len(shape.Faces))
print("wrote", os.path.join(out_dir, "demo_bracket.step"))
