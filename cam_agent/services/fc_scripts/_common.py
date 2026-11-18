"""Helpers shared by the headless FreeCAD scripts (executed inside freecadcmd)."""
import json
import os
import sys
import traceback


def read_request():
    with open(os.environ["CAM_FC_REQUEST"]) as f:
        return json.load(f)


def write_response(obj):
    with open(os.environ["CAM_FC_RESPONSE"], "w") as f:
        json.dump(obj, f, default=str)


def main(fn):
    try:
        req = read_request()
        write_response(fn(req))
    except Exception as e:  # noqa: BLE001
        write_response({"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()})
    finally:
        sys.stdout.flush()


def v(x, nd=3):
    return [round(x.x, nd), round(x.y, nd), round(x.z, nd)]


# machining side -> rotation that maps that direction onto +Z (axis, degrees); identical table in services/voxel.py
ROT = {"+Z": None, "-Z": ((1, 0, 0), 180), "+X": ((0, 1, 0), -90), "-X": ((0, 1, 0), 90), "+Y": ((1, 0, 0), 90), "-Y": ((1, 0, 0), -90)}


def load_oriented(path, up="+Z", translation=None):
    """Load a STEP/IGES/BREP/FCStd solid, rotate it in memory so that `up` becomes +Z and move its bounding-box
    minimum to the origin. In-memory rotation keeps the face/edge numbering, so 'FaceN' names are the same in every
    setup. Returns (shape, [tx, ty, tz]) where the translation was applied after the rotation."""
    import FreeCAD
    import Part
    ext = os.path.splitext(path)[1].lower()
    if ext == ".fcstd":
        doc = FreeCAD.openDocument(path)
        solids = [o for o in doc.Objects if hasattr(o, "Shape") and o.Shape.Solids]
        shape = solids[-1].Shape.copy()
    else:
        shape = Part.Shape()
        shape.read(path)
    if shape.Solids:
        shape = shape.Solids[0] if len(shape.Solids) == 1 else max(shape.Solids, key=lambda s: s.Volume)
    r = ROT[up]
    if r:
        shape.rotate(FreeCAD.Vector(0, 0, 0), FreeCAD.Vector(*r[0]), r[1])
    if translation is None:
        bb = shape.optimalBoundingBox()  # BoundBox of an untriangulated B-rep is loose (tori, splines); from brep2cam@savrasov
        translation = [-bb.XMin, -bb.YMin, -bb.ZMin]
    shape.translate(FreeCAD.Vector(*translation))
    return shape, [float(t) for t in translation]
