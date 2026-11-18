"""A34 safety validation of supplied recognised cylindrical faces; no discovery."""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_oriented, main, v
import FreeCAD as F
import Part


def run(req):
    shape, _ = load_oriented(req["path"], req["up"], req["translation"])
    if not shape.isValid() or shape.Volume <= 0:
        return {"accepted": False, "reason": "invalid/empty target"}
    h = req["hole"]
    top, bot = F.Vector(*h["top"]), F.Vector(*h["bottom_point"])
    r, length = h["diameter"] / 2, h["depth"]
    if not h.get("through") or not h.get("vertical") or h.get("subtype") not in (None, "hole"):
        return {"accepted": False, "reason": "not simple vertical through"}
    if abs(top.x-bot.x)+abs(top.y-bot.y)+abs(top.z-bot.z-length) > 1e-5:
        return {"accepted": False, "reason": "endpoint/depth mismatch"}
    spans = []
    for name in h["faces"]:
        face = shape.Faces[int(name[4:])-1]
        surf = face.Surface
        if surf.__class__.__name__ != "Cylinder" or abs(surf.Radius-r) > 1e-5 or abs(abs(surf.Axis.z)-1) > 1e-6:
            return {"accepted": False, "reason": "nonmatching cylinder"}
        pts = [p.Point for p in face.Vertexes]
        if any(math.hypot(surf.Center.x-top.x, surf.Center.y-top.y) > 1e-5 for p in pts):
            return {"accepted": False, "reason": "axis mismatch"}
        u0,u1,v0,v1=face.ParameterRange
        width=abs(u1-u0)
        point=face.valueAt(u0,(v0+v1)/2)
        angle=math.atan2(point.y-top.y,point.x-top.x)
        start=(angle-width if surf.Axis.z<0 else angle)%(2*math.pi)
        angular=[(0,2*math.pi)] if width>=2*math.pi-1e-6 else (
            [(start,start+width)] if start+width<=2*math.pi else [(start,2*math.pi),(0,start+width-2*math.pi)])
        spans.append((min(p.z for p in pts),max(p.z for p in pts),angular))
    cuts = sorted(set([bot.z, top.z] + [max(bot.z,min(top.z,x)) for lo,hi,a in spans for x in (lo,hi)]))
    for lo,hi in zip(cuts,cuts[1:]):
        if hi-lo <= 1e-5:
            continue
        end=0
        for a,b in sorted(x for a0,a1,angles in spans if a0<=(lo+hi)/2<=a1 for x in angles):
            if a>end+1e-5:
                return {"accepted": False, "reason": "incomplete angular wall union"}
            end=max(end,b)
        if end<2*math.pi-1e-5:
            return {"accepted": False, "reason": "disconnected or incomplete cylindrical walls"}
    # The complete radius, bore and both approaches through the supplied stock
    # bound must miss target material. Boundary contact is not removed volume.
    z0,z1 = req["stock_z"]
    envelope = Part.makeCylinder(r,z1-z0,F.Vector(top.x,top.y,z0))
    volume = float(envelope.common(shape).Volume)
    return {"accepted": volume <= 1e-7, "target_intersection_mm3": volume,
            "spans": spans, "entry_top": v(top), "entry_bottom": v(bot)}


main(run)
