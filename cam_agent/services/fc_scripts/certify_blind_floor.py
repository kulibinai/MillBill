"""A35: certify one supplied blind bore and planar floor, without discovery."""
import math
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from _common import load_oriented, main
import FreeCAD as F
import Part


def run(req):
    h=req['hole']
    if h.get('through') or not h.get('vertical',True) or len(h.get('faces',[]))!=1:
        return dict(accepted=False,reason='not single vertical blind cylinder')
    shape,_=load_oriented(req['path'],req['up'],req.get('translation'))
    if not shape.isValid():
        return dict(accepted=False,reason='invalid target')
    wall=shape.getElement(h['faces'][0]); surf=wall.Surface
    if surf.__class__.__name__!='Cylinder' or abs(abs(surf.Axis.z)-1)>1e-7:
        return dict(accepted=False,reason='not axial cylinder')
    u0,u1,v0,v1=wall.ParameterRange
    if abs(u1-u0)<2*math.pi-1e-6:
        return dict(accepted=False,reason='incomplete cylinder')
    r=surf.Radius; x,y=surf.Center.x,surf.Center.y
    floor,top=wall.BoundBox.ZMin,wall.BoundBox.ZMax
    if abs(2*r-h['diameter'])>.02 or abs(top-floor-h['depth'])>.02:
        return dict(accepted=False,reason='feature/geometry mismatch')
    planes=[]
    for i,f in enumerate(shape.Faces,1):
        if f.Surface.__class__.__name__!='Plane' or abs(f.BoundBox.ZMin-floor)>1e-6 or abs(f.BoundBox.ZMax-floor)>1e-6:
            continue
        a,b,c,d=f.ParameterRange; normal=f.normalAt((a+b)/2,(c+d)/2)
        if normal.z>1-1e-7 and abs(f.Area-math.pi*r*r)<.001 and any(e.isSame(w) for e in f.Edges for w in wall.Edges):
            planes.append(i)
    if len(planes)!=1:
        return dict(accepted=False,reason='no unique complete planar disk floor')
    entry=max(shape.optimalBoundingBox().ZMax,top)
    if req.get('stock'):
        stock,_=load_oriented(req['stock'],req['up'],req.get('translation'))
        entry=max(entry,stock.optimalBoundingBox().ZMax)
    envelope=Part.makeCylinder(r-.005,entry-floor-.001,F.Vector(x,y,floor+.001))
    volume=envelope.common(shape).Volume
    return dict(accepted=volume<1e-7,reason='exact cylinder floor and entry',
        target_intersection_mm3=volume,x=x,y=y,top_z=entry,floor_z=floor,
        diameter=2*r-.01,depth=entry-floor,floor_face='Face%d'%planes[0])


main(run)
