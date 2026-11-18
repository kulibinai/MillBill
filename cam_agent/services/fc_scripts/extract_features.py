"""Headless FreeCAD: B-Rep machining-feature recognition for one or several machining sides.

Two modes:
  * rules (default): holes (through / blind, counterbore, countersink), pockets, slots, steps, chamfers, fillets,
    bosses, planar faces, outer profile - classified from face geometry and edge adjacency.
  * ml groups: request "groups" = [{"class": <feature class>, "faces": [face indices, 1-based], "bottom": [...]}] from a
    learned recogniser (AAGNet, see cam_agent/recognition); the face sets come from the network, all parameters
    (diameter, depth, floor, corner radius, ...) are still measured here on the exact B-Rep.
The part is rotated in memory so that the machining side is +Z (face numbering is identical for every side).
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_oriented, main, v  # noqa: E402

import FreeCAD  # noqa: E402
import Part  # noqa: E402

TOL = 1e-4


def face_normal(face):
    """Outward normal at the face centre (Face.normalAt accounts for the face orientation)."""
    u0, u1, v0, v1 = face.ParameterRange
    return face.normalAt((u0 + u1) / 2, (v0 + v1) / 2)


def kind(face):
    return face.Surface.__class__.__name__


def is_planar(face):
    return kind(face) == "Plane"


def is_cyl(face):
    return kind(face) == "Cylinder"


class Topo:
    """Face adjacency through shared edges (hash map instead of the O(F^2) isSame loops)."""

    def __init__(self, shape):
        self.shape = shape
        self.faces = shape.Faces
        self.edge_faces = {}
        for i, f in enumerate(self.faces, start=1):
            for e in f.Edges:
                self.edge_faces.setdefault(e.hashCode(), set()).add(i)

    def neighbours(self, i):
        out = set()
        for e in self.faces[i - 1].Edges:
            out |= self.edge_faces.get(e.hashCode(), set())
        out.discard(i)
        return out

    def shares_edge(self, i, j):
        return j in self.neighbours(i)


def concave_cylinder(face):
    """A cylindrical face is a hole / internal wall if its normal points towards the axis (material outside)."""
    n = face_normal(face)
    u0, u1, v0, v1 = face.ParameterRange
    p = face.valueAt((u0 + u1) / 2, (v0 + v1) / 2)
    to_axis = face.Surface.Center - p
    to_axis = to_axis - face.Surface.Axis * to_axis.dot(face.Surface.Axis)
    return n.dot(to_axis) > 0


def angular_span(face):
    u0, u1, v0, v1 = face.ParameterRange
    return abs(u1 - u0)


def group_cylinders(faces_idx, shape):
    groups = []
    for i in faces_idx:
        s = shape.Faces[i - 1].Surface
        for g in groups:
            gs = g["surface"]
            d = gs.Center - s.Center
            if abs(gs.Radius - s.Radius) < TOL and abs(abs(gs.Axis.dot(s.Axis)) - 1) < 1e-6 and (d - gs.Axis * d.dot(gs.Axis)).Length < 1e-3:
                g["faces"].append(i)
                break
        else:
            groups.append({"surface": s, "faces": [i]})
    return groups


def detect_holes(shape, bb, only=None):
    cand = [i + 1 for i, f in enumerate(shape.Faces) if is_cyl(f) and concave_cylinder(f) and (only is None or (i + 1) in only)]
    holes = []
    for g in group_cylinders(cand, shape):
        faces = [shape.Faces[i - 1] for i in g["faces"]]
        s = g["surface"]
        if sum(angular_span(f) for f in faces) < 2 * math.pi - 1e-3:
            continue  # partial concave cylinder: internal corner of a pocket, not a hole
        axis = s.Axis
        params = [(vtx.Point - s.Center).dot(axis) for f in faces for vtx in f.Vertexes]
        lo, hi = min(params), max(params)
        top_pt, bot_pt = s.Center + axis * hi, s.Center + axis * lo
        if axis.z < -1e-6 or (abs(axis.z) < 1e-6 and axis.dot(FreeCAD.Vector(1, 1, 0)) < 0):
            axis = axis * -1
            top_pt, bot_pt = bot_pt, top_pt
        inside = lambda pt: shape.isInside(pt, 1e-3, True)  # noqa: E731
        through = (not inside(top_pt + axis * 0.5)) and (not inside(bot_pt - axis * 0.5))
        bottom_type = None
        if not through:
            for f in shape.Faces:
                if kind(f) in ("Plane", "Cone") and f.distToShape(Part.Vertex(bot_pt))[0] < 1e-2 and f.BoundBox.isInside(bot_pt):
                    bottom_type = "flat" if kind(f) == "Plane" else "conical"
                    break
        vertical = abs(abs(axis.z) - 1) < 1e-3
        # coaxial cylinders of both sides grouped into one "hole" with material between them (brep2cam@savrasov)
        bore = Part.makeCylinder(0.95 * s.Radius, hi - lo, bot_pt, axis)
        if hi - lo > 1e-3 and bore.common(shape).Volume > 0.05 * bore.Volume:
            continue
        if vertical:  # entry covered from above (a recess under an overhang): not machinable from +Z
            col = Part.makeCylinder(0.95 * s.Radius, bb.ZMax - top_pt.z + 1.0, top_pt + FreeCAD.Vector(0, 0, 0.05), FreeCAD.Vector(0, 0, 1))
            if col.common(shape).Volume > 0.05 * col.Volume:
                vertical = False
        holes.append({"type": "hole", "diameter": round(2 * s.Radius, 3), "depth": round(hi - lo, 3), "through": bool(through), "bottom": bottom_type,
                      "axis": v(axis), "vertical": vertical, "top": v(top_pt), "bottom_point": v(bot_pt), "face_ids": g["faces"],
                      "top_z": round(top_pt.z, 3), "from_top_of_part": round(bb.ZMax - top_pt.z, 3)})
    return merge_hole_stacks(shape, holes)


def merge_hole_stacks(shape, holes):
    """Coaxial stacks: a wide shallow cylinder on top of a narrower hole is a counterbore (DIN 974); a concave 90 deg cone
    at the entry is a countersink (ISO 15065). The stack becomes one hole feature with subtype and both diameters."""
    out, used = [], set()
    vert = [h for h in holes if h["vertical"]]
    for h in sorted(vert, key=lambda x: x["diameter"]):
        if id(h) in used:
            continue
        for big in vert:
            if big is h or id(big) in used or big["diameter"] <= h["diameter"] + 0.2:
                continue
            same_axis = math.hypot(big["top"][0] - h["top"][0], big["top"][1] - h["top"][1]) < 1e-2
            if same_axis and abs(big["bottom_point"][2] - h["top"][2]) < 1e-2 and big["depth"] < 1.5 * big["diameter"]:
                h.update(subtype="counterbore", counterbore_diameter=big["diameter"], counterbore_depth=big["depth"], through_diameter=h["diameter"],
                         top=big["top"], top_z=big["top_z"], from_top_of_part=big["from_top_of_part"], depth=round(h["depth"] + big["depth"], 3),
                         face_ids=h["face_ids"] + big["face_ids"])
                used.add(id(big))
                break
    for i, f in enumerate(shape.Faces, start=1):
        if kind(f) != "Cone":
            continue
        s = f.Surface
        if abs(abs(s.Axis.z) - 1) > 1e-3:
            continue
        for h in vert:
            if math.hypot(s.Apex.x - h["top"][0], s.Apex.y - h["top"][1]) < 1e-2 and abs(f.BoundBox.ZMin - h["top_z"]) < 0.05 and not h.get("subtype"):
                half = math.degrees(s.SemiAngle)
                h.update(subtype="countersink", countersink_diameter=round(max(f.BoundBox.XLength, f.BoundBox.YLength), 3),
                         countersink_angle=round(2 * (90 - abs(half)) if abs(half) > 45 else 2 * abs(half), 1), through_diameter=h["diameter"],
                         top_z=round(f.BoundBox.ZMax, 3), depth=round(h["depth"] + f.BoundBox.ZLength, 3), face_ids=h["face_ids"] + [i])
    out = [h for h in holes if id(h) not in used]
    out.sort(key=lambda h: (h["top"][0], h["top"][1]))
    return out


def region_entry(shape, topo, i, bb, hole_face_ids):
    """Upward planar face i: top face, bottom face, pocket / slot / step floor, or intermediate face."""
    f = shape.Faces[i - 1]
    z = f.CenterOfMass.z
    wall_ids, wall_heights, radii = [], [], []
    for j in topo.neighbours(i):
        if j in hole_face_ids:
            continue
        g = shape.Faces[j - 1]
        if is_planar(g) and abs(face_normal(g).z) < 1e-3:
            wall_ids.append(j)
            wall_heights.append(g.BoundBox.ZMax - z)
        elif is_cyl(g) and abs(abs(g.Surface.Axis.z) - 1) < 1e-3:
            wall_ids.append(j)
            wall_heights.append(g.BoundBox.ZMax - z)
            if concave_cylinder(g):
                radii.append(g.Surface.Radius)
    fb = f.BoundBox
    entry = {"face": "Face%d" % i, "face_index": i, "z": round(z, 3), "area": round(f.Area, 2),
             "bbox": [round(fb.XMin, 3), round(fb.YMin, 3), round(fb.XMax, 3), round(fb.YMax, 3)], "size": [round(fb.XLength, 3), round(fb.YLength, 3)],
             "inner_loops": len(f.Wires) - 1, "walls": wall_ids, "wall_height": round(max(wall_heights), 3) if wall_heights else 0.0,
             "corner_radius": round(min(radii), 3) if radii else 0.0}
    if abs(z - bb.ZMax) < 1e-3:
        entry.update(type="planar_face", role="top")
    elif abs(z - bb.ZMin) < 1e-3:
        entry.update(type="planar_face", role="bottom")
    else:
        depth = bb.ZMax - z
        wall_edges = {e.hashCode() for j in wall_ids for e in shape.Faces[j - 1].Edges}
        closed = bool(wall_ids) and all(e.hashCode() in wall_edges for e in f.OuterWire.Edges)
        touches = (abs(fb.XMin - bb.XMin) < 1e-3 or abs(fb.XMax - bb.XMax) < 1e-3 or abs(fb.YMin - bb.YMin) < 1e-3 or abs(fb.YMax - bb.YMax) < 1e-3)
        w = min(fb.XLength, fb.YLength)
        entry.update(depth=round(depth, 3), min_width=round(w, 3))
        if closed and not touches:
            entry.update(type="pocket" if max(fb.XLength, fb.YLength) < 4 * w else "slot", open=False)
        elif touches and wall_ids:
            entry.update(type="step" if (fb.XLength >= 0.5 * bb.XLength or fb.YLength >= 0.5 * bb.YLength) else "slot", open=True)
        elif wall_ids and depth > 1.0:
            entry.update(type="pocket" if max(fb.XLength, fb.YLength) < 4 * w else "slot", open=False)  # holes break the wall loop
        else:
            entry.update(type="planar_face", role="intermediate")
    return entry


def detect_planar_regions(shape, topo, bb, hole_face_ids):
    out = []
    for i, f in enumerate(shape.Faces, start=1):
        if not is_planar(f) or abs(face_normal(f).z - 1) > 1e-3:
            continue
        ow = f.OuterWire.Edges
        if len(ow) == 1 and kind_curve(ow[0]) == "Circle" and topo.neighbours(i) & hole_face_ids:
            continue  # flat bottom of a blind hole
        out.append(region_entry(shape, topo, i, bb, hole_face_ids))
    return out


def kind_curve(edge):
    return edge.Curve.__class__.__name__


def detect_edge_features(shape, topo, part_bb, revolved_ids=()):
    """revolved_ids: faces a learned recogniser labelled chamfer / round; for these, conical faces are measured as
    (revolved) chamfers and toroidal faces as fillets. No toolpath strategy cuts a conical chamfer yet: such entries
    have normal None, so the chamfer strategy skips them instead of cutting a planar strip."""
    feats = []
    for i, f in enumerate(shape.Faces, start=1):
        cls = kind(f)
        fb = f.BoundBox
        if i in revolved_ids and cls in ("Cone", "Toroid"):
            s = f.Surface
            ax = getattr(s, "Axis", None)
            entry = {"face": "Face%d" % i, "face_index": i, "revolved": True,
                     "axis": [round(ax.x, 4), round(ax.y, 4), round(ax.z, 4)] if ax is not None else None,
                     "bbox": [round(fb.XMin, 3), round(fb.YMin, 3), round(fb.XMax, 3), round(fb.YMax, 3)],
                     "z_min": round(fb.ZMin, 3), "z_max": round(fb.ZMax, 3)}
            if cls == "Cone":
                semi = math.degrees(abs(s.SemiAngle))
                # chamfer leg along the axis = projection of the face's extent on the cone axis
                vs = [v.Point for v in f.Vertexes] or [fb.Center]
                proj = [p.dot(ax) for p in vs] if ax is not None else [fb.ZMin, fb.ZMax]
                size = max(proj) - min(proj) if len(vs) > 1 else fb.ZLength
                feats.append({"type": "chamfer", "size": round(size, 3), "angle_deg": round(90.0 - semi, 1), "normal": None, **entry})
            else:
                feats.append({"type": "fillet", "radius": round(s.MinorRadius, 3), **entry})
            continue
        if cls == "Cylinder" and not concave_cylinder(f) and abs(f.Surface.Axis.z) < 1e-3:
            feats.append({"type": "fillet", "radius": round(f.Surface.Radius, 3), "face": "Face%d" % i, "face_index": i})
        elif cls == "Cylinder" and not concave_cylinder(f) and abs(abs(f.Surface.Axis.z) - 1) < 1e-3:
            h = fb.ZLength
            if h > 0.5 and fb.ZMax > part_bb.ZMin + 1e-3 and angular_span(f) > math.pi:
                feats.append({"type": "boss", "diameter": round(2 * f.Surface.Radius, 3), "height": round(h, 3), "size": [round(fb.XLength, 3), round(fb.YLength, 3), round(h, 3)],
                              "top_z": round(fb.ZMax, 3), "face": "Face%d" % i, "face_index": i})
        elif cls == "Plane":
            n = face_normal(f)
            if 0.2 < abs(n.z) < 0.98 and n.z > 0:
                # a chamfer is a narrow bevel between two faces; a large inclined face is a sloped surface (cavity work)
                narrow = fb.ZLength <= 5.0 and fb.ZLength <= 0.25 * part_bb.ZLength + 1e-6
                entry = {"size": round(fb.ZLength, 3), "angle_deg": round(math.degrees(math.acos(min(1.0, abs(n.z)))), 1),
                         "normal": [round(n.x, 4), round(n.y, 4), round(n.z, 4)], "bbox": [round(fb.XMin, 3), round(fb.YMin, 3), round(fb.XMax, 3), round(fb.YMax, 3)],
                         "z_min": round(fb.ZMin, 3), "z_max": round(fb.ZMax, 3), "face": "Face%d" % i, "face_index": i}
                feats.append({"type": "chamfer" if narrow else "sloped_face", **entry})
    return feats


_PROFILE_D = (2, 3, 4, 5, 6, 8, 10, 12, 16, 20)


def _closing_area(face, r):
    """Area of the section after a morphological closing by a disc of radius r, on a raster of its outer wire.

    Pure numpy and scipy, so it always returns. None if the raster is impossible."""
    try:
        import numpy as np
        from scipy import ndimage
        pts = face.OuterWire.discretize(Deflection=0.05)
        xy = np.array([[p.x, p.y] for p in pts])
        lo = xy.min(0) - r - 1.0
        hi = xy.max(0) + r + 1.0
        res = max(0.02, float(max(hi - lo)) / 500.0)
        nx, ny = int((hi[0] - lo[0]) / res) + 1, int((hi[1] - lo[1]) / res) + 1
        from matplotlib.path import Path as MPath
        X, Y = np.meshgrid(lo[0] + np.arange(nx) * res, lo[1] + np.arange(ny) * res)
        inside = MPath(xy).contains_points(np.c_[X.ravel(), Y.ravel()]).reshape(ny, nx)
        # closing by a disc of radius r: dilate (distance to the section <= r), then erode (distance to outside > r)
        grown = ndimage.distance_transform_edt(~inside) * res <= r
        # strictly deeper than r + 2 cells: the raster band along a convex edge is not counted as a notch
        closed = ndimage.distance_transform_edt(grown) * res > r + 2.0 * res
        return face.Area + float((closed & ~inside).sum()) * res * res
    except Exception:
        return None


def contour_max_d(shape, bb):
    """Largest library endmill that follows the outer section without cutting a concave notch.

    A cutter of radius r takes a notch when the section, offset outward by r and back, grows.
    A convex outline does not grow, and the cap stays 20 mm. A section with a busy wire is left
    at 20 mm: the offset does not return on it, and that used to burn the whole 900 s budget."""
    import time
    z0, z1 = bb.ZMin, bb.ZMax
    if z1 - z0 < 1e-6:
        return 20.0
    limit, t0 = 20.0, time.time()
    try:
        for k in range(1, 6):
            if time.time() - t0 > 5.0:
                break
            z = z0 + (z1 - z0) * k / 6.0
            sec = shape.slice(FreeCAD.Vector(0, 0, 1), z)
            faces = []
            for w in (sec if isinstance(sec, list) else []):
                try:
                    if w.isClosed() and 3 <= len(w.Edges) <= 16:
                        faces.append(Part.Face(w))
                except Exception:
                    continue
            if not faces:
                continue
            face = max(faces, key=lambda f: f.Area)
            if face.Area < 1.0:
                continue
            base, tol = face.Area, max(1.0, 0.02 * face.Area)
            for d in _PROFILE_D:
                if d > limit + 1e-9 or time.time() - t0 > 5.0:
                    break
                # OCC offset can hang inside one call (p00678: 900 s); measure it on a 2D polygon instead
                area = _closing_area(face, d / 2.0)
                if area is None:
                    break
                if area - base > tol:
                    prev = [x for x in _PROFILE_D if x < d]
                    limit = min(limit, prev[-1] if prev else float(d))
                    break
    except Exception:
        return 20.0
    return float(limit)


def outer_profile(shape, bb):
    side = []
    for i, f in enumerate(shape.Faces, start=1):
        fb = f.BoundBox
        vertical = (is_planar(f) and abs(face_normal(f).z) < 1e-3) or (is_cyl(f) and abs(abs(f.Surface.Axis.z) - 1) < 1e-3 and not concave_cylinder(f))
        on_bbox = any(abs(a - c) < 1e-3 and abs(b - c) < 1e-3 for a, b, c in ((fb.XMin, fb.XMax, bb.XMin), (fb.XMin, fb.XMax, bb.XMax),
                                                                            (fb.YMin, fb.YMax, bb.YMin), (fb.YMin, fb.YMax, bb.YMax)))
        if vertical and (on_bbox or is_cyl(f)):
            side.append(i)
    rect = len([i for i in side if is_planar(shape.Faces[i - 1])]) >= 4
    return {"type": "outer_profile", "face_ids": side, "rectangular": rect,
            "size": [round(bb.XLength, 3), round(bb.YLength, 3), round(bb.ZLength, 3)],
            "profile_max_d": contour_max_d(shape, bb)}


# ---------------------------------------------------------------------------------------------- learned recogniser
ML_CLASS = {  # MFInstSeg / MFCAD++ class names -> harness feature types
    "chamfer": "chamfer", "through_hole": "hole", "blind_hole": "hole", "round": "fillet", "Oring": "pocket", "o_ring": "pocket",
    "triangular_passage": "pocket", "rectangular_passage": "pocket", "6sides_passage": "pocket",
    "triangular_through_slot": "slot", "rectangular_through_slot": "slot", "circular_through_slot": "slot",
    "rectangular_through_step": "step", "2sides_through_step": "step", "slanted_through_step": "step",
    "triangular_pocket": "pocket", "rectangular_pocket": "pocket", "6sides_pocket": "pocket", "circular_end_pocket": "pocket",
    "rectangular_blind_slot": "slot", "v_circular_end_blind_slot": "slot", "h_circular_end_blind_slot": "slot",
    "triangular_blind_step": "step", "circular_blind_step": "step", "rectangular_blind_step": "step",
}


def vertical_wall(face):
    """A wall of a pocket opening along +Z: vertical plane, or concave cylinder with its axis along Z."""
    if is_planar(face):
        return abs(face_normal(face).z) < 1e-3
    return is_cyl(face) and concave_cylinder(face) and abs(abs(face.Surface.Axis.z) - 1) < 1e-3


def features_from_groups(shape, topo, bb, groups):
    """Feature dicts for the face groups predicted by a learned recogniser; parameters are measured on the B-Rep."""
    out = []
    for g in groups:
        t = ML_CLASS.get(g.get("class"))
        ids = [int(i) for i in g.get("faces", []) if 1 <= int(i) <= len(shape.Faces)]
        if not t or not ids:
            continue
        idset = set(ids)
        if t == "hole":
            hs = detect_holes(shape, bb, only=idset)
            for h in hs:
                h["ml_class"] = g.get("class")
            out += hs
            continue
        if t == "chamfer" or t == "fillet":
            for e in detect_edge_features(shape, topo, bb, revolved_ids=idset):
                if e["face_index"] in idset and e["type"] in ("chamfer", "sloped_face", "fillet"):
                    out.append({**e, "ml_class": g.get("class")})  # geometry decides chamfer vs. sloped face vs. fillet
            continue
        ups = [i for i in ids if is_planar(shape.Faces[i - 1]) and abs(face_normal(shape.Faces[i - 1]).z - 1) < 1e-3]
        # the network's bottom faces are side-independent: on this side a floor must face the tool (+Z)
        bottoms = [int(i) for i in g.get("bottom", []) if int(i) in idset and int(i) in ups]
        floor = max(bottoms or ups or [None], key=lambda i: shape.Faces[i - 1].Area if i else 0)
        if not floor and not all(vertical_wall(shape.Faces[i - 1]) for i in ids):
            # no floor facing +Z and not a straight through passage: seen from this side the group is an undercut
            # (an external groove of a turned part) or opens to the other side, so it is not machinable here
            continue
        fbs = [shape.Faces[i - 1].BoundBox for i in ids]
        x0, y0 = min(b.XMin for b in fbs), min(b.YMin for b in fbs)
        x1, y1 = max(b.XMax for b in fbs), max(b.YMax for b in fbs)
        if floor:
            e = region_entry(shape, topo, floor, bb, set())
            e["type"] = t
        else:  # through passage / through slot: no floor, cut through the part
            e = {"face": None, "face_index": None, "z": round(min(b.ZMin for b in fbs), 3), "through": True, "corner_radius": 0.0, "inner_loops": 0, "walls": ids}
            e["type"] = t
        w = min(x1 - x0, y1 - y0)
        radii = [shape.Faces[i - 1].Surface.Radius for i in ids if is_cyl(shape.Faces[i - 1]) and concave_cylinder(shape.Faces[i - 1])]
        e.update(bbox=[round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)], size=[round(x1 - x0, 3), round(y1 - y0, 3)], min_width=round(w, 3),
                 depth=round(bb.ZMax - e["z"], 3), corner_radius=round(min(radii), 3) if radii else e.get("corner_radius", 0.0),
                 open=t in ("step",) or "through" in str(g.get("class")), ml_class=g.get("class"), ml_faces=ids)
        out.append(e)
    return out


# ---------------------------------------------------------------------------------------------- main
def run(req):
    """req: {"path", "up"} or {"path", "ups": [sides]}; optional "groups" from a learned recogniser. Groups are face
    indices of the original STEP; orienting the part keeps the face order, so they apply to every side."""
    if req.get("ups"):
        return {"by_up": {u: analyse(req["path"], u, req.get("groups")) for u in req["ups"]}}
    return analyse(req["path"], req.get("up", "+Z"), req.get("groups"))


def analyse(path, up, groups):
    shape, translation = load_oriented(path, up)
    bb = shape.optimalBoundingBox()  # BoundBox of an untriangulated B-rep is loose (tori, splines); from brep2cam@savrasov
    topo = Topo(shape)
    rule_holes = detect_holes(shape, bb)
    hole_face_ids = {i for h in rule_holes for i in h["face_ids"]}
    regions = detect_planar_regions(shape, topo, bb, hole_face_ids)
    edges = detect_edge_features(shape, topo, bb)
    recogniser = "rules"
    if groups:
        ml = features_from_groups(shape, topo, bb, groups)
        if ml:
            recogniser = "ml"
            covered = {i for f in ml for i in (f.get("face_ids") or f.get("ml_faces") or ([f["face_index"]] if f.get("face_index") else []))}
            # learned instances replace the rule-based holes / pockets / slots / steps / chamfers / fillets they cover
            rule_holes = [h for h in rule_holes if not set(h["face_ids"]) & covered]
            regions = [r for r in regions if r["type"] == "planar_face" or r["face_index"] not in covered]
            edges = [e for e in edges if e["face_index"] not in covered]
            holes_ml = [f for f in ml if f["type"] == "hole"]
            regions += [f for f in ml if f["type"] in ("pocket", "slot", "step")]
            edges += [f for f in ml if f["type"] in ("chamfer", "fillet", "sloped_face")]
            rule_holes += holes_ml
    features, fid = [], 0
    for h in rule_holes:
        fid += 1
        features.append({"id": "F%02d" % fid, **{k: val for k, val in h.items() if k != "face_ids"}, "faces": ["Face%d" % i for i in h["face_ids"]]})
    for r in regions:
        fid += 1
        faces = ([r["face"]] if r.get("face") else []) + ["Face%d" % j for j in r.get("walls", [])]
        features.append({"id": "F%02d" % fid, **{k: val for k, val in r.items() if k not in ("walls", "ml_faces")}, "faces": faces})
    for e in edges:
        fid += 1
        features.append({"id": "F%02d" % fid, **e, "faces": [e["face"]]})
    fid += 1
    op = outer_profile(shape, bb)
    features.append({"id": "F%02d" % fid, **{k: val for k, val in op.items() if k != "face_ids"}, "faces": ["Face%d" % i for i in op["face_ids"]]})
    patterns = {}
    for h in rule_holes:
        patterns.setdefault((h["diameter"], h["depth"], h["through"], h["vertical"], h.get("subtype")), []).append(h["top"])
    hole_patterns = [{"diameter": k[0], "depth": k[1], "through": k[2], "vertical": k[3], "subtype": k[4], "count": len(p), "positions": p} for k, p in patterns.items()]
    count = lambda t: sum(1 for f in features if f["type"] == t)  # noqa: E731
    return {"file": path, "up": up, "translation": translation, "recogniser": recogniser,
            "part": {"bbox": [round(bb.XMin, 3), round(bb.YMin, 3), round(bb.ZMin, 3), round(bb.XMax, 3), round(bb.YMax, 3), round(bb.ZMax, 3)],
                     "size": [round(bb.XLength, 3), round(bb.YLength, 3), round(bb.ZLength, 3)], "volume": round(shape.Volume, 2), "area": round(shape.Area, 2),
                     "faces": len(shape.Faces), "edges": len(shape.Edges), "solids": len(shape.Solids), "valid": bool(shape.isValid())},
            "features": features, "hole_patterns": hole_patterns,
            "summary": {t: count(t) for t in ("hole", "pocket", "slot", "step", "chamfer", "fillet", "boss", "sloped_face")},
            "freecad_version": ".".join(FreeCAD.Version()[:3])}


main(run)
