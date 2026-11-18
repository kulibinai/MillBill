"""Generic height-map machining: cavity roughing and surface finishing without feature recognition.

Everything is computed in the setup frame (part minimum at the origin, tool axis = +Z) from the height-map of the
target as seen from above, H(x, y). For a flat endmill of radius r the tool centre may go down to

    A(x, y) = max of H over the tool disk around (x, y)          (grey dilation of H)

so every path generated on or above A is gouge-free by construction, and undercuts are ignored automatically.

  roughing : Z levels from the stock top, step = DOC. At level z the tool may be wherever A + stock_to_leave <= z.
             That region is cleared with a zig-zag at the step-over, then its boundary is followed once (walls).
  finishing: the same without stock to leave: every plateau (floor) level is cleared and contoured, walls between
             plateaus are contoured at a fine Z step; sloped / freeform areas get a drop-cutter raster.

G-code uses mm; feeds are written in mm/s because the text goes through FreeCAD's Custom operation (internal units).
"""
from __future__ import annotations

import math
import os

import numpy as np
from contourpy import contour_generator
from scipy import ndimage


def disk(radius_cells: float) -> np.ndarray:
    n = int(math.ceil(radius_cells))
    jj, ii = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    return (ii ** 2 + jj ** 2) <= radius_cells ** 2 + 1e-9


def _bdil(mask: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """binary_dilation by a disk; CAM_CPU_FAST=1: running maxima along the rows (same result)."""
    from .cpu_fast import dilate, enabled
    return dilate(mask, fp, binary=True) if enabled() else ndimage.binary_dilation(mask, structure=fp)


def _gdil(H: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """grey_dilation by a disk (edges repeated); CAM_CPU_FAST=1: running maxima along the rows (same result)."""
    from .cpu_fast import dilate, enabled
    return dilate(H, fp) if enabled() else ndimage.grey_dilation(H, footprint=fp, mode="nearest")


class HeightField:
    """Target height-map on a regular grid covering the stock (setup frame)."""

    def __init__(self, verts, faces, stock_bbox, res: float, holes: list[dict] | None = None, outside: str = "cut", pad: float = 14.0,
                 stock_surface: dict | None = None, stock_mesh: dict | None = None) -> None:
        from .raster import Grid, rasterize
        self.res = res
        self.stock = list(stock_bbox)
        x0, y0, zb, x1, y1, zt = stock_bbox
        # the grid extends beyond the stock by `pad` (> largest tool radius) so that the cutter can travel around the part
        self.x0, self.y0, self.zb, self.x1, self.y1, self.zt = x0 - pad, y0 - pad, zb, x1 + pad, y1 + pad, zt
        self.grid = Grid(self.x0, self.y0, self.x1, self.y1, res)
        self.nx, self.ny = self.grid.nx, self.grid.ny
        self.X, self.Y = self.grid.X, self.grid.Y

        def raster(vv, ff):
            return rasterize(self.grid, vv, ff, "top", empty=self.zb)
        H = raster(verts, faces)
        self._mesh = (verts, faces)
        self._fine = None
        V = np.asarray(verts, float)
        self.part_top = float(V[:, 2].max())
        self.H_raw = H.copy()  # before drilled holes are closed, so a later pass can pre-form them
        self.silhouette = H > self.zb + 1e-6
        # S = top of the material that is still there before this setup (primary: the stock box or the user's blank;
        # later: what is left)
        in_stock = (self.X >= x0 - 1e-6) & (self.X <= x1 + 1e-6) & (self.Y >= y0 - 1e-6) & (self.Y <= y1 + 1e-6)
        self.S = np.where(in_stock, zt, zb).astype(float)
        self.custom_stock = stock_mesh is not None
        if self.custom_stock:
            self.S = np.where(in_stock, np.maximum(raster(stock_mesh["vertices"], stock_mesh["faces"]), H), zb)
        if stock_surface:
            h = np.load(stock_surface["file"])
            jj = np.clip(((self.X - stock_surface["x0"]) / stock_surface["res"]).astype(int), 0, h.shape[1] - 1)
            ii = np.clip(((self.Y - stock_surface["y0"]) / stock_surface["res"]).astype(int), 0, h.shape[0] - 1)
            self.S = np.where(in_stock, np.minimum(h[ii, jj] + stock_surface["res"], self.S), zb)
        # a box stock above the part is the facing allowance (removed by facing); a blank can stand anywhere above
        # the part, and roughing has to start from its top
        self.rough_top = max(self.part_top, float(self.S.max())) if self.custom_stock else self.part_top
        self.keep_outside = outside == "keep"  # later setups: the outside of the part is already finished
        # drilled holes are made by drilling operations: close them so that milling ignores them.
        # the extra millimetre covers the rim cells a coarse grid would otherwise leave in the hole
        for h in holes or []:
            r = h["diameter"] / 2 + max(res, 1.0)
            m = (self.X - h["top"][0]) ** 2 + (self.Y - h["top"][1]) ** 2 <= r ** 2
            H[m] = np.maximum(H[m], h["top_z"])
        self.H = H

    def reserve_features(self, features: list, margin: float = 1.0) -> None:
        """Stop the cavity from cutting *below* a deeper feature, without skipping its footprint.

        Material over an inner pocket or slot, down to the surrounding floor (`protect_below`), is still
        removed. Only the volume beneath that floor stays for the feature's own operation. A feature at
        the same floor is not reserved: the outer pass includes its whole area.
        """
        if not features:
            return
        for item in features:
            if not isinstance(item, dict):
                continue
            below = item.get("protect_below")
            bb = item.get("bbox")
            if below is None or not bb or len(bb) < 4:
                continue
            ox0, oy0, ox1, oy1 = (float(v) for v in bb[:4])
            x0, y0, x1, y1 = ox0 - margin, oy0 - margin, ox1 + margin, oy1 + margin
            inside = (self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1) & self.silhouette
            self.H = np.where(inside, np.maximum(self.H, float(below)), self.H)

    def seal_chamfers(self, boxes: list) -> None:
        """Leave modelled chamfer faces for the chamfer operation. The pocket or cavity mill must not cut them."""
        for bb in boxes or []:
            if not bb or len(bb) < 4:
                continue
            x0, y0, x1, y1 = (float(v) for v in bb[:4])
            inside = (self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1) & self.silhouette
            if len(bb) >= 6:  # only the face's own height band: not the floor beside it (brep2cam@savrasov)
                inside &= (self.H >= float(bb[4]) - 0.5) & (self.H <= float(bb[5]) + 0.5)
            self.H = np.where(inside, np.maximum(self.H, self.part_top), self.H)

    def seal_features(self, features: list) -> None:
        """Hide pocket, slot and step footprints so a side pass cannot turn into them."""
        for item in features or []:
            if not isinstance(item, dict):
                continue
            bb = item.get("bbox")
            if not bb or len(bb) < 4:
                continue
            x0, y0, x1, y1 = (float(v) for v in bb[:4])
            inside = (self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1)
            self.H = np.where(inside, np.maximum(self.H, self.part_top), self.H)

    def prepare_holes(self, holes: list) -> np.ndarray:
        """Open only the holes that belong to a pocket or slot, so the interior pass can mill them to its floor.

        The side pass keeps the closed column. Here the cylinder is restored and stopped at `floor_z`:
        the endmill pre-forms the whole hole, including the part that sticks out of the pocket, and the
        drill cuts whatever is deeper. Returns the cells of those holes.
        """
        mask = np.zeros_like(self.silhouette)
        for h in holes or []:
            floor_z = h.get("floor_z")
            top = h.get("top") or []
            if floor_z is None or len(top) < 2 or not h.get("diameter"):
                continue
            r = float(h["diameter"]) / 2.0
            dist2 = (self.X - float(top[0])) ** 2 + (self.Y - float(top[1])) ** 2
            pad = r + max(self.res, 1.0)
            self.H = np.where(dist2 <= pad ** 2, self.H_raw, self.H)
            hole = dist2 <= r ** 2
            self.H = np.where(hole, np.maximum(float(floor_z), self.H_raw), self.H)  # never below the part itself
            mask |= hole
        return mask

    def in_stock(self) -> np.ndarray:
        x0, y0, _, x1, y1, _ = self.stock
        return (self.X >= x0 - 1e-6) & (self.X <= x1 + 1e-6) & (self.Y >= y0 - 1e-6) & (self.Y <= y1 + 1e-6)

    def work_mask(self, tool_d: float) -> np.ndarray:
        """Where the tool centre is allowed to work: everywhere (primary setup) or only over the part (later setups)."""
        if not self.keep_outside:
            return np.ones_like(self.silhouette)
        todo = self.silhouette | (self.S - np.maximum(self.H, self.zb) > 0.5)  # the part, plus wherever stock is still standing
        return _bdil(todo, disk((tool_d / 2 + 1.0) / self.res))

    def reachable_surface(self, tool_d: float) -> np.ndarray:
        """Surface a flat endmill of this diameter can actually cut (morphological closing of H by the tool disk)."""
        fp = disk(tool_d / 2 / self.res)
        return ndimage.grey_erosion(ndimage.grey_dilation(self.H, footprint=fp, mode="nearest"), footprint=fp, mode="nearest")

    def allowed(self, tool_d: float, leave: float = 0.0) -> np.ndarray:
        """Lowest gouge-free tool-centre height for a flat endmill (with radial + axial stock to leave)."""
        A = _gdil(self.H, disk((tool_d / 2 + leave) / self.res)) + leave
        if getattr(self, "depth_limit", None) is not None:
            A = np.maximum(A, self.depth_limit)  # the tool is not long enough to go deeper
        return np.where(self.work_mask(tool_d), A, self.zt + 1e3)


    def fine(self, res: float):
        """Target height-map on a finer grid (same origin), with the same closures (drilled holes, sealed chamfers /
        features) as the working field H. Used for the finishing contours, where the wall position matters."""
        if self._fine is not None and abs(self._fine[0].res - res) < 1e-9:
            return self._fine
        from .raster import Grid, rasterize
        g = Grid(self.x0, self.y0, self.x1, self.y1, res)
        Hf = rasterize(g, *self._mesh, "top", empty=self.zb)
        closed = self.H > self.H_raw + 1e-6
        if closed.any():
            i, j, _ = self.grid.index(g.X, g.Y)
            Hf = np.where(closed[i, j], np.maximum(Hf, self.H[i, j]), Hf)
        self._fine = (g, Hf)
        return self._fine


# --------------------------------------------------------------------------------------------- path primitives
class Emitter:
    def __init__(self, safe_z: float, feed: float, plunge: float) -> None:
        self.lines: list[str] = []
        self.safe, self.f, self.fp = safe_z, feed / 60.0, plunge / 60.0  # mm/min -> mm/s (FreeCAD internal unit)
        self.pos = None
        self.cut_len = 0.0

    def rapid_to(self, x, y):
        self.lines.append(f"G0 Z{self.safe:.3f}")
        self.lines.append(f"G0 X{x:.3f} Y{y:.3f}")
        self.pos = (x, y, self.safe)

    def plunge(self, z, from_z):
        x, y, _ = self.pos
        if from_z < self.safe:
            self.lines.append(f"G0 Z{from_z:.3f}")
        self.lines.append(f"G1 Z{z:.3f} F{self.fp:.3f}")
        self.pos = (x, y, z)

    def cut_to(self, x, y, z=None):
        px, py, pz = self.pos
        z = pz if z is None else z
        self.lines.append(f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{self.f:.3f}")
        self.cut_len += math.dist((px, py, pz), (x, y, z))
        self.pos = (x, y, z)

    def retract(self):
        if self.pos and self.pos[2] < self.safe:
            self.lines.append(f"G0 Z{self.safe:.3f}")
            self.pos = (self.pos[0], self.pos[1], self.safe)


def _runs(mask_row: np.ndarray):
    d = np.diff(np.r_[0, mask_row.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))


def _zigzag(em: Emitter, hf: HeightField, region: np.ndarray, z: float, above: float, stepover: float, ramp_len: float):
    """Clear a region at height z with raster lines along X; consecutive lines are linked when the link stays inside."""
    step = max(1, int(round(stepover / hf.res)))
    flip = False
    last = None  # (i, x_end) of the previous cut
    for i in range(0, hf.ny, step):
        runs = _runs(region[i])
        if not runs:
            continue
        if flip:
            runs = [(b, a) for a, b in reversed(runs)]
        for a, b in runs:
            xa, xb, y = hf.x0 + a * hf.res, hf.x0 + b * hf.res, hf.y0 + i * hf.res
            linked = False
            if last is not None and em.pos[2] <= z + 1e-6:
                li, lj = last
                if i - li == step and abs(lj - a) * hf.res <= 2.5 * stepover:
                    jlo, jhi = sorted((lj, a))
                    if region[li:i + 1, jlo:jhi + 1].all():
                        em.cut_to(xa, y, z)
                        linked = True
            if not linked:
                em.retract()
                em.rapid_to(xa, y)
                # ramp into the material along the run, then come back to clean the wedge
                L = abs(xb - xa)
                if L > 1.0:
                    rl = min(L, ramp_len)
                    xr = xa + math.copysign(rl, xb - xa)
                    em.plunge(above + 0.3, above + 2.0) if above + 0.3 < em.safe else None
                    em.lines.append(f"G1 X{xr:.3f} Y{y:.3f} Z{z:.3f} F{em.fp:.3f}")
                    em.pos = (xr, y, z)
                    em.cut_to(xa, y, z)
                else:
                    em.plunge(z, above + 2.0)
            em.cut_to(xb, y, z)
            last = (i, b)
        flip = not flip
    em.retract()


def _contours(em: Emitter, hf: HeightField, A: np.ndarray, z: float, above: float, min_len: float = 2.0):
    """Follow the boundary of the accessible region at height z (tool centre exactly where A == z)."""
    # on the allowed / not-allowed mask, not on A: interpolating A put the line up to a cell into the wall (brep2cam@savrasov)
    cg = contour_generator(x=hf.X[0], y=hf.Y[:, 0], z=(A > z + 1e-4).astype(float))
    for line in cg.lines(0.01):
        if len(line) < 3:
            continue
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1).sum()
        if seg < min_len:
            continue
        # stay inside the stock grid (contours on the outer border are the stock edge, not a wall)
        em.retract()
        em.rapid_to(*line[0])
        em.plunge(z, above + 2.0)
        for x, y in line[1::max(1, int(0.5 / hf.res))]:
            em.cut_to(float(x), float(y), z)
        em.cut_to(*map(float, line[-1]), z)
    em.retract()


def _offset_contours(em: Emitter, hf: HeightField, z: float, radius: float, above: float, res: float, min_len: float = 2.0,
                     allowed: np.ndarray | None = None):
    """Tool-centre paths at exactly `radius` from the material at height z: Euclidean distance transform of the
    material mask on a fine grid, contour at level `radius` with sub-cell interpolation (no grey-dilation staircase)."""
    g, Hf = hf.fine(res)
    # the material mask only changes where z crosses a height that occurs in the field: reuse the contours in between
    levels = getattr(hf, "_fine_levels", None)
    if levels is None:
        levels = hf._fine_levels = np.unique(np.round(Hf, 4))
        hf._offset_cache = {}
    band = (int(np.searchsorted(levels, z + 1e-6)), round(radius, 4))
    lines = hf._offset_cache.get(band)
    if lines is None:
        material = Hf > z + 1e-6
        if not material.any():
            hf._offset_cache[band] = []
            return
        dist = ndimage.distance_transform_edt(~material) * g.res
        lines = contour_generator(x=g.X[0], y=g.Y[:, 0], z=dist).lines(radius)
        hf._offset_cache[band] = lines
    # gouge guard on the working field, one coarse cell more forgiving than the exact offset
    if allowed is None:  # the same for every level of one operation: finishing passes it in under CAM_CPU_FAST=1
        allowed = hf.allowed(max(2 * radius - 2 * hf.res, 0.1), 0.0)
    for line in lines:
        if len(line) < 3 or np.linalg.norm(np.diff(line, axis=0), axis=1).sum() < min_len:
            continue
        i, j, _ = hf.grid.index(line[:, 0], line[:, 1])
        ok = allowed[i, j] <= z + 0.05  # the coarse field still guards against gouges below other parts of the cutter
        if ok.mean() < 0.5:
            continue
        em.retract()
        em.rapid_to(*map(float, line[0]))
        em.plunge(z, above + 2.0)
        for x, y in line[1::max(1, int(0.3 / g.res))]:
            em.cut_to(float(x), float(y), z)
        em.cut_to(*map(float, line[-1]), z)
    em.retract()


# --------------------------------------------------------------------------------------------- strategies
def _hole_xy(hole: dict):
    top = hole.get("top") or []
    if len(top) < 2 or not hole.get("diameter"):
        return None
    return float(top[0]), float(top[1]), float(hole["diameter"]) / 2.0


def _separate_hole(centers: np.ndarray, hf: HeightField, hole: dict, tool_d: float) -> np.ndarray:
    """Keep this cutter from straddling the hole wall.

    A tool thicker than the hole stays entirely outside it. A tool that fits may stand
    inside the finished cylinder or outside it, but not across the wall, so the opening
    it leaves is not wider than the hole.
    """
    xy = _hole_xy(hole)
    if xy is None:
        return centers
    cx, cy, radius = xy
    reach = tool_d / 2.0
    dist = np.sqrt((hf.X - cx) ** 2 + (hf.Y - cy) ** 2)
    if reach > radius + 1e-6:
        return centers & ~(dist < radius + reach)
    return centers & ~((dist > radius - reach) & (dist < radius + reach))


def _centers_inside_holes(hf: HeightField, holes: list, tool_d: float) -> np.ndarray:
    allowed = np.zeros(hf.H.shape, dtype=bool)
    reach = tool_d / 2.0
    for hole in holes or []:
        xy = _hole_xy(hole)
        if xy is None:
            continue
        cx, cy, radius = xy
        if reach > radius + 1e-6:
            continue
        dist = np.sqrt((hf.X - cx) ** 2 + (hf.Y - cy) ** 2)
        allowed |= dist <= max(radius - reach, 0.0)
    return allowed


def _open_feature_holes(hf: HeightField, holes: list | None, floor: float) -> np.ndarray:
    """Lower sealed hole columns to this pass's floor.

    The height map closes every hole up to the face it opens on, so a pocket or slot pass would
    otherwise walk around the column. The cutter clears that column, including the part of the
    circle that sticks out, and stops at this floor. Anything deeper stays for the drill.
    """
    extra = np.zeros(hf.H.shape, dtype=bool)
    for h in holes or []:
        top = h.get("top") or []
        if len(top) < 2 or not h.get("diameter"):
            continue
        top_z = h.get("top_z")
        if top_z is None:
            top_z = float(top[2]) if len(top) > 2 else hf.part_top
        top_z = float(top_z)
        if h.get("bottom_z") is not None:
            bottom = float(h["bottom_z"])
        elif h.get("depth") is not None:
            bottom = top_z - float(h["depth"])
        else:
            bottom = hf.zb
        stop = max(bottom, float(floor))
        radius = float(h["diameter"]) / 2.0
        dist2 = (hf.X - float(top[0])) ** 2 + (hf.Y - float(top[1])) ** 2
        hole = dist2 <= radius * radius
        hf.H = np.where(hole, np.maximum(stop, hf.H_raw), hf.H)  # never below the part itself
        extra |= hole
    return extra


def feature_clearing(hf: HeightField, bbox: list, open_sides: list, floor_z: float, tool_d: float, doc: float,
                     stepover: float, leave: float, feed: float, plunge: float, top_z: float | None = None,
                     holes: list | None = None) -> dict:
    """Clear one slot or pocket down to its floor, including a nested opening and the holes that meet it.

    Closed sides keep the tool centre one radius inside the wall. An open side is not a wall: the centre
    may cross that outline so the cutter covers every cell of the feature. A hole that meets this feature,
    or meets a pocket nested in it, is cleared with it and only down to this floor. The stepover is at
    most 40 % of the diameter, so the passes overlap. Nothing here uses the stock allowance.
    """
    ox0, oy0, ox1, oy1 = (float(v) for v in bbox[:4])
    r = tool_d / 2.0
    sides = set(open_sides or [])
    floor = float(floor_z) + leave
    saved_h = hf.H.copy()
    try:
        cover = (hf.X >= ox0) & (hf.X <= ox1) & (hf.Y >= oy0) & (hf.Y <= oy1)
        fitting = [h for h in (holes or []) if float(h.get("diameter") or 0) + 1e-6 >= tool_d]
        opened = _open_feature_holes(hf, fitting, floor)
        centers = _bdil(cover, disk(r / hf.res))
        if "-x" not in sides:
            centers &= hf.X >= ox0 + r
        if "+x" not in sides:
            centers &= hf.X <= ox1 - r
        if "-y" not in sides:
            centers &= hf.Y >= oy0 + r
        if "+y" not in sides:
            centers &= hf.Y <= oy1 - r
        # a cutter may enter a meeting hole only inside the finished cylinder, so the
        # upper section is not wider than the hole. A thicker cutter stays outside it.
        for h in holes or []:
            centers = _separate_hole(centers, hf, h, tool_d)
        centers |= opened & _centers_inside_holes(hf, fitting, tool_d)
        step = min(stepover, 0.4 * tool_d)
        em = Emitter(hf.zt + 5.0, feed, plunge)
        levels = []
        # a pocket inside a pocket or slot starts at that floor; the outer pass has already cleared above it
        origin = float(top_z) if top_z is not None else hf.part_top
        z = origin if top_z is not None else origin - doc
        while z > floor + 1e-6:
            levels.append(z)
            z -= doc
        levels.append(floor)
        prev = origin
        margin = 0.2
        if os.environ.get("CAM_FEATURE_SHALLOW", "0") == "1":
            # a step 0.1 mm deep never gets below part_top - 0.2 and its path comes out empty (evolution A16)
            margin = min(0.2, 0.5 * max(hf.part_top - float(floor_z), 0.0))
        material = cover & (hf.H < hf.part_top - margin)
        for z in levels:
            # include the floor itself: a cell whose target is exactly this level still has stock above it
            region = centers & _bdil(material & (hf.H <= z + 0.05), disk(r / hf.res))
            if region.any():
                _zigzag(em, hf, region, z, prev, step, ramp_len=max(tool_d, 1.0))
            prev = z
        return {"gcode": em.lines, "levels": levels, "cut_length_mm": round(em.cut_len, 1), "floor": floor}
    finally:
        hf.H = saved_h


def roughing(hf: HeightField, tool_d: float, doc: float, stepover: float, leave: float, feed: float, plunge: float, top: float | None = None, region_limit: np.ndarray | None = None) -> dict:
    A = hf.allowed(tool_d, leave)
    if region_limit is not None:
        A = np.where(region_limit, A, hf.zt + 1e3)
    top = hf.zt if top is None else top
    floor = max(float(A.min()), hf.zb)
    em = Emitter(hf.zt + 5.0, feed, plunge)
    fp = disk(tool_d / 2 / hf.res)
    levels = []
    z = top - doc
    while z > floor + 1e-6:
        levels.append(z)
        z -= doc
    levels.append(floor)
    # also stop at every plateau so that floors are roughed to their own height (+ leave), not only to the DOC grid
    for p in plateaus(A, hf.res):
        if floor < p < top and all(abs(p - l) > 0.05 for l in levels):
            levels.append(p)
    levels = sorted(set(levels), reverse=True)  # exact: a floor rounded down by 0.0004 mm leaves its level empty
    prev = top
    for z in levels:
        region = A <= z + 1e-6
        # only where material is left above this level (within the cutter footprint): no air cutting
        region &= _bdil(hf.S > z + 0.05, fp)
        if region_limit is not None:
            region &= region_limit
        if not region.any():
            prev = z
            continue
        _zigzag(em, hf, region, z, prev, stepover, ramp_len=3 * tool_d)
        _contours(em, hf, A, z, prev)
        prev = z
    return {"gcode": em.lines, "levels": levels, "cut_length_mm": round(em.cut_len, 1), "floor": floor}


def plateaus(A: np.ndarray, res: float, min_area_mm2: float = 12.0) -> list[float]:
    """Heights at which the tool-centre surface is genuinely flat (floors); chamfers and slopes are ignored."""
    ok = np.isfinite(A) & (A < 1e3)
    gy, gx = np.gradient(np.where(ok, A, 0.0))
    flat = ok & (np.abs(gx) < 1e-6) & (np.abs(gy) < 1e-6)
    vals = np.round(A[flat], 2)
    if not len(vals):
        return []
    if os.environ.get("CAM_PLATEAU_TOP", "1") != "0":
        # the highest exact height of each bin: a rounded-down level lies below the floor and finds no cell to cut
        # (roughing `A <= z`, finishing `cells & (A <= z)`); brep2cam@savrasov c9f1400. CAM_PLATEAU_TOP=0 - main's rounded value
        u, inv, c = np.unique(vals, return_inverse=True, return_counts=True)
        top = np.full(len(u), -np.inf)
        np.maximum.at(top, inv, A[flat])
        return [float(h) for h, n in zip(top, c) if n * res * res >= min_area_mm2]
    u, c = np.unique(vals, return_counts=True)
    return [float(h) for h, n in zip(u, c) if n * res * res >= min_area_mm2]


def finishing(hf: HeightField, tool_d: float, stepover: float, z_step: float, feed: float, plunge: float, top: float | None = None, region_limit: np.ndarray | None = None,
              wall_res: float | None = None, finish_top: bool = False) -> dict:
    """finish_top: the part's top plateau too (a setup without a facing step: roughing left its stock there)."""
    A = hf.allowed(tool_d, 0.0)
    if region_limit is not None:
        A = np.where(region_limit, A, hf.zt + 1e3)
    top = hf.part_top if top is None else top
    em = Emitter(hf.zt + 5.0, feed, plunge)
    # Keep a small tool's reachable wall orbit: the legacy dimensional cutoff
    # can exceed the entire path inside a narrow cavity. No floor/guard change.
    min_wall_len = min(2.0, 0.2 * tool_d) if os.environ.get("CAM_FINISH_TOOL_THRESHOLDS", "0") == "1" else 2.0
    levels = [p for p in plateaus(A, hf.res) if p < top - 1e-3 or (finish_top and p <= top + 1e-3)]
    floor = max(float(A.min()), hf.zb)
    # floors: clear exactly the plateau cells (the rest of that level was already cut by roughing, leave = wall stock)
    for z in sorted(levels, reverse=True):
        cells = np.abs(A - z) <= 0.011
        cells = ndimage.binary_dilation(cells, iterations=max(1, int(0.5 * stepover / hf.res)))  # overlap the walls' stock
        region = cells & (A <= z + 1e-6)
        if region.any():
            _zigzag(em, hf, region, z, z + 1.0, stepover, ramp_len=0.0)
    # walls: waterline contours between the top and the lowest floor
    zs = set(round(z, 3) for z in levels)
    z = top - z_step
    while z > floor:
        zs.add(round(z, 3))
        z -= z_step
    from .cpu_fast import enabled
    guard = hf.allowed(max(tool_d - 2 * hf.res, 0.1), 0.0) if enabled() and wall_res and region_limit is None and zs else None
    for z in sorted(zs, reverse=True):
        if wall_res and region_limit is None:
            _offset_contours(em, hf, z, tool_d / 2, z + 1.0, wall_res, min_len=min_wall_len, allowed=guard)
        else:
            _contours(em, hf, A, z, z + 1.0, min_len=min_wall_len)
    # freeform: drop-cutter raster where the surface is neither a floor nor a vertical wall
    flat = np.zeros_like(A, bool)
    for p in levels + [top]:
        flat |= np.abs(A - p) <= 0.011
    gy, gx = np.gradient(np.minimum(A, top + 1))
    slope = np.hypot(gx, gy) / hf.res
    sloped = (~flat) & (slope > 0.05) & (slope < 4.0) & (A < 1e3)
    share = float(sloped.mean())
    if share > 0.03:
        sloped = ndimage.binary_dilation(sloped, iterations=2)
        step = max(1, int(round(0.15 * tool_d / hf.res)))
        steep_up = os.environ.get("CAM_RASTER_STEEP", "0") == "1"
        for i in range(0, hf.ny, step):
            for a, b in _runs(sloped[i]):
                if (b - a) * hf.res < 1.0:
                    continue
                y = hf.y0 + i * hf.res
                em.retract()
                em.rapid_to(hf.x0 + a * hf.res, y)
                em.plunge(float(A[i, a]), float(A[i, a]) + 2.0)
                for j in range(a + 1, b + 1):
                    x, z, zp = hf.x0 + j * hf.res, float(A[i, j]), em.pos[2]
                    # CAM_RASTER_STEEP=1: a straight line between two samples cuts the corner of a step; over a rise
                    # steeper than 45 deg the tool stays at the higher sample between them (up first, or across first and
                    # then down); brep2cam@savrasov b0074f9. Off - main's straight line
                    if steep_up and z - zp > hf.res:
                        em.cut_to(em.pos[0], y, z)
                    elif steep_up and zp - z > hf.res:
                        em.cut_to(x, y, zp)
                    em.cut_to(x, y, z)
        em.retract()
    return {"gcode": em.lines, "levels": sorted(zs, reverse=True), "cut_length_mm": round(em.cut_len, 1), "freeform_share": round(share, 3)}


def chamfer_finishing(hf: HeightField, feat: dict, tool: dict, feed: float, plunge: float) -> dict:
    """Finish one modelled chamfer face, then stop.

    A 45 degree face is cut by a 90 degree chamfer mill. The flank lies on the face, so each strip is a
    plane, and a wide face is several overlapping strips. The tip stops at the bottom of the current strip.
    Any other angle is rastered with a ball endmill on that face only.
    """
    bb = feat.get("bbox") or []
    normal = feat.get("normal") or []
    if len(bb) < 4 or len(normal) < 3:
        return {"gcode": [], "passes": 0, "mode": "skipped"}
    angle = float(feat.get("angle_deg") or 45.0)
    z0 = float(feat.get("z_min") if feat.get("z_min") is not None else hf.zb)
    z1 = float(feat.get("z_max") if feat.get("z_max") is not None else hf.part_top)
    if z1 <= z0 + 0.2 or float(normal[2]) <= 0.2:
        return {"gcode": [], "passes": 0, "mode": "skipped"}
    nx, ny = float(normal[0]), float(normal[1])
    hlen = math.hypot(nx, ny)
    if hlen < 1e-6:
        return {"gcode": [], "passes": 0, "mode": "skipped"}
    ox, oy = nx / hlen, ny / hlen
    tx, ty = -oy, ox
    sp = tool.get("shape_params") or {}
    d = float(tool.get("diameter_mm") or sp.get("Diameter") or 6.0)
    edge = float(sp.get("CuttingEdgeAngle") or 90.0)
    if tool.get("category") == "chamfer_mill" and abs(angle - 45.0) <= 3.0 and abs(edge - 90.0) <= 1.0:
        return _chamfer_mill_face(hf, bb, z0, z1, ox, oy, tx, ty, d, sp, feed, plunge)
    return _ball_face(hf, bb, z0, z1, d, feed, plunge)


def _chamfer_mill_face(hf, bb, z0, z1, ox, oy, tx, ty, diameter, shape, feed, plunge) -> dict:
    tip_r = float(shape.get("TipDiameter") or 1.0) / 2.0
    edge_h = float(shape.get("CuttingEdgeHeight") or 5.0)
    cover = min(edge_h, (diameter - 2.0 * tip_r) / 2.0)
    cover = max(cover, 0.5)
    step = max(cover - 0.5, cover * 0.5)
    zs = []
    z = z0
    while z + cover < z1 - 0.05 and len(zs) < 40:
        zs.append(z)
        z += step
    last = max(z0, z1 - cover)
    if not zs or last > zs[-1] + 0.2:
        zs.append(last)
    em = Emitter(hf.zt + 5.0, feed, plunge)
    band = max(hf.res * 1.6, 0.75)
    x0, y0, x1, y1 = (float(v) for v in bb[:4])
    pad = max(hf.res, 0.6)
    inside = (hf.X >= x0 - pad) & (hf.X <= x1 + pad) & (hf.Y >= y0 - pad) & (hf.Y <= y1 + pad)
    passes = 0
    for z_pass in zs:
        sel = inside & (np.abs(hf.H - z_pass) <= band) & (hf.H >= z0 - pad) & (hf.H <= z1 + pad)
        if int(sel.sum()) < 4:
            continue
        xs, ys, hs = hf.X[sel], hf.Y[sel], hf.H[sel]
        vv = xs * tx + ys * ty
        order = np.argsort(vv)
        pts = []
        bin_w = max(hf.res, 0.5)
        v_cursor = None
        bucket = []
        for k in order:
            v = float(vv[k])
            if v_cursor is None:
                v_cursor = v
            if v > v_cursor + bin_w and bucket:
                pts.append(_chamfer_point(bucket, z_pass, ox, oy, tip_r))
                bucket = []
                v_cursor = v
            bucket.append((float(xs[k]), float(ys[k]), float(hs[k])))
        if bucket:
            pts.append(_chamfer_point(bucket, z_pass, ox, oy, tip_r))
        pts = [p for p in pts if p is not None]
        if pts and os.environ.get("CAM_CHAMFER_LINE", "0") == "1":
            # the face is a plane, so at one height its points lie on one line; a point off it is a cell of a
            # neighbouring face (the end of the chamfer) and would drag the flank into the chamfer; brep2cam@savrasov
            # dc126a7. Off - main's points as they are
            u_med = float(np.median([p[0] * ox + p[1] * oy for p in pts]))
            pts = [p for p in pts if abs(p[0] * ox + p[1] * oy - u_med) <= max(hf.res, 0.5)]
        if len(pts) < 2:
            continue
        # a short overrun past each end so the flank covers the corners of this face
        span = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1]) or 1.0
        ux, uy = (pts[-1][0] - pts[0][0]) / span, (pts[-1][1] - pts[0][1]) / span
        over = 0.6
        start = (pts[0][0] - ux * over, pts[0][1] - uy * over, z_pass)
        end = (pts[-1][0] + ux * over, pts[-1][1] + uy * over, z_pass)
        em.retract()
        em.rapid_to(start[0], start[1])
        em.plunge(z_pass, min(hf.part_top, z_pass + cover) + 1.0)
        for px, py, pz in pts:
            em.cut_to(px, py, pz)
        em.cut_to(*end)
        passes += 1
    em.retract()
    return {"gcode": em.lines, "passes": passes, "mode": "chamfer_mill", "cover_mm": round(cover, 2), "levels": [round(z, 3) for z in zs]}


def _chamfer_point(bucket, z_pass, ox, oy, tip_r):
    """Face point at this height, then the tip, half a tip-diameter outboard so the flank lies on the face."""
    x, y, h = min(bucket, key=lambda p: abs(p[2] - z_pass))
    # 45 deg: moving outward drops the face one millimetre per millimetre
    px = x + (h - z_pass) * ox
    py = y + (h - z_pass) * oy
    return (px + tip_r * ox, py + tip_r * oy, z_pass)


def _ball_face(hf, bb, z0, z1, diameter, feed, plunge) -> dict:
    """Raster one non-45-degree chamfer with a ball endmill. The tip follows the sphere that rests on the face."""
    rad = diameter / 2.0
    x0, y0, x1, y1 = (float(v) for v in bb[:4])
    pad = rad
    inside = (hf.X >= x0 - pad) & (hf.X <= x1 + pad) & (hf.Y >= y0 - pad) & (hf.Y <= y1 + pad)
    on = inside & (hf.H >= z0 - hf.res) & (hf.H <= z1 + hf.res)
    if int(on.sum()) < 4:
        return {"gcode": [], "passes": 0, "mode": "ball"}
    n = max(1, int(math.ceil(rad / hf.res)))
    ii, jj = np.nonzero(on)
    tip = np.full(hf.H.shape, -np.inf)
    for i, j in zip(ii, jj):
        i0, i1 = max(0, i - n), min(hf.ny, i + n + 1)
        j0, j1 = max(0, j - n), min(hf.nx, j + n + 1)
        yy = hf.Y[i0:i1, j0:j1] - hf.Y[i, j]
        xx = hf.X[i0:i1, j0:j1] - hf.X[i, j]
        dist = np.hypot(xx, yy)
        near = dist <= rad + 1e-9
        if not near.any():
            continue
        h = rad - np.sqrt(np.maximum(rad * rad - np.minimum(dist, rad) ** 2, 0.0))
        tip[i, j] = float(np.max(hf.H[i0:i1, j0:j1][near] - h[near]))
    step = max(1, int(round(0.45 / hf.res)))
    em = Emitter(hf.zt + 5.0, feed, plunge)
    passes = 0
    flip = False
    rows = list(range(0, hf.ny, step))
    for i in rows:
        runs = _runs(np.isfinite(tip[i]) & on[i])
        if not runs:
            continue
        if flip:
            runs = [(b, a) for a, b in reversed(runs)]
        for a, b in runs:
            cols = range(a, b + 1, step) if b >= a else range(a, b - 1, -step)
            cols = list(cols)
            if b not in cols and a != b:
                cols.append(b)
            if len(cols) < 2:
                continue
            y = hf.y0 + i * hf.res
            em.retract()
            em.rapid_to(hf.x0 + cols[0] * hf.res, y)
            em.plunge(float(tip[i, cols[0]]), float(tip[i, cols[0]]) + 2.0)
            for j in cols[1:]:
                em.cut_to(hf.x0 + j * hf.res, y, float(tip[i, j]))
            passes += 1
        flip = not flip
    em.retract()
    return {"gcode": em.lines, "passes": passes, "mode": "ball"}


# --------------------------------------------------------------------------------------------- analysis
def _cavity_cylinder(hole: dict) -> bool:
    """Evidence for a finite vertical cylinder, already in the detector setup frame."""
    try:
        if hole.get("type") != "hole" or not hole.get("vertical") or hole.get("subtype"):
            return False
        d, depth, z = float(hole["diameter"]), float(hole["depth"]), float(hole["top_z"])
        top = np.asarray(hole["top"], float)
        if top.shape != (3,) or not np.isfinite([d, depth, z, *top]).all() or min(d, depth) <= 0:
            return False
        if abs(top[2] - z) > 0.01:
            return False
        if hole.get("axis") is not None:
            axis = np.asarray(hole["axis"], float)
            if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis[:2]) > 1e-3 or abs(abs(axis[2])-1) > 1e-3:
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def _drillable_hole(hole: dict, tools: list[dict]) -> bool:
    """Only a catalog-supported vertical cylinder may explain removal (A28)."""
    from ..agents.tool_selection import _reach
    if not _cavity_cylinder(hole):
        return False
    try:
        d, depth, top = float(hole["diameter"]), float(hole["depth"]), float(hole["top_z"])
        if not np.isfinite([d, depth, top]).all() or min(d, depth) <= 0:
            return False
        if len(hole.get("top", [])) < 3 or abs(float(hole["top"][2]) - top) > 0.01:
            return False
        for t in tools:
            if t.get("category") != "twist_drill":
                continue
            rng = t.get("diameter_range_mm")
            fixed = t.get("diameter_mm")
            if fixed is not None:
                fits = np.isfinite(float(fixed)) and abs(float(fixed) - d) <= 1e-6
            elif rng and len(rng) == 2:
                if not np.isfinite(np.asarray(rng, float)).all():
                    continue
                fits = float(rng[0]) <= d <= float(rng[1])
                step = float(t.get("diameter_step_mm") or 0)
                if not np.isfinite(step) or step < 0:
                    continue
                if step > 0:
                    fits &= abs((d - float(rng[0])) / step - round((d - float(rng[0])) / step)) <= 1e-6
            else:
                continue
            # No invented length/depth default: uncertain manufacture remains a cavity.
            length = t.get("overall_length_mm") or (t.get("shape_params") or {}).get("Length")
            ratio = t.get("max_depth_x_d")
            if not fits or not length or not ratio or not np.isfinite([float(length), float(ratio), _reach(t)]).all():
                continue
            limit = min(float(ratio) * d, _reach(t))
            if t.get("flute_length_mm") is not None:
                flute = float(t["flute_length_mm"])
                if not np.isfinite(flute):
                    continue
                limit = min(limit, flute)
            if depth <= limit + 1e-6:
                return True
    except (KeyError, TypeError, ValueError):
        return False
    return False


def cavity_tool_reaches(tools: list[dict], diameters: list[float], iso: str | None = None) -> dict:
    """Fixed endmills only, using the TS reach contract; pending material must be ISO-invariant."""
    from ..agents.tool_selection import _reach
    groups = [iso] if iso else sorted({g for t in tools for g in t.get("iso", [])})
    result = {}
    for d in diameters:
        limits = []
        for group in groups:
            valid = []
            for t in tools:
                if t.get("category") != "endmill" or group not in t.get("iso", []) or t.get("diameter_mm") != d:
                    continue
                # Do not inherit TS's legacy invented 60 mm length for unknown catalog data.
                length = t.get("overall_length_mm") or (t.get("shape_params") or {}).get("Length")
                try:
                    if length is None or not np.isfinite(float(length)) or float(length) <= 0:
                        continue
                    reach = _reach(t)
                    if np.isfinite(reach) and reach > 0:
                        valid.append(reach)
                except (TypeError, ValueError):
                    continue
            limits.append(max(valid, default=0.0))
        result[d] = limits[0] if limits and all(x == limits[0] for x in limits) else 0.0
    return result


def residual_analysis(hf: HeightField, features: list[dict], tool_diameters: list[float],
                      drill_tools: list[dict] | None = None, tool_reaches: dict | None = None) -> dict:
    """How much of what must be removed from this side is NOT covered by recognised features, and which tool fits."""
    cell = hf.res ** 2
    bounded_holes = os.environ.get("CAM_CAVITY_HOLE_DEPTH", "0") == "1"
    part_cols = hf.silhouette.copy() if bounded_holes else hf.silhouette
    if bounded_holes:
        # A through cylinder rasterises to zb, outside the top-surface silhouette.
        # Its recognised footprint still contains stock to remove; do not seal H.
        for f in features:
            if _cavity_cylinder(f):
                part_cols |= (hf.X - f["top"][0]) ** 2 + (hf.Y - f["top"][1]) ** 2 <= (f["diameter"] / 2) ** 2
    custom = getattr(hf, "custom_stock", False)
    # the facing allowance of a box stock is not part of the cavity; a user blank above the part is
    s_eff = hf.S if custom else np.minimum(hf.S, hf.part_top)
    removal = np.where(part_cols, np.clip(s_eff - np.minimum(hf.H, hf.part_top), 0, None), 0.0)  # inside the silhouette
    # later setups: stock that an earlier setup could not reach (e.g. below its tool reach) counts too; a user blank
    # can stand well outside the silhouette, further than one outer-profile pass removes
    if hf.keep_outside or custom:
        removal = np.where(~part_cols, np.clip(s_eff - hf.zb, 0, None), removal)
    covered = np.zeros_like(part_cols)
    hole_intervals = []
    for f in features:
        if f.get("type") in ("pocket", "slot", "step") and f.get("bbox"):
            x0, y0, x1, y1 = f["bbox"]
            fill = float(f.get("area") or 0.0) / max((x1 - x0) * (y1 - y0), 1e-9)
            if f.get("inner_loops") or fill < 0.8:
                continue  # floor with islands / irregular outline: the pocket operation will not clear its bounding box
            covered |= (hf.X >= x0 - hf.res) & (hf.X <= x1 + hf.res) & (hf.Y >= y0 - hf.res) & (hf.Y <= y1 + hf.res)
        elif f.get("type") == "hole":
            if bounded_holes:
                if _drillable_hole(f, drill_tools or []):
                    circle = (hf.X - f["top"][0]) ** 2 + (hf.Y - f["top"][1]) ** 2 <= (f["diameter"] / 2) ** 2
                    hole_intervals.append((float(f["top_z"]) - float(f["depth"]), float(f["top_z"]), circle))
                continue
            r = f["diameter"] / 2 + hf.res
            covered |= (hf.X - f["top"][0]) ** 2 + (hf.Y - f["top"][1]) ** 2 <= r ** 2
        elif f.get("type") == "chamfer":
            pass
    total = float(removal.sum() * cell)
    resid_map = np.where(covered, 0.0, removal)
    if bounded_holes:
        # Union the vertical intervals; overlapping/coaxial features cannot pay twice.
        paid = np.zeros_like(removal)
        endpoints = sorted({z for bottom, top, _ in hole_intervals for z in (bottom, top)})
        for bottom, top in zip(endpoints, endpoints[1:]):
            circle = np.zeros_like(part_cols)
            for lo, hi, mask in hole_intervals:
                if lo <= bottom and hi >= top:
                    circle |= mask
            amount = np.clip(np.minimum(s_eff, top) - np.maximum(np.minimum(hf.H, hf.part_top), bottom), 0, None)
            paid += np.where(circle & part_cols, amount, 0.0)
        resid_map = np.where(covered, 0.0, np.maximum(0.0, removal - paid))
    resid_map[resid_map < 1.2] = 0.0  # chamfers / fillets / edge breaks are not a cavity
    resid = float(resid_map.sum() * cell)
    reach = {}
    for d in tool_diameters:
        C = hf.reachable_surface(d)
        got = np.where(part_cols | custom, np.clip(s_eff - np.minimum(C, hf.part_top), 0, None), 0.0)
        if os.environ.get("CAM_CAVITY_TOOL_REACH", "0") == "1":
            # Whole-column credit only; the datum is the same global part top used by CAM's scalar limit.
            length = (tool_reaches or {}).get(d, 0.0)
            got = np.where((length > 0) & (hf.part_top - C <= length + 1e-6), got, 0.0)
        reach[d] = round(float(got.sum() * cell) / max(total, 1e-9), 3)
    ii, jj = np.nonzero(resid_map > 0)
    bbox = [float(hf.x0 + jj.min() * hf.res), float(hf.y0 + ii.min() * hf.res), float(hf.x0 + jj.max() * hf.res), float(hf.y0 + ii.max() * hf.res)] if len(ii) else None
    inner = (resid_map > 0) & part_cols
    floor = float(np.maximum(hf.H, hf.zb)[inner if inner.any() else resid_map > 0].min()) if len(ii) else None
    return {"removal_mm3": round(total, 1), "residual_mm3": round(resid, 1), "residual_share": round(resid / max(total, 1e-9), 3),
            "max_depth": round(float(removal.max()), 3), "floor_z": floor, "bbox": bbox, "reach_by_tool_d": reach}


def _largest_tool_reaching(ds: list[float], reach: dict, target: float) -> float:
    """Largest tool reaching `target` of the removal; if none does, the largest one within 2 % of the best reach.

    Falling back to the smallest tool is wrong when the part is simply not reachable from this side (a topology
    optimised bracket): a 2 mm cutter then buys ~1 % more reach for 36x the time and cannot reach the depth at all.
    """
    best = max(reach[d] for d in ds)
    thr = target if best >= target else best - 0.02
    return next((d for d in ds if reach[d] >= thr), ds[-1])


def detect_cavity(verts, faces, size, features: list[dict], tool_diameters: list[float], primary: bool = True,
                  min_share: float = 0.05, min_mm3: float = 1000.0, stock_surface: dict | None = None,
                  stock_mesh: dict | None = None, margins: dict | None = None,
                  drill_tools: list[dict] | None = None, tool_reaches: dict | None = None) -> dict | None:
    """A 'cavity' feature when recognised features do not explain what must be removed from this side."""
    # the stock box must sit on the part as it is really modelled: a part that is not built around the origin (a STEP
    # exported from an assembly) would otherwise get a height-map shifted away from its own geometry, and the
    # toolpaths would cut where the part is not - for a connecting rod that means milling both bosses away
    V = np.asarray(verts, float)
    lo, hi = V.min(0), V.max(0)
    mg = {"xy_margin": 2.0, "z_top": 1.0, "z_bottom": 0.0, **(margins or {})}
    stock = [lo[0] - mg["xy_margin"], lo[1] - mg["xy_margin"], lo[2] - mg["z_bottom"], hi[0] + mg["xy_margin"], hi[1] + mg["xy_margin"], hi[2] + mg["z_top"]]
    res = max(0.5, max(size[:2]) / 400)
    holes = [f for f in features if f.get("type") == "hole" and f.get("vertical") and f["diameter"] <= 13]
    if os.environ.get("CAM_CAVITY_HOLE_DEPTH", "0") == "1":
        holes = []  # analysis must see raw depths, not the CAM-only drilling closure
    if not primary:
        stock = [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]]
    if stock_mesh is not None:
        from .stock import union_bbox
        stock = union_bbox([*lo, *hi], stock_mesh["bbox"])
    hf = HeightField(verts, faces, stock, res, holes=holes, outside="cut" if primary else "keep", stock_surface=stock_surface, stock_mesh=stock_mesh)
    a = residual_analysis(hf, features, tool_diameters, drill_tools=drill_tools, tool_reaches=tool_reaches)
    if os.environ.get("CAM_REL_THRESHOLDS", "0") == "1":
        min_mm3 = min(min_mm3, 0.2 * a["removal_mm3"])  # a small part removes less than the absolute floor in all (evolution A04)
    if a["residual_mm3"] < min_mm3 or a["residual_share"] < min_share:
        return None
    ds = sorted(tool_diameters, reverse=True)
    rough = _largest_tool_reaching(ds, a["reach_by_tool_d"], 0.85)
    finish = _largest_tool_reaching(ds, a["reach_by_tool_d"], 0.985)
    finish = min(finish, rough)
    x0, y0, x1, y1 = a["bbox"]
    # under a blank the deepest removal includes the allowance above the part, so take the floor from the target
    z = a["floor_z"] if stock_mesh is not None and a["floor_z"] is not None else float(hi[2]) - a["max_depth"]
    cavity = {"type": "cavity", "bbox": a["bbox"], "size": [round(x1 - x0, 2), round(y1 - y0, 2)], "depth": a["max_depth"], "z": round(z, 3),
            "residual_mm3": a["residual_mm3"], "residual_share": a["residual_share"], "removal_mm3": a["removal_mm3"],
            "rough_tool_d": rough, "finish_tool_d": finish, "reach_by_tool_d": {str(k): v for k, v in a["reach_by_tool_d"].items()}, "faces": []}
    if os.environ.get("CAM_CAVITY_TOOL_REACH", "0") == "1":
        cavity["tool_reach_mm"] = {str(d): (tool_reaches or {}).get(d, 0.0) for d in ds}
        cavity["tool_reach_all_zero"] = not any(a["reach_by_tool_d"].values())
    return cavity
