"""Simulation-driven rest machining: finish what the simulated program left, with a tool that can, gouge-free.

1. residual   the cells where the simulated workpiece stands more than the tolerance above the target (in a
              multi-setup job only where material is really there: the other setups may have cut it from their side);
2. tool       for every candidate (ball and flat endmills) the morphological closing C of the target by the tool
              (shield.py) tells, before any path exists, which residual cells that tool can bring within tolerance
              (C - T; a flat endmill only where the slope keeps its staircase s tan(a) under tol / 4: floors and the
              sharp corners at their walls, which a ball cannot reach); the largest tool that does (nearly) as well
              as any is taken, and a smaller one for what it cannot reach;
3. path       a cross-hatched raster (along X, then along Y) of drop-cutter positions: the tool tip at S (the lowest
              gouge-free height, shield.Shield.drop) at every cell of the lines, stepover from the scallop height
              h = tol / 4 (s = sqrt(8 R h)); points are dropped only where the straight move between the kept ones stays
              between S and S + tol / 10 (never below S); links between runs go over the highest S on the way.
Every path is gouge-free in the verifier's column model by construction; the shield still certifies it.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from .dexel import ToolShape
from .shield import _NB, EMPTY, Shield


def residual_mask(sim: dict, tol: float, inside=None, frame=None, min_area_mm2: float = 1.0) -> np.ndarray:
    """Cells (sim grid) with material more than tol above the target. inside / frame: the multi-setup workpiece test."""
    dm, th, g = sim["dexel"], sim["target_h"], sim["grid"]
    m = (dm.h - th > tol) & dm.material()
    if inside is not None and m.any():
        ii, jj = np.nonzero(m)
        z0, z1 = th[ii, jj] + tol, dm.h[ii, jj]
        X, Y = g.x0 + jj * g.res, g.y0 + ii * g.res
        there = np.zeros(len(ii), bool)
        for t in (0.15, 0.5, 0.85):
            P = np.c_[X, Y, z0 + t * (z1 - z0)]
            there |= inside(frame.to_original(P) if frame is not None else P)
        m[ii[~there], jj[~there]] = False
    if m.any():  # specks of a cell or two along walls are column artefacts, not material
        lab, n = ndimage.label(m)
        area = ndimage.sum(m, lab, np.arange(1, n + 1)) * g.res ** 2
        m &= np.isin(lab, np.flatnonzero(area >= min_area_mm2) + 1)
    return m


def _window(mask: np.ndarray, pad: int):
    ii, jj = np.nonzero(mask)
    return (max(int(ii.min()) - pad, 0), min(int(ii.max()) + pad + 1, mask.shape[0]),
            max(int(jj.min()) - pad, 0), min(int(jj.max()) + pad + 1, mask.shape[1]))


def plan_tools(sh: Shield, M: np.ndarray, balls: list[tuple[dict, ToolShape]], tol: float, z_floor: float,
               max_tools: int = 2, share: float = 0.85, min_area_mm2: float = 4.0, z_limit: dict | None = None) -> list[dict]:
    """balls: [(library tool, shape)] (ball, bull and flat endmills); z_limit: {tool id: lowest tip height its reach allows}.
    Candidates are tried from the widest stepover (shortest path) down; a tool takes the cells it brings within
    tolerance and can reach. Returns [{"tool", "shape", "fix" (cells it finishes), "S" (drop heights), "window"}]."""
    cell = sh.res ** 2
    if M.sum() * cell < min_area_mm2 or not balls:
        return []
    balls = sorted(balls, key=lambda b: -stepover(b[1], tol))
    pad = int(math.ceil(2 * max(b[1].R for b in balls) / sh.res)) + 2
    w = _window(M, pad)
    T = sh.T[w[0]:w[1], w[2]:w[3]]
    Tf = np.where(T > EMPTY / 2, T, z_floor)  # outside the part the tool may go down to the part's lowest level
    left = M[w[0]:w[1], w[2]:w[3]].copy()
    cache = {}

    gy, gx = np.gradient(Tf, sh.res)
    slope = np.hypot(gx, gy)
    sub = Shield(Tf, 0.0, 0.0, sh.res)
    from .cpu_fast import dilate, enabled
    part = enabled() and _NB is not None  # CAM_CPU_FAST: ok only on M, S only within a disk of M (same values there)
    Mw = left.copy()
    near = {}

    def evaluate(tool, shape):
        if id(shape) not in cache:
            D = None
            if part:
                D = near[id(shape)] = dilate(Mw, sub.disk(shape), binary=True)
            S = np.maximum(sub.drop(shape, need=D), z_floor)
            C = sub.closing(S, shape, need=Mw if part else None)
            ok = C - Tf <= tol / 2
            if z_limit and tool.get("id") in z_limit:  # not deeper than this tool reaches
                ok &= S >= z_limit[tool["id"]] - 1e-6
            if shape.kind == "flat":  # staircase of a flat bottom on a slope: s tan(a) <= tol / 4
                ok &= slope * stepover(shape, tol) <= tol / 4
            cache[id(shape)] = (S, ok)
        return cache[id(shape)]

    out = []
    pool = list(balls)
    while pool and len(out) < max_tools and left.sum() * cell >= min_area_mm2:
        gains = []
        for tool, shape in pool:
            S, ok = evaluate(tool, shape)
            gains.append(int((left & ok).sum()))
        best = max(gains)
        if best * cell < max(min_area_mm2, 0.1 * M.sum() * cell if out else 0.0):
            break
        k = next(i for i, g in enumerate(gains) if g >= share * best)  # the widest stepover that does nearly as well
        tool, shape = pool[k]
        S, ok = evaluate(tool, shape)
        if np.isnan(S).any():  # the raster reads S anywhere in the window: the cells outside the disk of M
            D = near[id(shape)]
            S = np.where(D, S, np.maximum(sub.drop(shape, need=~D), z_floor))
        fix = left & ok
        out.append({"tool": tool, "shape": shape, "fix": fix, "S": S, "window": w, "area_mm2": round(float(fix.sum() * cell), 1)})
        left &= ~fix
        pool = pool[k + 1:]  # a second tool only with a finer stepover (a smaller tool)
    return out


def stepover(shape: ToolShape, tol: float) -> float:
    """Raster stepover for a scallop of tol / 4: ball radius on floors and slopes, the corner radius of a bull nose (it
    cuts slopes with its corner), the cutter radius of a flat endmill (cusps of its flank on walls across the lines)."""
    r = shape.cr if shape.kind == "bull" and shape.cr > 0 else shape.R
    return math.sqrt(8.0 * r * tol / 4)


def _make_simplify():
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=True)
    def simplify(z, eps, maxw):
        n = len(z)
        keep = np.empty(n, np.int64)
        keep[0] = 0
        m = 1
        a = 0
        while a < n - 1:
            best = a + 1
            b = a + 2
            while b < n and b - a <= maxw:
                ok = True
                for j in range(a + 1, b):
                    zi = z[a] + (z[b] - z[a]) * (j - a) / (b - a)
                    if zi < z[j] - 1e-9 or zi > z[j] + eps:
                        ok = False
                        break
                if not ok:
                    break
                best = b
                b += 1
            keep[m] = best
            m += 1
            a = best
        return keep[:m]

    return simplify


_SIMPLIFY = _make_simplify()


def _simplify(z: np.ndarray, eps: float, maxw: int = 400) -> np.ndarray:
    if len(z) <= 2:
        return np.arange(len(z))
    if _SIMPLIFY is not None:
        return _SIMPLIFY(np.ascontiguousarray(z, np.float64), float(eps), int(maxw))
    return np.arange(len(z))


def raster(plan: dict, sh: Shield, tol: float, z_limit: float, safe_z: float, feed: float, plunge: float) -> tuple[list[str], dict]:
    """Cross-hatched drop-cutter raster over the cells this tool finishes. feed / plunge in mm/min (G-code F in mm/s,
    FreeCAD's internal unit, as zlevel.Emitter writes it)."""
    shape, S, fix, (i0, i1, j0, j1) = plan["shape"], plan["S"], plan["fix"], plan["window"]
    res = sh.res
    R = shape.R
    from .cpu_fast import dilate, enabled
    fp = _disk(R / res)
    visit = dilate(fix, fp, binary=True) if enabled() else ndimage.binary_dilation(fix, structure=fp)
    Z = np.maximum(S, z_limit)  # not deeper than the tool reaches
    s = stepover(shape, tol)  # scallop height tol / 4
    k = max(1, int(s / res))
    f, fp = feed / 60.0, plunge / 60.0
    lines: list[str] = [f"G0 Z{safe_z:.3f}"]
    pos = None
    n_pts = 0
    cut_len = 0.0
    x0 = sh.x0 + j0 * res
    y0 = sh.y0 + i0 * res

    def xy(i, j):
        return x0 + j * res, y0 + i * res

    def link_height(ia, ja, ib, jb):
        n = int(max(abs(ib - ia), abs(jb - ja))) + 1
        ii = np.rint(np.linspace(ia, ib, n)).astype(int)
        jj = np.rint(np.linspace(ja, jb, n)).astype(int)
        r = int(math.ceil(R / res))
        lo_i, hi_i = max(min(ii.min(), ib) - r, 0), min(max(ii.max(), ib) + r + 1, Z.shape[0])
        lo_j, hi_j = max(min(jj.min(), jb) - r, 0), min(max(jj.max(), jb) + r + 1, Z.shape[1])
        return float(Z[lo_i:hi_i, lo_j:hi_j].max()) + 0.5  # over everything within a tool radius of the link

    for axis in (1, 0):  # lines along X (rows), then along Y (columns)
        A = visit if axis == 1 else visit.T
        ZZ = Z if axis == 1 else Z.T
        flip = False
        for a in range(0, A.shape[0], k):
            row = A[a]
            if not row.any():
                continue
            d = np.diff(np.r_[0, row.astype(np.int8), 0])
            runs = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1))
            if flip:
                runs = [(e, b) for b, e in reversed(runs)]
            flip = not flip
            for b, e in runs:
                idx = np.arange(b, e + 1) if e >= b else np.arange(b, e - 1, -1)
                z = ZZ[a, idx]
                keep = idx[_simplify(z, tol / 10)]
                pts = [((a, j) if axis == 1 else (j, a)) for j in keep]
                zs = [float(ZZ[a, j]) for j in keep]
                i_s, j_s = pts[0]
                if pos is None or math.hypot(pts[0][0] - pos[0], pts[0][1] - pos[1]) * res > 10.0:
                    lines.append(f"G0 Z{safe_z:.3f}")
                    xs, ys = xy(i_s, j_s)
                    lines.append(f"G0 X{xs:.4f} Y{ys:.4f}")
                    lines.append(f"G0 Z{min(safe_z, zs[0] + 2.0):.4f}")
                else:
                    zl = link_height(pos[0], pos[1], i_s, j_s)
                    lines.append(f"G1 Z{max(zl, pos[2]):.4f} F{f:.3f}")
                    xs, ys = xy(i_s, j_s)
                    lines.append(f"G1 X{xs:.4f} Y{ys:.4f} F{f:.3f}")
                lines.append(f"G1 Z{zs[0]:.4f} F{fp:.3f}")
                prev = (*xy(i_s, j_s), zs[0])
                for (i, j), zz in zip(pts[1:], zs[1:]):
                    x, y = xy(i, j)
                    if abs(zz - prev[2]) > max(3.0 * math.hypot(x - prev[0], y - prev[1]), 0.5):
                        # a step of S (a wall): never diagonally through it - up first, or across first and then down
                        if zz > prev[2]:
                            lines.append(f"G1 Z{zz:.4f} F{f:.3f}")
                        else:
                            lines.append(f"G1 X{x:.4f} Y{y:.4f} Z{prev[2]:.4f} F{f:.3f}")
                            lines.append(f"G1 Z{zz:.4f} F{fp:.3f}")
                            cut_len += math.dist(prev, (x, y, zz))
                            prev = (x, y, zz)
                            continue
                    lines.append(f"G1 X{x:.4f} Y{y:.4f} Z{zz:.4f} F{f:.3f}")
                    cut_len += math.dist(prev, (x, y, zz))
                    prev = (x, y, zz)
                n_pts += len(pts)
                pos = (pts[-1][0], pts[-1][1], zs[-1])
    lines.append(f"G0 Z{safe_z:.3f}")
    info = {"tool": plan["tool"].get("id"), "diameter_mm": 2 * R, "stepover_mm": round(k * res, 3), "points": n_pts,
            "cut_length_mm": round(cut_len, 1), "area_mm2": plan.get("area_mm2")}
    return lines, info


def _disk(r: float) -> np.ndarray:
    n = int(math.ceil(r))
    jj, ii = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    return ii ** 2 + jj ** 2 <= r * r + 1e-9
