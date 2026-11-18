"""Gouge shield and drop-cutter primitives on the simulation grid.

The material-removal simulation (dexel.py) lowers a grid column c under a move m to  z(m) + h(d(c, m)),  h = profile of
the cutting surface above the tool tip, d = XY distance of the column centre to the move. A path is gouge-free in that
model iff for every move m and every column c within the tool radius

        z_low(m) + h(d(c, m)) >= T(c) + delta            (T = top view of the target from this setup)

i.e. iff the lift  lambda(m) = max(0, max_c [T(c) + delta - h(d(c, m))] - z_low(m))  is zero (z_low = lower end of the
move: conservative for ramps). Columns are points: a flank that touches a wall column centre by a micrometre counts as a
cut through the whole column, so only columns closer than R - eps_xy are tested (eps_xy: the lateral depth still
accepted, like delta below the target; certified = no gouge deeper than -delta or wider than eps_xy). The shield evaluates lambda for every piece of every move on the very grid the verifier
uses (CUDA kernel, numba fallback) and raises exactly the violating pieces by their lambda, linked by vertical moves (a
higher tool never cuts deeper). A path that comes back unchanged is certified gouge-free; a repaired one is
gouge-free by construction (runtime assurance: the planner proposes, the shield applies the minimal correction that
restores the invariant). delta < 0 is the depth below the target that is still accepted (a fraction of the tolerance).

Drop-cutter primitives for rest machining on the same grid:
    drop(shape)     S(p) = max_{|q-p|<=R} T(q) - h(|q-p|)      lowest gouge-free tip height at every cell
    closing(S)      C(c) = min_{|p-c|<=R} S(p) + h(|c-p|)      lowest surface a tool of that shape can leave at c
C - T is the material that this tool cannot remove however dense its path (corners sharper than the tool, slots
narrower than it): it decides which tool a rest pass needs before any path is generated.
"""
from __future__ import annotations

import math
import re

import numpy as np

from .dexel import ToolShape, _gpu_ready

KIND = {"flat": 0, "ball": 1, "bull": 2}
EMPTY = -1e9  # target height of a column without target material
LIFT_BLOCK = 16  # cells per side of the target block maxima for the lift bound (CAM_CPU_FAST)

_SRC = r"""
__device__ __forceinline__ double prof(int kind, double R, double cr, double tip_r, double cot, double d) {
    if (kind == 0) return 0.0;
    if (kind == 1) return R - sqrt(fmax(R * R - d * d, 0.0));
    if (kind == 2) { double e = fmax(d - (R - cr), 0.0); return cr - sqrt(fmax(cr * cr - e * e, 0.0)); }
    return fmax(d - tip_r, 0.0) * cot;
}

extern "C" __global__ void lift(const double* T, const double* S, double* out, const long long nseg, const int ny, const int nx,
                                const double gx0, const double gy0, const double res, const int kind,
                                const double R, const double cr, const double tip_r, const double cot, const double delta,
                                const double rcut) {
    __shared__ double red[256];
    const long long q = blockIdx.x;
    const double x0 = S[6 * q], y0 = S[6 * q + 1], z0 = S[6 * q + 2], x1 = S[6 * q + 3], y1 = S[6 * q + 4], z1 = S[6 * q + 5];
    const int j0 = max((int)floor((fmin(x0, x1) - R - gx0) / res), 0);
    const int j1 = min((int)ceil((fmax(x0, x1) + R - gx0) / res), nx - 1);
    const int i0 = max((int)floor((fmin(y0, y1) - R - gy0) / res), 0);
    const int i1 = min((int)ceil((fmax(y0, y1) + R - gy0) / res), ny - 1);
    const double dx = x1 - x0, dy = y1 - y0, L2 = dx * dx + dy * dy;
    double need = -1e300;
    if (j1 >= j0 && i1 >= i0) {
        const long long w = j1 - j0 + 1, n = w * (long long)(i1 - i0 + 1);
        for (long long c = threadIdx.x; c < n; c += blockDim.x) {
            const int i = i0 + (int)(c / w), j = j0 + (int)(c % w);
            const double X = gx0 + j * res, Y = gy0 + i * res;
            const double dz = z1 - z0;
            if (kind == 0 && L2 >= 1e-12 && fabs(dz) >= 1e-9) {  // inclined flat move: lowest point over the disk (as the sweep)
                const double L = sqrt(L2);
                const double t = ((X - x0) * dx + (Y - y0) * dy) / L2;
                const double dp = fabs((X - x0) * dy - (Y - y0) * dx) / L;
                if (dp > rcut) continue;
                const double ww = sqrt(fmax(rcut * rcut - dp * dp, 0.0)) / L;
                if (t + ww < 0.0 || t - ww > 1.0) continue;
                const double ta = fmin(fmax(t - ww, 0.0), 1.0), tb = fmin(fmax(t + ww, 0.0), 1.0);
                need = fmax(need, T[(long long)i * nx + j] + delta - fmin(z0 + ta * dz, z0 + tb * dz));
                continue;
            }
            double d;
            if (L2 < 1e-12) d = hypot(X - x0, Y - y0);
            else {
                const double t = fmin(fmax(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0), 1.0);
                d = hypot(X - (x0 + t * dx), Y - (y0 + t * dy));
            }
            if (d > rcut) continue;
            need = fmax(need, T[(long long)i * nx + j] + delta - prof(kind, R, cr, tip_r, cot, d) - fmin(z0, z1));
        }
    }
    red[threadIdx.x] = need;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) out[q] = fmax(0.0, red[0]);
}

extern "C" __global__ void dilate(const double* T, double* S, const int ny, const int nx, const double res, const int r,
                                  const int kind, const double R, const double cr, const double tip_r, const double cot) {
    const long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= (long long)ny * nx) return;
    const int i = (int)(p / nx), j = (int)(p % nx);
    double best = -1e300;
    for (int a = max(i - r, 0); a <= min(i + r, ny - 1); a++) {
        const double dy = (a - i) * res;
        for (int b = max(j - r, 0); b <= min(j + r, nx - 1); b++) {
            const double dx = (b - j) * res;
            const double d = sqrt(dx * dx + dy * dy);
            if (d > R) continue;
            best = fmax(best, T[(long long)a * nx + b] - prof(kind, R, cr, tip_r, cot, d));
        }
    }
    S[p] = best;
}

extern "C" __global__ void erode(const double* S, double* C, const int ny, const int nx, const double res, const int r,
                                 const int kind, const double R, const double cr, const double tip_r, const double cot) {
    const long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (p >= (long long)ny * nx) return;
    const int i = (int)(p / nx), j = (int)(p % nx);
    double best = 1e300;
    for (int a = max(i - r, 0); a <= min(i + r, ny - 1); a++) {
        const double dy = (a - i) * res;
        for (int b = max(j - r, 0); b <= min(j + r, nx - 1); b++) {
            const double dx = (b - j) * res;
            const double d = sqrt(dx * dx + dy * dy);
            if (d > R) continue;
            best = fmin(best, S[(long long)a * nx + b] + prof(kind, R, cr, tip_r, cot, d));
        }
    }
    C[p] = best;
}
"""

_K: dict = {}


def _kernel(name: str):
    if name not in _K:
        import cupy as cp
        _K[name] = cp.RawKernel(_SRC, name, options=("--fmad=false",))
    return _K[name]


def _make_nb():
    try:
        import numba
    except ImportError:
        return None

    @numba.njit(cache=True)
    def prof(kind, R, cr, tip_r, cot, d):
        if kind == 0:
            return 0.0
        if kind == 1:
            return R - math.sqrt(max(R * R - d * d, 0.0))
        if kind == 2:
            e = max(d - (R - cr), 0.0)
            return cr - math.sqrt(max(cr * cr - e * e, 0.0))
        return max(d - tip_r, 0.0) * cot

    @numba.njit(parallel=True, cache=True)
    def lift(T, S, gx0, gy0, res, kind, R, cr, tip_r, cot, delta, rcut, out, Bm, bs):
        """bs > 0: Bm = max of T over bs x bs blocks; a segment whose box cannot hold a point above its lowest end
        (block max + delta - min z < -1e-9; every v <= that bound up to rounding) needs no lift: 0, as the full scan."""
        ny, nx = T.shape
        for q in numba.prange(S.shape[0]):
            x0, y0, z0, x1, y1, z1 = S[q, 0], S[q, 1], S[q, 2], S[q, 3], S[q, 4], S[q, 5]
            j0 = max(int(math.floor((min(x0, x1) - R - gx0) / res)), 0)
            j1 = min(int(math.ceil((max(x0, x1) + R - gx0) / res)), nx - 1)
            i0 = max(int(math.floor((min(y0, y1) - R - gy0) / res)), 0)
            i1 = min(int(math.ceil((max(y0, y1) + R - gy0) / res)), ny - 1)
            if bs > 0 and i0 <= i1 and j0 <= j1:
                bmax = -1e300
                for bi in range(i0 // bs, i1 // bs + 1):
                    for bj in range(j0 // bs, j1 // bs + 1):
                        if Bm[bi, bj] > bmax:
                            bmax = Bm[bi, bj]
                if bmax + delta - min(z0, z1) < -1e-9:
                    out[q] = 0.0
                    continue
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            need = -1e300
            for i in range(i0, i1 + 1):
                Y = gy0 + i * res
                for j in range(j0, j1 + 1):
                    X = gx0 + j * res
                    dz = z1 - z0
                    if kind == 0 and L2 >= 1e-12 and abs(dz) >= 1e-9:
                        L = math.sqrt(L2)
                        t = ((X - x0) * dx + (Y - y0) * dy) / L2
                        dp = abs((X - x0) * dy - (Y - y0) * dx) / L
                        if dp > rcut:
                            continue
                        ww = math.sqrt(max(rcut * rcut - dp * dp, 0.0)) / L
                        if t + ww < 0.0 or t - ww > 1.0:
                            continue
                        ta, tb = min(max(t - ww, 0.0), 1.0), min(max(t + ww, 0.0), 1.0)
                        v = T[i, j] + delta - min(z0 + ta * dz, z0 + tb * dz)
                    else:
                        if L2 < 1e-12:
                            d = math.hypot(X - x0, Y - y0)
                        else:
                            t = min(max(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0), 1.0)
                            d = math.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
                        if d > rcut:
                            continue
                        v = T[i, j] + delta - prof(kind, R, cr, tip_r, cot, d) - min(z0, z1)
                    if v > need:
                        need = v
            out[q] = max(0.0, need)

    @numba.njit(cache=True)
    def chord(a, b, lo, hi, xl, xh):
        """Clip [xl, xh] to {X: lo <= a * X + b <= hi}."""
        if a == 0.0:
            if b < lo or b > hi:
                return 1.0, 0.0
            return xl, xh
        p, q = (lo - b) / a, (hi - b) / a
        if p > q:
            p, q = q, p
        return max(xl, p), min(xh, q)

    @numba.njit(cache=True)
    def lift_block(T, x0, y0, z0, x1, y1, z1, gx0, gy0, res, kind, R, cr, tip_r, cot, delta, rcut, ia, ib, ja, jb, need):
        """`lift`'s scan on rows ia..ib, columns ja..jb, but only where a cell can beat need: v <= T + delta - min z
        (the profile is >= 0; 1e-7 covers the rounding of z0 + t dz), and per row only the chord of the capsule of
        radius rcut around the segment (a cell of margin)."""
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        L2 = dx * dx + dy * dy
        zmin = min(z0, z1)
        for i in range(ia, ib + 1):
            Y = gy0 + i * res
            xl, xh = 1e300, -1e300
            for e, cx in ((rcut * rcut - (Y - y0) * (Y - y0), x0), (rcut * rcut - (Y - y1) * (Y - y1), x1)):  # end discs
                if e >= 0.0:
                    r = math.sqrt(e)
                    xl, xh = min(xl, cx - r), max(xh, cx + r)
            if L2 >= 1e-12:  # the rectangle: 0 <= t <= 1 and |offset| <= rcut, both linear in X on this row
                L = math.sqrt(L2)
                a, b = chord(dx / L2, ((Y - y0) * dy - x0 * dx) / L2, 0.0, 1.0, -1e300, 1e300)
                a, b = chord(dy / L, (-x0 * dy - (Y - y0) * dx) / L, -rcut, rcut, a, b)
                if a <= b:
                    xl, xh = min(xl, a), max(xh, b)
            if xl > xh:
                continue
            for j in range(max(ja, int(math.floor((xl - gx0) / res)) - 1), min(jb, int(math.ceil((xh - gx0) / res)) + 1) + 1):
                if T[i, j] + delta - zmin + 1e-7 <= need:
                    continue
                X = gx0 + j * res
                if kind == 0 and L2 >= 1e-12 and abs(dz) >= 1e-9:
                    L = math.sqrt(L2)
                    t = ((X - x0) * dx + (Y - y0) * dy) / L2
                    dp = abs((X - x0) * dy - (Y - y0) * dx) / L
                    if dp > rcut:
                        continue
                    ww = math.sqrt(max(rcut * rcut - dp * dp, 0.0)) / L
                    if t + ww < 0.0 or t - ww > 1.0:
                        continue
                    ta, tb = min(max(t - ww, 0.0), 1.0), min(max(t + ww, 0.0), 1.0)
                    v = T[i, j] + delta - min(z0 + ta * dz, z0 + tb * dz)
                else:
                    if L2 < 1e-12:
                        d = math.hypot(X - x0, Y - y0)
                    else:
                        t = min(max(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0), 1.0)
                        d = math.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
                    if d > rcut:
                        continue
                    v = T[i, j] + delta - prof(kind, R, cr, tip_r, cot, d) - zmin
                if v > need:
                    need = v
        return need

    @numba.njit(parallel=True, cache=True)
    def lift_skip(T, S, gx0, gy0, res, kind, R, cr, tip_r, cot, delta, rcut, out, Bm, bs):
        """`lift` (bs > 0) giving the same out: the answer is max(0, max v), so the scan starts at need = 0 and skips the
        blocks / cells whose bound T + delta - min z cannot beat it; the block of the largest T in the box goes first."""
        ny, nx = T.shape
        for q in numba.prange(S.shape[0]):
            x0, y0, z0, x1, y1, z1 = S[q, 0], S[q, 1], S[q, 2], S[q, 3], S[q, 4], S[q, 5]
            j0 = max(int(math.floor((min(x0, x1) - R - gx0) / res)), 0)
            j1 = min(int(math.ceil((max(x0, x1) + R - gx0) / res)), nx - 1)
            i0 = max(int(math.floor((min(y0, y1) - R - gy0) / res)), 0)
            i1 = min(int(math.ceil((max(y0, y1) + R - gy0) / res)), ny - 1)
            out[q] = 0.0
            if i0 > i1 or j0 > j1:
                continue
            zmin = min(z0, z1)
            bmax, ti, tj = -1e300, i0 // bs, j0 // bs
            for bi in range(i0 // bs, i1 // bs + 1):
                for bj in range(j0 // bs, j1 // bs + 1):
                    if Bm[bi, bj] > bmax:
                        bmax, ti, tj = Bm[bi, bj], bi, bj
            if bmax + delta - zmin < -1e-9:
                continue
            need = lift_block(T, x0, y0, z0, x1, y1, z1, gx0, gy0, res, kind, R, cr, tip_r, cot, delta, rcut,
                              max(i0, ti * bs), min(i1, ti * bs + bs - 1), max(j0, tj * bs), min(j1, tj * bs + bs - 1), 0.0)
            for bi in range(i0 // bs, i1 // bs + 1):
                for bj in range(j0 // bs, j1 // bs + 1):
                    if (bi == ti and bj == tj) or Bm[bi, bj] + delta - zmin + 1e-7 <= need:
                        continue
                    need = lift_block(T, x0, y0, z0, x1, y1, z1, gx0, gy0, res, kind, R, cr, tip_r, cot, delta, rcut,
                                      max(i0, bi * bs), min(i1, bi * bs + bs - 1), max(j0, bj * bs), min(j1, bj * bs + bs - 1), need)
            out[q] = need

    @numba.njit(parallel=True, cache=True)
    def morph(A, res, r, kind, R, cr, tip_r, cot, sign, out):
        """sign +1: out = max(A - h) (dilation, drop-cutter); sign -1: out = min(A + h) (erosion)."""
        ny, nx = A.shape
        for i in numba.prange(ny):
            for j in range(nx):
                best = -1e300 if sign > 0 else 1e300
                for a in range(max(i - r, 0), min(i + r, ny - 1) + 1):
                    dy = (a - i) * res
                    for b in range(max(j - r, 0), min(j + r, nx - 1) + 1):
                        dx = (b - j) * res
                        d = math.sqrt(dx * dx + dy * dy)
                        if d > R:
                            continue
                        hh = prof(kind, R, cr, tip_r, cot, d)
                        if sign > 0:
                            v = A[a, b] - hh
                            if v > best:
                                best = v
                        else:
                            v = A[a, b] + hh
                            if v < best:
                                best = v
                out[i, j] = best

    @numba.njit(cache=True)
    def table(res, r, kind, R, cr, tip_r, cot):
        """Offsets (da, db) of the tool disk and their profile heights, computed as in `morph`."""
        n = (2 * r + 1) * (2 * r + 1)
        da, db, hh = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n)
        m = 0
        for a in range(-r, r + 1):
            dy = a * res
            for b in range(-r, r + 1):
                dx = b * res
                d = math.sqrt(dx * dx + dy * dy)
                if d > R:
                    continue
                da[m], db[m], hh[m] = a, b, prof(kind, R, cr, tip_r, cot, d)
                m += 1
        return da[:m], db[:m], hh[:m]

    @numba.njit(parallel=True, cache=True)
    def morph_sorted(A, da, db, hh, sign, M, need, out):
        """`morph` over offsets sorted by h: stop once the bound M (max / min of A around the cell) cannot beat the best
        (every later offset has a larger h), so the result is the same. Only cells where need is set (need of shape
        (1, 1): every cell); the rest of out is left as it is."""
        ny, nx = A.shape
        every = need.shape[0] == 1 and need.shape[1] == 1
        for i in numba.prange(ny):
            for j in range(nx):
                if not every and not need[i, j]:
                    continue
                best = -1e300 if sign > 0 else 1e300
                m = M[i, j]
                for k in range(hh.shape[0]):
                    h = hh[k]
                    if sign > 0:
                        if m - h <= best:
                            break
                    elif m + h >= best:
                        break
                    a, b = i + da[k], j + db[k]
                    if a < 0 or a >= ny or b < 0 or b >= nx:
                        continue
                    if sign > 0:
                        v = A[a, b] - h
                        if v > best:
                            best = v
                    else:
                        v = A[a, b] + h
                        if v < best:
                            best = v
                out[i, j] = best

    @numba.njit(parallel=True, cache=True)
    def morph_ring(A, da, db, hh, sign, M, need, out, init, k0):
        """`morph_sorted` from offset k0 on, starting at best = init (the result over the offsets before k0)."""
        ny, nx = A.shape
        every = need.shape[0] == 1 and need.shape[1] == 1
        for i in numba.prange(ny):
            for j in range(nx):
                if not every and not need[i, j]:
                    continue
                best = init[i, j]
                m = M[i, j]
                for k in range(k0, hh.shape[0]):
                    h = hh[k]
                    if sign > 0:
                        if m - h <= best:
                            break
                    elif m + h >= best:
                        break
                    a, b = i + da[k], j + db[k]
                    if a < 0 or a >= ny or b < 0 or b >= nx:
                        continue
                    if sign > 0:
                        v = A[a, b] - h
                        if v > best:
                            best = v
                    else:
                        v = A[a, b] + h
                        if v < best:
                            best = v
                out[i, j] = best

    return lift, morph, table, morph_sorted, lift_skip, morph_ring


_NB = _make_nb()


def _args(shape: ToolShape):
    return KIND.get(shape.kind, 3), float(shape.R), float(shape.cr), float(shape.tip_r), float(shape.cot)


class Shield:
    """Target top view T on a grid (column centres x0 + j res, y0 + i res; EMPTY where there is no target)."""

    def __init__(self, T: np.ndarray, x0: float, y0: float, res: float) -> None:
        self.T = np.ascontiguousarray(np.where(np.isfinite(T), T, EMPTY), np.float64)
        self.x0, self.y0, self.res = float(x0), float(y0), float(res)
        self.ny, self.nx = self.T.shape

    @classmethod
    def for_setup(cls, part_k: dict, stock_k: dict, tools: list[dict], tol: float, max_cells: float,
                  operations: list[dict] | None = None) -> "Shield":
        """The verifier's grid and target view for this setup (verify.simulate_setup builds exactly these)."""
        from .dexel import stock_dexel
        from .verify import sim_grid
        from .grid_tools import grid_tool_selection, grid_receipt
        tool_max_r, selection = grid_tool_selection(tools, operations)
        bb = [min(part_k["bbox"][0], stock_k["bbox"][0]), min(part_k["bbox"][1], stock_k["bbox"][1]),
              max(part_k["bbox"][3], stock_k["bbox"][3]), max(part_k["bbox"][4], stock_k["bbox"][4])]
        grid = sim_grid(bb, tool_max_r + 1.0, tol, max_cells)
        _, th = stock_dexel(grid, stock_k, part_k, return_target=True)
        sh = cls(th, grid.x0, grid.y0, grid.res)
        sh.grid = grid
        sh.grid_selection = grid_receipt(selection, grid)
        return sh

    # ------------------------------------------------------------------------------------------------ lifts
    def lifts(self, segs: np.ndarray, shape: ToolShape, delta: float, eps_xy: float = 0.0, gpu_min: int = 2000) -> np.ndarray:
        S = np.ascontiguousarray(np.asarray(segs, np.float64).reshape(-1, 6))
        out = np.zeros(len(S))
        if not len(S):
            return out
        kind, R, cr, tip_r, cot = _args(shape)
        rcut = max(R - eps_xy, 0.0)
        if len(S) >= gpu_min and _gpu_ready():
            try:
                import cupy as cp
                d_out = cp.zeros(len(S))
                _kernel("lift")((len(S),), (256,), (cp.asarray(self.T), cp.asarray(S), d_out, np.int64(len(S)), np.int32(self.ny), np.int32(self.nx),
                                                    np.float64(self.x0), np.float64(self.y0), np.float64(self.res), np.int32(kind), np.float64(R),
                                                    np.float64(cr), np.float64(tip_r), np.float64(cot), np.float64(delta), np.float64(rcut)))
                return d_out.get()
            except Exception:  # noqa: BLE001  (fall back to the CPU)
                pass
        if _NB is not None:
            from .cpu_fast import enabled
            bs = LIFT_BLOCK if enabled() else 0  # CAM_CPU_FAST: skip blocks / cells that cannot raise the lift (same result)
            (_NB[4] if bs else _NB[0])(self.T, S, self.x0, self.y0, self.res, kind, R, cr, tip_r, cot, float(delta), rcut,
                                       out, self._block_max(bs), bs)
            return out
        prof = shape.profile
        for q, (x0, y0, z0, x1, y1, z1) in enumerate(S):  # reference implementation (slow)
            j0, j1 = max(int(math.floor((min(x0, x1) - R - self.x0) / self.res)), 0), min(int(math.ceil((max(x0, x1) + R - self.x0) / self.res)), self.nx - 1)
            i0, i1 = max(int(math.floor((min(y0, y1) - R - self.y0) / self.res)), 0), min(int(math.ceil((max(y0, y1) + R - self.y0) / self.res)), self.ny - 1)
            if j1 < j0 or i1 < i0:
                continue
            X = self.x0 + np.arange(j0, j1 + 1)[None, :] * self.res
            Y = self.y0 + np.arange(i0, i1 + 1)[:, None] * self.res
            dx, dy = x1 - x0, y1 - y0
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 < 1e-12 else np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0, 1.0)
            d = np.hypot(X - (x0 + t * dx), Y - (y0 + t * dy))
            m = d <= rcut
            if m.any():
                out[q] = max(0.0, float((self.T[i0:i1 + 1, j0:j1 + 1] + delta - prof(np.minimum(d, R)))[m].max()) - min(z0, z1))
        return out

    def _block_max(self, bs: int) -> np.ndarray:
        """Max of T over bs x bs blocks (cached on the shield; T does not change)."""
        if bs <= 0:
            return np.zeros((1, 1))
        bm = getattr(self, "_bm", None)
        if bm is None or bm[0] != bs:
            nby, nbx = -(-self.ny // bs), -(-self.nx // bs)
            P = np.full((nby * bs, nbx * bs), EMPTY)
            P[:self.ny, :self.nx] = self.T
            self._bm = bm = (bs, np.ascontiguousarray(P.reshape(nby, bs, nbx, bs).max(axis=(1, 3))))
        return bm[1]

    # ------------------------------------------------------------------------------------------------ drop cutter
    def _morph(self, A: np.ndarray, shape: ToolShape, sign: int, need: np.ndarray | None = None) -> np.ndarray:
        """need (CAM_CPU_FAST, ball / bull on the CPU): the cells to compute, the rest is NaN; A outside need dilated by
        the tool disk is not read. Otherwise (GPU, flat, flag off) every cell is computed."""
        kind, R, cr, tip_r, cot = _args(shape)
        r = int(math.ceil(R / self.res))
        A = np.ascontiguousarray(A, np.float64)
        ny, nx = A.shape
        if _gpu_ready():
            try:
                import cupy as cp
                d_out = cp.empty((ny, nx))
                n = ny * nx
                _kernel("dilate" if sign > 0 else "erode")(((n + 255) // 256,), (256,), (cp.asarray(A), d_out, np.int32(ny), np.int32(nx), np.float64(self.res),
                                                                                      np.int32(r), np.int32(kind), np.float64(R), np.float64(cr),
                                                                                      np.float64(tip_r), np.float64(cot)))
                return d_out.get()
            except Exception:  # noqa: BLE001
                pass
        out = np.empty((ny, nx))
        from .cpu_fast import enabled
        if _NB is not None and enabled():  # sorted offsets with a bound (same result, several times faster)
            from scipy.ndimage import maximum_filter1d, minimum_filter1d
            f = maximum_filter1d if sign > 0 else minimum_filter1d
            # NaN (not computed) cells cannot win: the bound stays one as long as the disks it serves have no NaN
            B = np.where(np.isnan(A), -np.inf if sign > 0 else np.inf, A) if need is not None else A
            M = f(f(B, 2 * r + 1, axis=0, mode="nearest"), 2 * r + 1, axis=1, mode="nearest")
            da, db, hh = _NB[2](float(self.res), r, kind, R, cr, tip_r, cot)
            if kind == 0:  # flat: h = 0 on the disk, a plain dilation / erosion (outside cells = edge cells, never nearer)
                from .cpu_fast import dilate
                fp = np.zeros((2 * r + 1, 2 * r + 1), bool)
                fp[da + r, db + r] = True
                return dilate(A, fp) if sign > 0 else -dilate(-A, fp)
            o = np.argsort(hh, kind="stable")
            da, db, hh = da[o], db[o], hh[o]
            if need is None:
                need = np.ones((1, 1), bool)
            else:
                out[:] = np.nan
            need = np.ascontiguousarray(need, bool)
            k0 = int(np.count_nonzero(hh == 0.0))  # the flat core of the tool (bull: radius R - cr): h = 0 there
            if k0 > 1:  # the core as a plain dilation / erosion of B by rows, then the sorted ring from that value on
                from .cpu_fast import dilate
                fp = np.zeros((2 * r + 1, 2 * r + 1), bool)
                fp[da[:k0] + r, db[:k0] + r] = True
                if sign > 0:  # A - 0 = A; edge cells repeated are core cells too, never nearer
                    init = np.maximum(dilate(B, fp), -1e300)
                else:  # A + 0 turns -0 into +0, as the kernel does
                    init = np.minimum(-dilate(-B, fp) + 0.0, 1e300)
                _NB[5](A, da, db, hh, sign, M, need, out, init, k0)
                return out
            _NB[3](A, da, db, hh, sign, M, need, out)
            return out
        if _NB is not None:
            _NB[1](A, self.res, r, kind, R, cr, tip_r, cot, sign, out)
            return out
        raise RuntimeError("drop-cutter needs numba or a CUDA device")

    def disk(self, shape: ToolShape) -> np.ndarray:
        """Footprint ((2r + 1)^2, centred) of the cells `_morph` reads around a cell for this tool."""
        kind, R, cr, tip_r, cot = _args(shape)
        r = int(math.ceil(R / self.res))
        da, db, _ = _NB[2](float(self.res), r, kind, R, cr, tip_r, cot)
        fp = np.zeros((2 * r + 1, 2 * r + 1), bool)
        fp[da + r, db + r] = True
        return fp

    def drop(self, shape: ToolShape, window=None, need: np.ndarray | None = None) -> np.ndarray:
        """Lowest gouge-free tip height at every cell (optionally on a window (i0, i1, j0, j1) of the grid; need: see
        `_morph`)."""
        T = self.T if window is None else self.T[window[0]:window[1], window[2]:window[3]]
        return self._morph(T, shape, +1, need)

    def closing(self, S: np.ndarray, shape: ToolShape, need: np.ndarray | None = None) -> np.ndarray:
        return self._morph(S, shape, -1, need)


# ---------------------------------------------------------------------------------------------------- G-code
_WORD = re.compile(r"([GXYZF])\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)", re.I)


def shield_gcode(lines: list[str], sh: Shield, shape: ToolShape, delta: float, eps_xy: float = 0.0, piece: float | None = None) -> tuple[list[str], dict]:
    """Certify / repair one operation's G-code (G0/G1, absolute XYZ, as the height-map strategies write it)."""
    piece = piece or max(2.0 * sh.res, 0.25)
    rep = {"moves": 0, "pieces": 0, "lifted_pieces": 0, "max_lift_mm": 0.0, "lifted_len_mm": 0.0, "delta_mm": round(delta, 4), "eps_xy_mm": round(eps_xy, 4),
           "certified": True}
    pos = [None, None, None]
    motion, feed = None, None
    recs = []
    for ln in lines:
        body = ln.split(";")[0].split("(")[0]
        words = _WORD.findall(body)
        gs = [float(v) for k, v in words if k.upper() == "G"]
        if any(g in (2.0, 3.0) for g in gs):  # arcs: not written by the height-map strategies; leave the operation alone
            rep.update(certified=None, note="arcs: not checked")
            return lines, rep
        for g in gs:
            if g in (0.0, 1.0):
                motion = int(g)
        fs = [float(v) for k, v in words if k.upper() == "F"]
        if fs:
            feed = fs[-1]
        ax = {k.upper(): float(v) for k, v in words if k.upper() in ("X", "Y", "Z")}
        new = [ax.get("X", pos[0]), ax.get("Y", pos[1]), ax.get("Z", pos[2])]
        if not ax or motion is None or None in pos or None in new:
            recs.append(None)
        else:
            recs.append((motion, tuple(pos), tuple(new), feed))
        if ax:
            pos = new
    idx = [k for k, r in enumerate(recs) if r is not None]
    if not idx:
        return lines, rep
    P0 = np.array([recs[k][1] for k in idx], float)
    P1 = np.array([recs[k][2] for k in idx], float)
    mode = np.array([recs[k][0] for k in idx])
    L = np.hypot(P1[:, 0] - P0[:, 0], P1[:, 1] - P0[:, 1])
    n = np.where(mode == 0, 1, np.maximum(1, np.ceil(L / piece))).astype(np.int64)
    if KIND.get(shape.kind, 3) != 0:  # profiled tool: inclined moves in pieces with dz <= cell / 4, as the sweep does
        n = np.where((mode != 0) & (L > 1e-9), np.maximum(n, np.ceil(np.abs(P1[:, 2] - P0[:, 2]) / (sh.res / 4))), n).astype(np.int64)
    rep_i = np.repeat(np.arange(len(idx)), n)
    p = np.arange(len(rep_i)) - np.repeat(np.cumsum(n) - n, n)
    a, b = (p / n[rep_i])[:, None], ((p + 1) / n[rep_i])[:, None]
    A = P0[rep_i] + a * (P1[rep_i] - P0[rep_i])
    B = P0[rep_i] + b * (P1[rep_i] - P0[rep_i])
    lam = sh.lifts(np.hstack([A, B]), shape, delta, eps_xy)
    lam = np.where(lam > 1e-6, np.ceil(lam / 0.01 - 1e-9) * 0.01, 0.0)  # up to 0.01 mm: neighbouring pieces merge into one move
    rep.update(moves=len(idx), pieces=int(len(lam)), lifted_pieces=int((lam > 0).sum()), max_lift_mm=round(float(lam.max()), 4),
               lifted_len_mm=round(float((np.hypot(B[:, 0] - A[:, 0], B[:, 1] - A[:, 1]) * (lam > 0)).sum()), 2), certified=bool(not (lam > 0).any()))
    if rep["certified"]:
        return lines, rep
    first = np.cumsum(n) - n
    up = lambda z: math.ceil(z * 1e4 - 1e-6) / 1e4  # noqa: E731  (never round a lifted height down)
    out, cur = [], None
    q = 0
    for k, ln in enumerate(lines):
        r = recs[k]
        if r is None:
            out.append(ln)
            z = _WORD.findall(ln.split(";")[0].split("(")[0])
            zz = [float(v) for kk, v in z if kk.upper() == "Z"]
            if zz:
                cur = zz[-1]
            continue
        m, p0, p1, f = r
        lo, hi = first[q], first[q] + n[q]
        ls = lam[lo:hi]
        q += 1
        if not ls.any() and (cur is None or abs(cur - p0[2]) < 1e-9):
            out.append(ln)
            cur = p1[2]
            continue
        G = "G0" if m == 0 else "G1"
        F = "" if m == 0 or f is None else f" F{f:.3f}"
        s = 0
        while s < len(ls):  # runs of equal lift become one move
            e = s
            while e + 1 < len(ls) and abs(ls[e + 1] - ls[s]) < 1e-9:
                e += 1
            ta, tb = s / len(ls), (e + 1) / len(ls)
            zs = p0[2] + ta * (p1[2] - p0[2]) + ls[s]
            xe, ye, ze = (p0[0] + tb * (p1[0] - p0[0]), p0[1] + tb * (p1[1] - p0[1]), p0[2] + tb * (p1[2] - p0[2]) + ls[s])
            if ls[s] > 0:
                zs, ze = up(zs), up(ze)
            if cur is None or abs(cur - zs) > 1e-9:
                out.append(f"{G} Z{zs:.4f}{F}")  # vertical link at the start of the run
            out.append(f"{G} X{xe:.4f} Y{ye:.4f} Z{ze:.4f}{F}")
            cur = ze
            s = e + 1
    return out, rep
