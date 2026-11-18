"""GPU (CUDA, CuPy) version of the dexel sweep in dexel.py: the same analytic swept volume of the tool per move.

Every move lowers the columns under it to the height of the tool's cutting surface (dexel.DexelMap.cut / the numba
kernel): the result is the minimum over the moves, so the moves of one operation are independent and run in parallel,
one CUDA block per move, the threads of the block over the move's footprint cells, the column lowered with an atomic
minimum. Inclined moves of profiled tools are split into horizontal pieces exactly as on the CPU (dz <= cell / 4).
FMA contraction is off so that the arithmetic matches the CPU kernel.

Optional dependency: cupy-cuda12x. Use:
    gm = GpuDexel(dm)                      # copies the dexel map to the device
    gm.run_moves(moves, shape, k)          # per operation, as DexelMap.run_moves
    gm.download()                          # h and owner back into dm
"""
from __future__ import annotations

import numpy as np

_SRC = r"""
__device__ __forceinline__ double prof(int kind, double R, double cr, double tip_r, double cot, double d) {
    if (kind == 0) return 0.0;
    if (kind == 1) return R - sqrt(fmax(R * R - d * d, 0.0));
    if (kind == 2) { double e = fmax(d - (R - cr), 0.0); return cr - sqrt(fmax(cr * cr - e * e, 0.0)); }
    return fmax(d - tip_r, 0.0) * cot;
}

__device__ __forceinline__ void atomic_min_double(double* addr, double val) {
    unsigned long long* p = (unsigned long long*)addr;
    unsigned long long old = *p, assumed;
    while (val < __longlong_as_double(old)) {
        assumed = old;
        old = atomicCAS(p, assumed, __double_as_longlong(val));
        if (old == assumed) break;
    }
}

extern "C" __global__ void sweep(double* h, const double* S, const long long nseg, const int ny, const int nx,
                                 const double gx0, const double gy0, const double res, const int kind,
                                 const double R, const double cr, const double tip_r, const double cot) {
    long long q = blockIdx.x;
    if (q >= nseg) return;
    const double x0 = S[6 * q], y0 = S[6 * q + 1], z0 = S[6 * q + 2], x1 = S[6 * q + 3], y1 = S[6 * q + 4], z1 = S[6 * q + 5];
    int j0 = max((int)floor((fmin(x0, x1) - R - gx0) / res), 0);
    int j1 = min((int)ceil((fmax(x0, x1) + R - gx0) / res), nx - 1);
    int i0 = max((int)floor((fmin(y0, y1) - R - gy0) / res), 0);
    int i1 = min((int)ceil((fmax(y0, y1) + R - gy0) / res), ny - 1);
    if (j1 < j0 || i1 < i0) return;
    const double dx = x1 - x0, dy = y1 - y0, dz = z1 - z0;
    const double L2 = dx * dx + dy * dy;
    const double L = sqrt(L2);
    const long long w = j1 - j0 + 1;
    const long long n = w * (long long)(i1 - i0 + 1);
    for (long long c = threadIdx.x; c < n; c += blockDim.x) {
        const int i = i0 + (int)(c / w), j = j0 + (int)(c % w);
        const double X = gx0 + j * res, Y = gy0 + i * res;
        double cv;
        if (L2 < 1e-12) {
            const double d = hypot(X - x0, Y - y0);
            if (d > R) continue;
            cv = fmin(z0, z1) + prof(kind, R, cr, tip_r, cot, d);
        } else if (fabs(dz) < 1e-9 || kind != 0) {
            const double t = fmin(fmax(((X - x0) * dx + (Y - y0) * dy) / L2, 0.0), 1.0);
            const double d = hypot(X - (x0 + t * dx), Y - (y0 + t * dy));
            if (d > R) continue;
            cv = z0 + prof(kind, R, cr, tip_r, cot, d);
        } else {
            const double t = ((X - x0) * dx + (Y - y0) * dy) / L2;
            const double dp = fabs((X - x0) * dy - (Y - y0) * dx) / L;
            if (dp > R) continue;
            const double ww = sqrt(fmax(R * R - dp * dp, 0.0)) / L;
            if (t + ww < 0.0 || t - ww > 1.0) continue;
            const double ta = fmin(fmax(t - ww, 0.0), 1.0), tb = fmin(fmax(t + ww, 0.0), 1.0);
            cv = fmin(z0 + ta * dz, z0 + tb * dz);
        }
        double* cell = &h[(long long)i * nx + j];
        if (cv < *cell - 1e-7) atomic_min_double(cell, cv);
    }
}
"""

KIND = {"flat": 0, "ball": 1, "bull": 2}


def expand_segments(moves, shape, res: float) -> np.ndarray:
    """Cut moves -> (n, 6) segments, inclined moves of profiled tools split into horizontal pieces as on the CPU."""
    return split_segments(np.asarray([mv[:6] for mv in moves if mv[6]], np.float64).reshape(-1, 6), shape, res)


def split_segments(S: np.ndarray, shape, res: float) -> np.ndarray:
    """(n, 6) cut segments; inclined segments of profiled tools split into horizontal pieces (dz <= cell / 4)."""
    if not len(S) or KIND.get(shape.kind, 3) == 0:
        return S
    dz = S[:, 5] - S[:, 2]
    split = (np.abs(dz) >= 1e-9) & ~((S[:, 0] == S[:, 3]) & (S[:, 1] == S[:, 4]))
    if not split.any():
        return S
    out = [S[~split]]
    T = S[split]
    n = np.minimum(400, np.maximum(1, np.ceil(np.abs(T[:, 5] - T[:, 2]) / (res / 4)))).astype(np.int64)
    rep = np.repeat(np.arange(len(T)), n)
    p = np.arange(len(rep)) - np.repeat(np.cumsum(n) - n, n)
    a, b = p / n[rep], (p + 1) / n[rep]
    t = T[rep]
    zm = t[:, 2] + (a + b) / 2 * (t[:, 5] - t[:, 2])
    pieces = np.stack([t[:, 0] + a * (t[:, 3] - t[:, 0]), t[:, 1] + a * (t[:, 4] - t[:, 1]), zm,
                       t[:, 0] + b * (t[:, 3] - t[:, 0]), t[:, 1] + b * (t[:, 4] - t[:, 1]), zm], axis=1)
    out.append(pieces)
    return np.concatenate(out)


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp
        _KERNEL = cp.RawKernel(_SRC, "sweep", options=("--fmad=false",))
    return _KERNEL


def sweep_inplace(dm, segs: np.ndarray, shape, k: int, threads: int = 256) -> None:
    """DexelMap.run_moves on the GPU for one operation: the map is uploaded, swept and downloaded (h, owner in place)."""
    import cupy as cp
    g = dm.g
    S = split_segments(np.asarray(segs, np.float64).reshape(-1, 6), shape, g.res)
    if not len(S):
        return
    h = cp.asarray(dm.h)
    before = h.copy()
    _kernel()((len(S),), (threads,),
              (h, cp.asarray(S), np.int64(len(S)), np.int32(g.ny), np.int32(g.nx), np.float64(g.x0), np.float64(g.y0),
               np.float64(g.res), np.int32(KIND.get(shape.kind, 3)), np.float64(shape.R), np.float64(shape.cr),
               np.float64(shape.tip_r), np.float64(shape.cot)))
    lowered = (h < before - 1e-7).get()
    dm.h[...] = h.get()
    dm.owner[lowered] = k  # every move of one call belongs to operation k


class GpuDexel:
    """A DexelMap mirrored on the GPU (h, owner); run the operations, then download()."""

    _kernel = None

    def __init__(self, dm, threads: int = 256) -> None:
        import cupy as cp
        self.cp, self.dm, self.threads = cp, dm, threads
        if GpuDexel._kernel is None:
            GpuDexel._kernel = cp.RawKernel(_SRC, "sweep", options=("--fmad=false",))
        self.h = cp.asarray(dm.h, dtype=cp.float64)
        self.owner = cp.asarray(dm.owner)

    def run_moves(self, moves, shape, k: int) -> int:
        """Returns the number of segments swept (after splitting)."""
        if shape is None:
            return 0
        g = self.dm.g
        S = expand_segments(moves, shape, g.res)
        if not len(S):
            return 0
        cp = self.cp
        before = self.h.copy()
        dS = cp.asarray(S)
        GpuDexel._kernel((len(S),), (self.threads,),
                         (self.h, dS, np.int64(len(S)), np.int32(g.ny), np.int32(g.nx), np.float64(g.x0), np.float64(g.y0),
                          np.float64(g.res), np.int32(KIND.get(shape.kind, 3)), np.float64(shape.R), np.float64(shape.cr),
                          np.float64(shape.tip_r), np.float64(shape.cot)))
        self.owner[self.h < before - 1e-7] = k  # every move of one call belongs to operation k
        return len(S)

    def synchronize(self) -> None:
        self.cp.cuda.Device().synchronize()

    def download(self) -> None:
        self.dm.h[...] = self.h.get()
        self.dm.owner[...] = self.owner.get()
