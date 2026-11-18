"""Tool accessibility of a part from the six axis directions (height-map projection of the tessellated part)."""
from __future__ import annotations

import numpy as np

DIRS = {"+Z": (0, 1, 2, 1), "-Z": (0, 1, 2, -1), "+X": (1, 2, 0, 1), "-X": (1, 2, 0, -1), "+Y": (2, 0, 1, 1), "-Y": (2, 0, 1, -1)}


def heightmap(V, F, a, b, c, sign, n=220):
    P = np.c_[V[:, a], V[:, b], V[:, c] * sign]
    lo, hi = P.min(0), P.max(0)
    res = max(hi[0] - lo[0], hi[1] - lo[1]) / n
    nx, ny = int((hi[0] - lo[0]) / res) + 2, int((hi[1] - lo[1]) / res) + 2
    hm = np.full((ny, nx), -np.inf)
    for t in P[F]:
        j0, j1 = int((t[:, 0].min() - lo[0]) / res), int((t[:, 0].max() - lo[0]) / res) + 2
        i0, i1 = int((t[:, 1].min() - lo[1]) / res), int((t[:, 1].max() - lo[1]) / res) + 2
        X, Y = np.meshgrid(lo[0] + (np.arange(j0, min(j1, nx)) + 0.5) * res, lo[1] + (np.arange(i0, min(i1, ny)) + 0.5) * res)
        (xa, ya, za), (xb, yb, zb), (xc, yc, zc) = t
        det = (yb - yc) * (xa - xc) + (xc - xb) * (ya - yc)
        if abs(det) < 1e-12:
            continue
        l1 = ((yb - yc) * (X - xc) + (xc - xb) * (Y - yc)) / det
        l2 = ((yc - ya) * (X - xc) + (xa - xc) * (Y - yc)) / det
        m = (l1 >= -1e-9) & (l2 >= -1e-9) & (1 - l1 - l2 >= -1e-9)
        if m.any():
            z = l1 * za + l2 * zb + (1 - l1 - l2) * zc
            sub = hm[i0:i0 + X.shape[0], j0:j0 + X.shape[1]]
            sub[m] = np.maximum(sub[m], z[m])
    return hm, res * res, lo[2]


def accessibility(vertices, faces, volume) -> dict:
    """{direction: share of the part volume a tool from that side can produce without undercuts (1.0 = fully)}."""
    V, F = np.asarray(vertices, float), np.asarray(faces, int)
    out = {}
    for name, (a, b, c, s) in DIRS.items():
        hm, cell, base = heightmap(V, F, a, b, c, s)
        under = np.where(np.isfinite(hm), hm - base, 0).sum() * cell
        out[name] = round(float(volume / under), 3) if under > 0 else 0.0
    return out
