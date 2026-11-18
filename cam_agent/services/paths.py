"""Toolpaths generated outside FreeCAD and inserted as Custom operations: helical counterbores and helical thread milling.
Feeds are written in mm/s (FreeCAD's internal unit for Custom operations), arcs as G2/G3 with I/J (XY plane)."""
from __future__ import annotations

import math


def _f(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def helical_counterbore(holes: list[dict], tool_d: float, doc: float, stepover: float, feed: float, plunge: float, safe_z: float) -> list[str]:
    """Counterbore (DIN 974): helical entry on the bore circle, then concentric rings out to the counterbore wall, down
    in steps of `doc` to the counterbore floor. Holes: [{"x","y","top_z","diameter","depth"}] (counterbore diameter/depth)."""
    f, fp = feed / 60.0, plunge / 60.0
    r_t = tool_d / 2
    out = []
    for h in holes:
        R = h["diameter"] / 2 - r_t
        if R <= 0.05:
            continue
        cx, cy, top, floor = h["x"], h["y"], h["top_z"], h["top_z"] - h["depth"]
        r0 = min(R, max(0.3 * tool_d, 0.2))
        out += [f"G0 Z{_f(safe_z)}", f"G0 X{_f(cx + r0)} Y{_f(cy)}", f"G0 Z{_f(top + 1.0)}"]
        z = top
        while z > floor + 1e-6:
            zn = max(floor, z - doc)
            out.append(f"G3 X{_f(cx + r0)} Y{_f(cy)} Z{_f(zn)} I{_f(-r0)} J0 F{_f(fp)}")  # one helical turn
            out.append(f"G3 X{_f(cx + r0)} Y{_f(cy)} I{_f(-r0)} J0 F{_f(f)}")              # flat turn at this depth
            r = r0
            while r < R - 1e-6:
                r = min(R, r + stepover)
                out += [f"G1 X{_f(cx + r)} Y{_f(cy)} F{_f(f)}", f"G3 X{_f(cx + r)} Y{_f(cy)} I{_f(-r)} J0 F{_f(f)}"]
            out.append(f"G1 X{_f(cx + r0)} Y{_f(cy)} F{_f(f)}")
            z = zn
        out.append(f"G0 Z{_f(safe_z)}")
    return out


def thread_milling(holes: list[dict], tool_d: float, pitch: float, major_d: float, feed: float, safe_z: float, right_hand: bool = True) -> list[str]:
    """Internal thread, climb milling bottom-up: centre -> bottom of the thread -> arc in -> one helix turn per pitch up to
    the top -> arc out. Holes: [{"x","y","top_z","thread_depth"}]. Feed is at the tool centre."""
    f = feed / 60.0
    R = (major_d - tool_d) / 2
    out = []
    for h in holes:
        cx, cy, top = h["x"], h["y"], h["top_z"]
        z = top - h["thread_depth"]
        turns = max(1, int(math.ceil(h["thread_depth"] / pitch)) + 1)
        out += [f"G0 Z{_f(safe_z)}", f"G0 X{_f(cx)} Y{_f(cy)}", f"G0 Z{_f(top + 1.0)}", f"G1 Z{_f(z)} F{_f(f * 3)}"]
        g = "G3" if right_hand else "G2"  # bottom-up climb for a right-hand thread
        out.append(f"{g} X{_f(cx + R)} Y{_f(cy)} Z{_f(z + pitch / 2)} I{_f(R / 2)} J0 F{_f(f)}")  # half-turn arc-in (radius R/2)
        z += pitch / 2
        for _ in range(turns):
            z += pitch
            out.append(f"{g} X{_f(cx + R)} Y{_f(cy)} Z{_f(z)} I{_f(-R)} J0 F{_f(f)}")
        out += [f"G1 X{_f(cx)} Y{_f(cy)} F{_f(f)}", f"G0 Z{_f(safe_z)}"]
    return out
