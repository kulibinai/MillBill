"""Corrections of the posted G-code that FreeCAD's drilling operation cannot express.

FreeCAD's Drilling op emits G81/G82/G83/G73 for every hole operation. That is wrong and unsafe for:
  * tapping: G81 retracts at rapid WITHOUT reversing the spindle -> broken tap / stripped thread. Rewritten to a rigid
    tapping cycle: G84 (ISO / Fanuc / Haas / LinuxCNC 2.9+, optional M29 S.. for Fanuc rigid mode) with F = pitch x S,
    or LinuxCNC G33.1 Z.. K<pitch> per hole (CAM_TAP_CYCLE=G33.1);
  * reaming and boring: rapid retract scores the finished bore -> G85 (feed in, feed out).
Operations are found by the "(begin operation: <label>)" comments and the step's process from the request.
"""
from __future__ import annotations

import re
from .cnc import safe_comment_label, flat_comment_label, code_body

_OP = re.compile(r"\(\s*begin operation\s*:\s*(.*)\)\s*$", re.I)
_CYCLE = re.compile(r"\bG(73|81|82|83)\b")
_WORD = re.compile(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))")


def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


def fix_program(path: str, operations: list[dict], tap_cycle: str = "G84", m29: bool = False) -> dict:
    """Rewrite the canned cycles of tapping / reaming / boring operations in place. Returns a change report."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    by_label = {o["label"]: o for o in operations}
    by_step = {f"{o['step']:02d}": o for o in operations}
    out, changes = [], []
    cur, spindle, in_cycle, tap_points, cycle_r, cycle_z = None, None, False, [], None, None
    for raw in lines:
        code = code_body(raw).upper()
        m = _OP.search(raw) if not code.strip() else None
        if m:
            if cur and cur.get("process") == "tapping" and tap_cycle == "G33.1" and tap_points:
                out += _g33(tap_points, cycle_r, cycle_z, cur, spindle)
                tap_points = []
            label = m.group(1).strip()
            cur = by_label.get(label) or by_step.get(label[:2])
            in_cycle = False
            out.append(raw)
            continue
        s_match = re.search(r"\bS\s*([\d.]+)", code)
        if s_match:
            spindle = float(s_match.group(1))
        proc = (cur or {}).get("process")
        if proc not in ("tapping", "reaming", "boring"):
            out.append(raw)
            continue
        if re.search(r"\bG80\b", code):
            in_cycle = False
        cyc = _CYCLE.search(code)
        continuation = in_cycle and not re.search(r"\bG[0-3]\b", code) and re.search(r"\b[XY]", code)
        if not cyc and not continuation:
            if proc == "tapping" and tap_cycle == "G33.1" and re.search(r"\bG80\b", code) and tap_points:
                out += _g33(tap_points, cycle_r, cycle_z, cur, spindle)
                tap_points = []
            out.append(raw)
            continue
        in_cycle = True
        words = dict(_WORD.findall(code))
        if "R" in words:
            cycle_r = float(words["R"])
        if "Z" in words:
            cycle_z = float(words["Z"])
        if proc == "tapping":
            pitch = float(cur.get("pitch_mm") or 1.0)
            if tap_cycle == "G33.1":
                tap_points.append((words.get("X"), words.get("Y")))
                changes.append({"step": cur["step"], "process": proc, "from": raw.strip(), "to": "G33.1 rigid tapping"})
                continue
            new = re.sub(r"\s*\bQ\s*[-+]?[\d.]+", "", raw)
            new = re.sub(r"\s*\bP\s*[-+]?[\d.]+", "", new)
            if cyc:
                new = _CYCLE.sub("G84", new)
                f = pitch * (spindle or float(cur.get("spindle_rpm") or 0))
                new = re.sub(r"\bF\s*[-+]?[\d.]+", "", new).rstrip() + (f" F{_fmt(f)}" if f else "")
                if m29 and spindle:
                    out.append(f"M29 S{int(spindle)}")
            changes.append({"step": cur["step"], "process": proc, "from": raw.strip(), "to": new.strip()})
            out.append(new)
        else:  # reaming / boring: feed in, feed out
            new = re.sub(r"\s*\bQ\s*[-+]?[\d.]+", "", raw)
            if cyc:
                new = _CYCLE.sub("G85", new)
            changes.append({"step": cur["step"], "process": proc, "from": raw.strip(), "to": new.strip()})
            out.append(new)
    if cur and cur.get("process") == "tapping" and tap_cycle == "G33.1" and tap_points:
        out += _g33(tap_points, cycle_r, cycle_z, cur, spindle)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(out) + "\n")
    return {"changes": len(changes), "details": changes[:50], "tap_cycle": tap_cycle}


def _g33(points, r, z, op, spindle) -> list[str]:
    pitch = float(op.get("pitch_mm") or 1.0)
    out = [f"(rigid tapping {op.get('thread', '')}: G33.1, pitch {pitch:g})"]
    for x, y in points:
        xy = " ".join(w for w in (f"X{x}" if x is not None else "", f"Y{y}" if y is not None else "") if w)
        out += [f"G0 {xy}".strip(), f"G0 Z{_fmt(r or 2.0)}", f"G33.1 Z{_fmt(z or 0.0)} K{_fmt(pitch)}"]
    out.append(f"G0 Z{_fmt((r or 2.0) + 5.0)}")
    return out


def safety_report(parsed: dict, operations: list[dict]) -> list[str]:
    """Problems that would hurt the machine or the part: tapping / reaming not done with the right cycle."""
    issues = []
    by_name = {label: o for o in operations for label in (o["label"], safe_comment_label(o["label"]), flat_comment_label(o["label"]))}
    for o in parsed["ops"]:
        rq = by_name.get(o["name"]) or {}
        cyc = {c["cycle"] for c in o.get("cycles", [])}
        if rq.get("process") == "tapping" and cyc and not cyc <= {"G84", "G74", "G33.1"}:
            issues.append(f"safety: {o['name']} taps with {sorted(cyc)} instead of a rigid tapping cycle (G84 / G33.1)")
        if rq.get("process") in ("reaming", "boring") and cyc & {"G81", "G83", "G73"}:
            issues.append(f"safety: {o['name']} {rq['process']} retracts at rapid ({sorted(cyc)}): use G85")
    return issues
