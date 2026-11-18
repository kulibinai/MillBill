"""Harness-independent evaluation: the same task for any planner, the same metrics for any answer.

  export-task  part.step [--stock blank.step] --material ... --notes ...  ->  task/ (task.json, prompt.md, part + stock STEP)
               The default stock is written as a real STEP box (part bounding box + margins in the part's +Z frame), so the
               harness (`run --stock task/stock.step`) and a model without the harness machine exactly the same blank.
  evaluate     --task task/ (--answer answer_dir | --session outputs/<session>)  ->  metrics.json + metrics.md

Answer format (what the model without the harness must return, described in prompt.md):
  answer.json = {"material": str, "setups": [{"side": "+Z", "gcode_file": "setup1.ngc",
                   "tools": [{"t": 1, "type": <tool type>, "diameter_mm": float, "flutes": int, "tip_angle_deg": float, "corner_radius_mm": float,
                              "tool_material": "carbide"|"HSS"}],
                   "operations": [{"step": int, "process": <process name>, "tool": int, "features": str, "spindle_rpm": float,
                                   "feed_mm_min": float, "doc_mm": float, "woc_mm": float}]}]}
  G-code of a setup is written in that setup's frame: the part rotated so that `side` points up (+Z), with the minimum
  corner of the PART's bounding box at X0 Y0 Z0 (identical to the harness).

Metrics: surface share within tolerance (all setups combined, the harness's surface metric: points on the B-rep faces,
probes along the normals into the exact swept-volume workpiece; services/verify.py),
material left / overcut, per-setup views, cycle time, tool changes (and avoidable ones), standard-practice ordering
violations, cutting data outside 0.5-1.5x the reference model, machine-limit violations, safety (tapping without a
rigid tapping cycle, reaming/boring with rapid retract, rapids into the stock), G-code parse warnings, plus agent
time and tokens for harness sessions.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import numpy as np

from .config import SETTINGS
from .knowledge import get_store
from .services import FreeCADRunner
from .services.voxel import Frame

TOOL_TYPES = {"endmill": "endmill", "flat_endmill": "endmill", "ball_endmill": "ball_endmill", "bull_endmill": "bull_endmill", "corner_radius_endmill": "bull_endmill",
              "face_mill": "face_mill", "drill": "twist_drill", "twist_drill": "twist_drill", "spot_drill": "spot_drill", "center_drill": "spot_drill",
              "chamfer_mill": "chamfer_mill", "countersink": "countersink", "reamer": "reamer", "boring_bar": "boring_bar", "tap": "tap", "thread_mill": "thread_mill"}


# ============================================================================================ export
def export_task(cad: str | Path, out_dir: str | Path, material: str | None = None, notes: str = "", stock: str | Path | None = None,
                tol: float | None = None, include_tools: bool = True, fc: FreeCADRunner | None = None) -> Path:
    fc = fc or FreeCADRunner()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cad = Path(cad).resolve()
    shutil.copy(cad, out / cad.name)
    part = fc.run("tessellate.py", {"path": str(cad), "deflection": 0.2, "raw": True}, out / "fc")
    tol = tol or SETTINGS.tol_mm
    if stock:
        shutil.copy(Path(stock), out / "stock.step")
        stock_desc = {"kind": "user", "file": "stock.step", "note": "user blank, modelled in the part's coordinate system"}
    else:
        m = SETTINGS.stock_margins
        b = part["bbox"]
        lo = [b[0] - m["xy_margin"], b[1] - m["xy_margin"], b[2] - m["z_bottom"]]
        hi = [b[3] + m["xy_margin"], b[4] + m["xy_margin"], b[5] + m["z_top"]]
        fc.run("make_box.py", {"lo": lo, "hi": hi, "out": str((out / "stock.step").resolve())}, out / "fc")
        stock_desc = {"kind": "box", "file": "stock.step", "lo": [round(float(v), 4) for v in lo], "hi": [round(float(v), 4) for v in hi], "margins": m}
    store = get_store()
    task = {"part_file": cad.name, "part_bbox": [round(float(v), 4) for v in part["bbox"]], "stock": stock_desc, "material": material, "notes": notes,
            "tolerance_mm": tol, "machine": store.machine, "frame_convention": "setup frame = part rotated so that the machining side is +Z, part bbox minimum at the origin",
            "rotations": {"+Z": "none", "-Z": "180 deg about X", "+X": "-90 deg about Y", "-X": "+90 deg about Y", "+Y": "+90 deg about X", "-Y": "-90 deg about X"},
            "process_vocabulary": sorted(store.templates["process_catalog"]), "tool_types": sorted(set(TOOL_TYPES))}
    (out / "task.json").write_text(json.dumps(task, indent=1, ensure_ascii=False), encoding="utf-8")
    if include_tools:
        tools = [{k: t.get(k) for k in ("id", "category", "name", "diameter_mm", "diameter_range_mm", "flutes", "tool_material", "flute_length_mm",
                                        "overall_length_mm", "iso", "thread_standards", "thread_major_range_mm", "pitch_range_mm") if t.get(k) is not None}
                 for t in store.tool_library["tools"]]
        (out / "tool_library.json").write_text(json.dumps(tools, indent=1), encoding="utf-8")
    (out / "prompt.md").write_text(_prompt(task, include_tools), encoding="utf-8")
    return out


def _prompt(task: dict, include_tools: bool) -> str:
    st = task["stock"]
    stock_txt = (f"an axis-aligned box from {st['lo']} to {st['hi']} mm in the part's coordinate system (part bounding box + "
                 f"{st['margins']['xy_margin']} mm per side, + {st['margins']['z_top']} mm on top, + {st['margins']['z_bottom']} mm below); also given as stock.step"
                 if st["kind"] == "box" else "the blank in stock.step, modelled in the part's coordinate system")
    return f"""# CNC milling task

You are a CNC process planner and CAM programmer. Plan the machining of the part in `{task['part_file']}` (STEP, mm) and write
the G-code for a 3-axis vertical machining centre. Several setups (re-clamping with another side up) are allowed.

## Inputs
- Part: `{task['part_file']}`, bounding box {task['part_bbox']} (mm, part coordinates).
- Stock: {stock_txt}.
- Material: {task['material'] or 'see notes'}.
- Drawing notes: {task['notes'] or 'none'}.
- Machine: {json.dumps(task['machine'])}.
- Verification tolerance: +-{task['tolerance_mm']} mm on every target surface.
{"- Tool library: `tool_library.json` (use these tools)." if include_tools else "- Tools: choose standard tools; describe each in the answer."}

## Conventions
- Units mm (G21), absolute (G90), XY plane (G17). Any ISO/Fanuc/LinuxCNC G-code (G0/G1/G2/G3, canned cycles G73/G81-G89, G84 rigid tapping, G33.1).
- Setup frame: {task['frame_convention']}. Rotations per side: {json.dumps(task['rotations'])}.
  The rotation is applied to the part about the origin, then the part is moved so that its bounding-box minimum is X0 Y0 Z0.
- One G-code file per setup. Start every operation with a comment `(begin operation: <step> <process>)` and load tools with `T<n> M6`.

## Answer
Write `answer.json` and the G-code files next to it:
```json
{{"material": "<material>", "setups": [{{"side": "+Z", "gcode_file": "setup1.ngc",
  "tools": [{{"t": 1, "type": "<one of {task['tool_types']}>", "diameter_mm": 10, "flutes": 3, "tip_angle_deg": 118, "corner_radius_mm": 0, "tool_material": "carbide"}}],
  "operations": [{{"step": 1, "process": "<one of {task['process_vocabulary']}>", "tool": 1, "features": "<what it machines>",
                   "spindle_rpm": 0, "feed_mm_min": 0, "doc_mm": 0, "woc_mm": 0}}]}}]}}
```
"""


# ============================================================================================ evaluate
def _rotated_min(part_raw: dict, side: str) -> list[float]:
    fr = Frame(side, [0.0, 0.0, 0.0])
    return (-fr.to_setup(np.asarray(part_raw["vertices"], float)).min(0)).tolist()


def _setup_translation(part_raw: dict, setup: dict) -> list[float]:
    """Explicit frames register existing G-code; absent frames keep the task convention."""
    if os.environ.get("CAM_EVAL_EXPLICIT_FRAME", "0") == "1" and setup.get("translation") is not None:
        translation = np.asarray(setup["translation"], dtype=float)
        if translation.shape != (3,) or not np.isfinite(translation).all():
            raise ValueError("setup translation must contain three finite coordinates")
        return translation.tolist()
    return _rotated_min(part_raw, setup["side"])


def answer_from_session(session_dir: Path) -> dict:
    """The harness's own output in the answer format (so both are scored by the same code)."""
    s = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    entries = [e for e in (s.get("setups") or []) if e.get("cam", {}).get("gcode_file")] or \
              [{"index": 1, "up": (s.get("orientation") or {}).get("up", "+Z"), "cam": s.get("cam") or {}, "ts": s.get("ts")}]
    setups = []
    for e in entries:
        req = (e.get("cam") or {}).get("request") or {}
        tools = [{"t": t["tool_number"], "type": t.get("category"), "diameter_mm": t.get("diameter_mm"), "flutes": t.get("flutes"),
                  "tip_angle_deg": (t.get("shape_params") or {}).get("TipAngle") or (t.get("shape_params") or {}).get("CuttingEdgeAngle"),
                  "corner_radius_mm": t.get("corner_radius_mm"), "tool_material": "HSS" if "HSS" in str(t.get("name", "")) else "carbide",
                  "shape_params": t.get("shape_params")} for t in req.get("tools", [])]
        ops = [{"step": o["step"], "process": o["process"], "tool": o["tool_number"], "features": ",".join(o.get("feature_ids", [])), "label": o.get("label"),
                "spindle_rpm": o.get("spindle_rpm"), "feed_mm_min": o.get("feed_mm_min"), "doc_mm": o.get("doc_mm"), "woc_mm": o.get("woc_mm")}
               for o in req.get("operations", [])]
        setups.append({"side": e.get("up", "+Z"), "gcode_file": (e.get("cam") or {}).get("gcode_file"), "tools": tools, "operations": ops})
        if e.get("translation") is not None:
            setups[-1]["translation"] = e["translation"]
    return {"material": (s.get("fused") or {}).get("material", {}).get("id"), "setups": setups, "usage": s.get("usage"), "seconds": s.get("seconds"),
            "timings": s.get("timings"), "source": "harness"}


def evaluate(task_dir: str | Path, answer_dir: str | Path | None = None, session_dir: str | Path | None = None, out: str | Path | None = None,
             fc: FreeCADRunner | None = None, samples: int | None = None) -> dict:
    from .normalizer import Normalizer
    from .planning import sequencing as SEQ
    from .planning.cutting_data import cutting_data
    from .services import gcode as G
    from .services.multi_verify import MultiSetupVerifier
    fc = fc or FreeCADRunner()
    store = get_store()
    task_dir = Path(task_dir)
    task = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    tol = float(task["tolerance_mm"])
    if session_dir:
        base = Path(session_dir)
        ans = answer_from_session(base)
    else:
        base = Path(answer_dir)
        ans = json.loads((base / "answer.json").read_text(encoding="utf-8"))
        ans["source"] = "answer"
    part_raw = fc.run("tessellate.py", {"path": str((task_dir / task["part_file"]).resolve()), "deflection": max(0.01, tol / 4), "raw": True}, task_dir / "fc")
    stock_raw = fc.run("tessellate.py", {"path": str((task_dir / task["stock"]["file"]).resolve()), "deflection": 0.05, "raw": True}, task_dir / "fc")
    mid = Normalizer(store.material_aliases(), None, "material").normalize(ans.get("material") or task.get("material"))[0]
    mat = store.material(mid)
    proc_norm = Normalizer({k: [k.replace("_", " ")] for k in store.templates["process_catalog"]}, None, "process")
    setups = [{"index": i + 1, "up": s["side"], "translation": _setup_translation(part_raw, s)} for i, s in enumerate(ans["setups"])]
    metrics: dict = {"source": ans.get("source"), "material": mid, "tolerance_mm": tol, "setups": [], "issues": []}
    if not setups:
        metrics["issues"].append("answer has no setups")
        return _write(metrics, out or base / "metrics.json")
    ver = MultiSetupVerifier(part_raw, setups, stock_raw, tol, SETTINGS.sim_max_cells, None, samples or SETTINGS.surface_samples,
                             part_path=task_dir / task["part_file"], freecad=fc, workdir=task_dir / "fc")
    m = store.machine
    total_time, total_changes, avoidable = 0.0, 0, 0
    cd_checked = cd_out = 0
    for sp, s in zip(setups, ans["setups"]):
        k = sp["index"]
        entry = {"index": k, "side": s["side"], "issues": []}
        tools = []
        for t in s.get("tools", []):
            cat = TOOL_TYPES.get(str(t.get("type", "")).lower(), "endmill")
            spar = dict(t.get("shape_params") or {})
            spar.setdefault("Diameter", t.get("diameter_mm"))
            if cat in ("twist_drill", "spot_drill"):
                spar.setdefault("TipAngle", t.get("tip_angle_deg") or (90 if cat == "spot_drill" else 118))
            if cat in ("chamfer_mill", "countersink"):
                spar.setdefault("CuttingEdgeAngle", t.get("tip_angle_deg") or 90)
            tools.append({"tool_number": int(t["t"]), "category": cat, "diameter_mm": float(t.get("diameter_mm") or 0), "flutes": t.get("flutes") or 2,
                          "shape_params": spar, "corner_radius_mm": t.get("corner_radius_mm") or 0.0, "tool_material": t.get("tool_material") or "carbide"})
        tool_by = {t["tool_number"]: t for t in tools}
        ops = s.get("operations", [])
        names = []
        for o in ops:
            p = proc_norm.normalize(str(o.get("process", "")))[0]
            names.append(p or str(o.get("process")))
            o["_process"] = p
        # plan-level checks: standard-practice phase order, tool changes, cutting data vs the reference model, machine limits
        seq = [{"step": i + 1, "process": n, "feature_ids": ["*"]} for i, n in enumerate(names)]
        v = SEQ.validate(seq, [], (mat or {}).get("group", "*"), None, "standard",
                         [{"step": i + 1, "tool_id": str(o.get("tool"))} for i, o in enumerate(ops)])
        inversions = sum(1 for i in range(len(names)) for j in range(i + 1, len(names)) if SEQ.phase(names[j]) < SEQ.phase(names[i]) and names[i] not in SEQ.FINAL)
        entry.update(processes=names, unknown_processes=[n for n, o in zip(names, ops) if not o["_process"]], ordering_inversions=inversions,
                     validation_critical=[c["message"] for c in v["critical"]], tool_changes=v["stats"].get("tool_changes"),
                     avoidable_tool_changes=v["stats"].get("avoidable_tool_changes"))
        total_changes += v["stats"].get("tool_changes") or 0
        avoidable += v["stats"].get("avoidable_tool_changes") or 0
        cd = []
        for o in ops:
            t = tool_by.get(int(o.get("tool") or -1))
            if not t or not o.get("spindle_rpm") or not mat or not o["_process"]:
                continue
            ref = cutting_data(t, o["_process"], mat, {"diameter": t["diameter_mm"]})
            if ref.get("error") or not ref.get("spindle_rpm"):
                continue
            cd_checked += 1
            r_n = float(o["spindle_rpm"]) / ref["spindle_rpm"]
            r_f = float(o.get("feed_mm_min") or 0) / max(ref["feed_mm_min"], 1)
            bad = not (0.5 <= r_n <= 1.5) or (o.get("feed_mm_min") and not (0.5 <= r_f <= 1.5))
            cd_out += bool(bad)
            cd.append({"step": o.get("step"), "rpm": o["spindle_rpm"], "ref_rpm": ref["spindle_rpm"], "feed": o.get("feed_mm_min"), "ref_feed": ref["feed_mm_min"], "out_of_range": bool(bad)})
            if float(o["spindle_rpm"]) > m["max_rpm"] + 1 or float(o.get("feed_mm_min") or 0) > m["max_feed_mm_min"] + 1:
                entry["issues"].append(f"step {o.get('step')}: above the machine limits ({o['spindle_rpm']} rpm, {o.get('feed_mm_min')} mm/min)")
        entry["cutting_data"] = cd
        # G-code: simulation, safety, cycle time
        gfile = s.get("gcode_file")
        gpath = Path(gfile) if gfile and Path(gfile).is_absolute() else base / str(gfile)
        if not gfile or not gpath.is_file():
            entry["issues"].append("no G-code for this setup")
            metrics["setups"].append(entry)
            continue
        parsed = G.parse(str(gpath))
        by_tool_proc = {}
        for o in ops:
            by_tool_proc.setdefault(int(o.get("tool") or -1), o.get("_process") or "")
        # the answer operation of a program operation: by its label (harness sessions), else by the step number its
        # "(begin operation: <step> ...)" comment starts with; its tool when the program loads none (no T.. M6: FreeCAD
        # posts without tool changes) or one the answer does not describe
        op_by_label = {str(o["label"]): o for o in ops if o.get("label")}
        op_by_step = {}
        for o in ops:
            try:
                op_by_step.setdefault(int(o.get("step")), o)
            except (TypeError, ValueError):
                pass

        def answer_op(po):
            o = op_by_label.get(po["name"])
            mstep = re.match(r"\s*(\d+)", po["name"])
            return o or (op_by_step.get(int(mstep.group(1))) if mstep else None)

        def tool_of(po):
            if po.get("tool") in tool_by:
                return po["tool"]
            o = answer_op(po)
            try:
                return int(o["tool"]) if o is not None and o.get("tool") is not None else po.get("tool")
            except (TypeError, ValueError):
                return po.get("tool")

        req_ops = []
        for i, po in enumerate(parsed["ops"]):
            t, o = tool_of(po), answer_op(po)
            req_ops.append({"step": i + 1, "label": po["name"], "process": (o or {}).get("_process") or by_tool_proc.get(t, ""),
                            "tool_number": t, "feature_ids": []})
        for po, ro in zip(parsed["ops"], req_ops):
            cat = (tool_by.get(ro["tool_number"]) or {}).get("category")
            cyc = {c["cycle"] for c in po.get("cycles", [])}
            if cat == "tap" and cyc and not cyc <= {"G84", "G74", "G33.1"}:
                entry["issues"].append(f"safety: {po['name']} taps with {sorted(cyc)} instead of a rigid tapping cycle")
            if cat in ("reamer", "boring_bar") and cyc & {"G81", "G83", "G73"}:
                entry["issues"].append(f"safety: {po['name']} retracts at rapid from a reamed/bored hole ({sorted(cyc)})")
            if ro["tool_number"] not in tool_by:
                entry["issues"].append(f"{po['name']}: tool T{ro['tool_number']} not described in the answer (simulated as nothing)")
        ct = G.cycle_time_min(parsed, m["rapid_mm_min"], m["tool_change_s"])
        total_time += ct["total_min"]
        sim = ver.check_setup(k, str(gpath), {"operations": req_ops, "tools": tools}, [])
        top = ver.results[k]["dexel"]
        stock_top = float(np.max(top.h)) if top is not None else 0.0
        rim = G.rapids_in_material(parsed, stock_top)
        entry.update(cycle_time=ct, gcode_warnings=parsed["warnings"], view_stats=sim["stats"], rapids_below_stock_top=rim)
        metrics["setups"].append(entry)
    final = ver.final()
    ver.save_surface(Path(out or base / "metrics.json").with_name("surface_final.npz"))
    metrics.update(final=final, cycle_time_min=round(total_time, 2), tool_changes=total_changes, avoidable_tool_changes=avoidable,
                   cutting_data_checked=cd_checked, cutting_data_out_of_range=cd_out,
                   safety_issues=sum(1 for e in metrics["setups"] for i in e.get("issues", []) if i.startswith("safety")),
                   ordering_inversions=sum(e.get("ordering_inversions", 0) for e in metrics["setups"]),
                   agent={"seconds": ans.get("seconds"), "usage": ans.get("usage"), "timings": ans.get("timings")})
    return _write(metrics, out or base / "metrics.json")


def _write(metrics: dict, path) -> dict:
    path = Path(path)
    path.write_text(json.dumps(metrics, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    f = metrics.get("final") or {}
    lines = [f"# Evaluation ({metrics.get('source')})", "",
             f"- surface within +-{metrics.get('tolerance_mm')} mm: **{f.get('surface_within_tol_pct')} %** (without drill points {f.get('surface_within_tol_pct_excl_expected')} %; "
             f"another {f.get('band_pct')} % out only by less than {f.get('band_factor')}x the tolerance)",
             f"- material left {f.get('excess_cm3')} cm3 on {f.get('excess_area_pct')} % of the surface; overcut {f.get('gouge_cm3')} cm3 on {f.get('gouge_area_pct')} %",
             f"- cycle time {metrics.get('cycle_time_min')} min; tool changes {metrics.get('tool_changes')} (avoidable {metrics.get('avoidable_tool_changes')})",
             f"- ordering inversions vs standard practice: {metrics.get('ordering_inversions')}; cutting data outside 0.5-1.5x reference: "
             f"{metrics.get('cutting_data_out_of_range')}/{metrics.get('cutting_data_checked')}; safety issues: {metrics.get('safety_issues')}",
             f"- agent: {json.dumps(metrics.get('agent'))}"]
    for e in metrics.get("setups", []):
        lines.append(f"- setup {e['index']} {e['side']}: {e.get('processes')} issues {e.get('issues')}")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics
