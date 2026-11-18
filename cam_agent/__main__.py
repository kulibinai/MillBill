"""CLI:
  python -m cam_agent run part.step --material "AL6061-T6" --notes "2x M6 tapped holes ..." [--setups auto] [--stock blank.step] [--tol 0.1]
  python -m cam_agent demo | check
  python -m cam_agent export-task part.step --out task_dir [--stock blank.step] --material ... --notes ...
  python -m cam_agent evaluate --task task_dir (--answer answer_dir | --session outputs/<session>)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .config import ROOT_DIR, SETTINGS


def cmd_check(args) -> int:
    from .llm import LLM
    from .services import FreeCADRunner
    ok = True
    print(f"LLM endpoint: {SETTINGS.base_url} model={SETTINGS.model} mode={SETTINGS.mode}")
    if SETTINGS.mode != "deterministic":
        try:
            llm = LLM(SETTINGS)
            r = llm.chat([{"role": "user", "content": "Reply with the single word OK."}], thinking=False, max_tokens=8)
            print("  chat:", repr(r.content.strip()), r.usage.as_dict())
            tools = [{"type": "function", "function": {"name": "add", "description": "add two numbers", "parameters": {"type": "object",
                      "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}}]
            r = llm.chat([{"role": "user", "content": "Use the add tool to compute 2+3."}], tools=tools, thinking=False)
            print("  tool call:", r.tool_calls)
            ok &= bool(r.tool_calls)
        except Exception as e:  # noqa: BLE001
            print("  LLM FAILED:", e)
            ok = False
    fc = FreeCADRunner()
    print(f"FreeCAD: {fc.exe}")
    if fc.available():
        try:
            demo = ROOT_DIR / "examples" / "demo_bracket.step"
            if not demo.exists():
                demo = demo_part()
            r = fc.run("extract_features.py", {"path": str(demo)})
            print("  features:", r["summary"], "FreeCAD", r["freecad_version"])
        except Exception as e:  # noqa: BLE001
            print("  FreeCAD FAILED:", e)
            ok = False
    else:
        print("  freecadcmd not found (set FREECADCMD)")
        ok = False
    from .services import dexel
    print("numba kernel for the simulation:", "yes" if dexel._jit is not None else "no (pip install numba: about 2x faster simulation, identical results)")
    from .recognition import aagnet
    print("AAGNet recogniser:", "configured" if aagnet.available() else "not configured (rule-based recognition)")
    print("OK" if ok else "PROBLEMS FOUND")
    return 0 if ok else 1


def demo_part() -> Path:
    """Builds the demo bracket into <out>/demo_part (STEP, FCStd, FreeCAD backups); examples/ is only read."""
    import os
    from .services import FreeCADRunner
    fc = FreeCADRunner()
    out = (SETTINGS.out_dir / "demo_part").resolve()
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([fc.exe, str(ROOT_DIR / "examples" / "make_demo_part.py")], check=True, capture_output=True, text=True, cwd=str(out),
                   env={**os.environ, "CAM_DEMO_OUT": str(out)})
    return out / "demo_bracket.step"


def _apply(args) -> None:
    if getattr(args, "mode", None):
        SETTINGS.mode = args.mode
    if getattr(args, "thinking", False):
        SETTINGS.thinking = True
    if getattr(args, "out", None):
        SETTINGS.out_dir = Path(args.out)
    if getattr(args, "tol", None):
        SETTINGS.tol_mm = float(args.tol)
    if getattr(args, "sequencing", None):
        SETTINGS.sequencing = args.sequencing
    if getattr(args, "recognizer", None):
        SETTINGS.recognizer = args.recognizer


def cmd_run(args) -> int:
    from .orchestrator import Orchestrator, interactive_hitl
    _apply(args)
    orch = Orchestrator(SETTINGS, hitl=interactive_hitl if args.interactive else None)
    session = orch.run(args.cad, material=args.material, notes=args.notes or "", part_name=args.name, skip_cam=args.skip_cam, detail=args.detail,
                       up=args.up, setups=args.setups, stock=args.stock)
    print("\nREPORT:", session["report_path"])
    return 0


def cmd_demo(args) -> int:
    _apply(args)  # --out before demo_part: the part goes under it
    args.cad = str(demo_part())
    args.material = args.material or "Aluminum 6061-T6"
    args.notes = args.notes or ("Material AL 6061-T6. 4x through holes 6.6 mm (M6 clearance) at the corners. 2x M6 tapped blind holes 10 deep on the "
                                "top face. Pocket 50x30 depth 8, corner R5. Open slot 8 wide from the +X edge. Chamfer 1x45 on top outer edges. Ra 1.6 on "
                                "top face. ISO 2768-m. Anodize after machining.")
    args.name = args.name or "demo_bracket"
    return cmd_run(args)


def cmd_export(args) -> int:
    from .evaluate import export_task
    _apply(args)
    out = export_task(args.cad, args.out_task, args.material, args.notes or "", args.stock, SETTINGS.tol_mm, not args.no_tools)
    print(f"task written to {out}: give prompt.md (+ the STEP files{'' if args.no_tools else ' and tool_library.json'}) to the model; "
          f"run the harness on the same blank with:  python -m cam_agent run {Path(args.cad).name} --stock {out / 'stock.step'}")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import evaluate
    m = evaluate(args.task, answer_dir=args.answer, session_dir=args.session, out=args.metrics)
    print(json.dumps({k: m.get(k) for k in ("source", "final", "cycle_time_min", "tool_changes", "avoidable_tool_changes", "ordering_inversions",
                                            "cutting_data_out_of_range", "safety_issues")}, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cam_agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="check LLM endpoint, FreeCAD, optional accelerators").set_defaults(fn=cmd_check)
    for name, fn in (("run", cmd_run), ("demo", cmd_demo)):
        r = sub.add_parser(name, help="plan a part" if name == "run" else "build the demo bracket and plan it")
        if name == "run":
            r.add_argument("cad", help="STEP / FCStd file")
        r.add_argument("--material", default=None)
        r.add_argument("--notes", default=None, help="drawing notes / annotations as free text")
        r.add_argument("--name", default=None)
        r.add_argument("--mode", choices=["sequential", "parallel", "deterministic"], default=None, help="agent mode (paper Sec. 3.4)")
        r.add_argument("--sequencing", choices=["standard", "element"], default=None, help="standard practice (default) or element-by-element")
        r.add_argument("--recognizer", choices=["auto", "rules", "aagnet"], default=None)
        r.add_argument("--thinking", action="store_true")
        r.add_argument("--interactive", action="store_true", help="human-in-the-loop prompts")
        r.add_argument("--skip-cam", action="store_true")
        r.add_argument("--up", default="+Z", choices=["auto", "+Z", "-Z", "+X", "-X", "+Y", "-Y"], help="machining side (auto = fewest undercuts)")
        r.add_argument("--setups", default="1", help='"1" (default), "auto" (up to 6) or a maximum like "3"')
        r.add_argument("--stock", default=None, help="STEP of the blank in the part's coordinate system (default: bounding box + margins, CAM_STOCK_*)")
        r.add_argument("--tol", default=None, help="verification tolerance in mm (default 0.1; simulation cell = tol / 2)")
        r.add_argument("--detail", choices=["executive", "standard", "detailed"], default="standard")
        r.add_argument("--out", default=None)
        r.set_defaults(fn=fn)
    e = sub.add_parser("export-task", help="task package for a planner without the harness (same stock, same conventions)")
    e.add_argument("cad")
    e.add_argument("--out", dest="out_task", required=True)
    e.add_argument("--stock", default=None)
    e.add_argument("--material", default=None)
    e.add_argument("--notes", default=None)
    e.add_argument("--tol", default=None)
    e.add_argument("--no-tools", action="store_true", help="do not hand the tool library to the model")
    e.set_defaults(fn=cmd_export)
    v = sub.add_parser("evaluate", help="score an answer (or a harness session) on a task with the harness metrics")
    v.add_argument("--task", required=True)
    g = v.add_mutually_exclusive_group(required=True)
    g.add_argument("--answer", default=None, help="directory with answer.json + G-code")
    g.add_argument("--session", default=None, help="harness session directory (outputs/<session>)")
    v.add_argument("--metrics", default=None, help="output JSON (default: <answer or session>/metrics.json)")
    v.set_defaults(fn=cmd_evaluate)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
