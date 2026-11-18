"""Tool Selection (TS) Agent.

The four-stage draft is a hint. Outside deterministic mode the LLM assigns the tool and the cutting data for every
tooling step. A library id is accepted as written. Spindle and feed are capped only by the machine limits. There is
no geometry veto and no band around the formula values.
"""
from __future__ import annotations

import json
import os
from decimal import Decimal, ROUND_FLOOR

from ..knowledge import get_store
from ..llm import LLM
from ..planning.cutting_data import cutting_data, strategy_for
from ..react import AgentResult, SequentialReAct
from ..tools import Tool, ToolRegistry
from .feature_extraction import describe_features

NO_REACH = os.environ.get("CAM_NO_REACH", "1") == "1"  # a cutter goes as deep as the opening lets it: no stick-out filter

TS_SYSTEM = """SYSTEM PROMPT: Tool Selection Agent
ROLE: Choose the tool and the cutting data for every tooling step. Your assignment is what gets machined.
A draft is attached as a hint (a library tool and formula speeds). Replace any of it. tool_id must be an id from the
library; call tool_selection to see the tools that exist for a step. Set spindle_rpm, feed_mm_min, doc_mm and woc_mm
yourself. Return one assignment per tooling step, not only the steps you disagree with. Do not call a constraint
checker. Aim to finish within 3 turns.
OUTPUT SCHEMA (JSON object):
{"assignments": [{"step": <int>, "tool_id": <library id>, "tool_diameter_mm": <number>, "spindle_rpm": <number>, "feed_mm_min": <number>,
  "doc_mm": <number>, "woc_mm": <number>, "justification": <short>}],
 "special_tooling": [<str>], "notes": <str>}"""


class ToolSelectionAgent:
    name = "tool_selection"

    def __init__(self, llm: LLM, mode: str = "parallel", max_iterations: int = 8) -> None:
        self.llm, self.mode, self.max_iterations = llm, mode, max_iterations
        self.store = get_store()
        self.lib = self.store.tool_library
        self.by_id = {t["id"]: t for t in self.lib["tools"]}

    # ------------------------------------------------------------------ public
    def run(self, fused: dict, ps_output: dict, kr_output: dict | None = None) -> AgentResult:
        feats = {f["id"]: f for f in fused["features"]}
        mat = self.store.material(fused["material"]["id"])
        draft = [self._select(step, feats, mat, fused) for step in ps_output["sequence"]]
        tool_calls = [{"name": "tool_selection", "arguments": {}, "result_preview": ""}, {"name": "parameter_estimation", "arguments": {}, "result_preview": ""}]
        iterations, tokens, notes, special, llm_changes = 1, 0, "", [], []
        assignments = draft
        open_steps = [a["step"] for a in draft if a["requires_tool"]]
        if self.mode != "deterministic":
            res = self._react(fused, ps_output, draft, mat, kr_output or {}, open_steps)
            tool_calls += res.tool_calls
            iterations, tokens = res.iterations, res.tokens
            if res.success and isinstance(res.output.get("assignments"), list):
                assignments, llm_changes = self._merge(draft, res.output["assignments"], feats, mat, fused)
                notes, special = res.output.get("notes", ""), res.output.get("special_tooling", [])
                rejected = [c for c in llm_changes if c.get("change") == "tool" and c.get("accepted") is False]
                if rejected:
                    revised = self._revise_tools(fused, draft, assignments, rejected)
                    tokens += self.llm.last_usage.total
                    if revised:
                        assignments, more = self._merge(draft, revised, feats, mat, fused, keep_unfit=True)
                        llm_changes = llm_changes + more
                        notes = (notes + " Fit observations were shown once; the revised tools are kept.").strip()
        n_required = sum(1 for a in assignments if a["requires_tool"])
        n_tooled = sum(1 for a in assignments if a["requires_tool"] and a.get("tool_id"))
        confidence = round(0.3 + 0.7 * (n_tooled / n_required), 3) if n_required else 1.0  # Eq. 7
        out = {"assignments": assignments, "tooling_list": tooling_list(assignments), "special_tooling": special, "notes": notes,
               "confidence": confidence, "n_required": n_required, "n_tooled": n_tooled, "llm_changes": llm_changes,
               "material_data": {k: (mat or {}).get(k) for k in ("id", "iso_group", "kc11_n_mm2", "mc", "coolant")}}
        self.llm.trace.log("agent", agent=self.name, tooled=n_tooled, required=n_required, confidence=confidence, llm_changes=len(llm_changes))
        return AgentResult(self.name, out, tool_calls=tool_calls, iterations=iterations, tokens=tokens)

    # ------------------------------------------------------ 4-stage selection
    def _select(self, step: dict, feats: dict, mat: dict | None, fused: dict) -> dict:
        p = step["process"]
        a = {"step": step["step"], "process": p, "feature_ids": step.get("feature_ids", []), "requires_tool": False}
        if p in self.lib["non_tooling_processes"] or not self.store.templates["process_catalog"].get(p, {}).get("tooling", False):
            a["justification"] = "no cutting tool required"
            return a
        a["requires_tool"] = True
        fs = [feats[i] for i in (step.get("feature_ids") or []) if isinstance(i, str) and i in feats]
        geom = self.geometry(p, fs, fused)
        a["geometry"] = geom
        iso = (mat or {}).get("iso_group", "P")
        cands = self.candidates(p, iso, geom)
        if not cands and geom.get("reach"):
            loose = self.candidates(p, iso, {**geom, "reach": None})
            if loose:
                best_reach = max(_reach(t) for t in loose)
                cands = [t for t in loose if _reach(t) >= best_reach - 1e-6]
                a["reach_limit_mm"] = round(best_reach, 1)
                a["warning"] = f"no library tool reaches {geom['reach']:.0f} mm: depth limited to {best_reach:.0f} mm (longer tool or another setup)"
        if not cands and geom.get("max_d"):
            loose = self.candidates(p, iso, {**geom, "max_d": None})
            if loose:
                dmin = min(float(t["diameter_mm"]) for t in loose)
                cands = [t for t in loose if float(t["diameter_mm"]) == dmin]
                a["warning"] = f"no {iso}-rated tool <= D{geom['max_d']:g} in the library: smallest D{dmin:g} used, material stays in corners/gaps"
        a["candidates"] = [c["id"] for c in cands[:8]]
        if not cands:
            a["justification"] = f"no library tool for {p} fits {_brief(geom)} in ISO group {iso}"
            return a
        best = self.primary(p, cands, geom)
        return self._assign(a, best, p, mat, geom, why=best.get("_why", "best fit"))

    def _assign(self, a: dict, tool: dict, p: str, mat: dict | None, geom: dict, why: str) -> dict:
        t = instance(tool, geom)
        cd = cutting_data(t, p, mat, geom, strategy=strategy_for(p, geom.get("op_type"), geom, float(t.get("diameter_mm") or 0)))
        if cd.get("error"):
            a["justification"] = cd["error"]
            a.pop("tool_id", None)
            return a
        a.update({"tool_id": tool["id"], "tool_name": tool["name"], "tool_category": tool["category"], "justification": why}, **cd)
        crit = [n for n in cd.get("notes", []) if "exceeds" in n]
        if crit:
            a["cutting_notes_critical"] = crit
        return a

    def geometry(self, process: str, fs: list[dict], fused: dict | None = None) -> dict:
        """What the tool of this step must fit: required diameter, max diameter, reach, thread, tolerance."""
        g: dict = {}
        holes = [f for f in fs if f["type"] == "hole"]
        pockets = [f for f in fs if f["type"] in ("pocket", "slot", "step")]
        if holes:
            h = holes[0]
            d = float(h["diameter"])
            g.update(diameter=d, depth=max(float(x["depth"]) for x in holes), through=all(x["through"] for x in holes), count=len(holes))
            spec = self.store.thread(h.get("thread"))
            if spec:
                g.update(thread=spec["thread"], pitch=spec["pitch"], series=spec["series"], major=spec["major_mm"], minor=spec["minor_mm"])
            if h.get("tolerance_grade"):
                g["tolerance_it"] = int("".join(c for c in h["tolerance_grade"] if c.isdigit()) or 7)
            if process == "drilling":
                if spec:  # the thread may be modelled at the tap-drill or at the nominal size: the drill is the tap drill (ISO 2306)
                    g["diameter"] = spec["tap_drill_mm"]
                if h.get("tolerance_grade"):
                    g["diameter"] = round(d - self.store.reaming_stock(d), 2)
                if h.get("subtype") == "large_precision_hole":
                    g["diameter"] = round(d - self.store.boring_allowance, 2)
                if h.get("through_diameter"):  # counterbore / countersink: the drill makes the through hole
                    g["diameter"] = float(h["through_diameter"])
                g.pop("tolerance_it", None)
            elif process in ("spot_drilling",):
                g = {"depth": 0.45 * min(float(x["diameter"]) for x in holes), "hole_diameter": min(float(x["diameter"]) for x in holes)}
            elif process == "counterboring":
                g.update(max_d=0.8 * float(h.get("counterbore_diameter") or d), reach=float(h.get("counterbore_depth") or 5) + 2)
            elif process == "countersinking":
                g.update(min_d=float(h.get("countersink_diameter") or d + 2))
            elif process == "thread_milling" and spec:
                g["max_d"] = 0.75 * spec["minor_mm"]
        if pockets:
            depth = max(float(p.get("depth") or 0) for p in pockets)
            width = min(float(p.get("min_width") or p["size"][0]) for p in pockets)
            r = min(float(p.get("corner_radius") or 0) for p in pockets)
            kinds = {p["type"] for p in pockets}
            closed = all(p["type"] == "pocket" and not p.get("open") for p in pockets)
            g.update(depth=depth, width=width, corner_radius=r)
            if process.endswith("finishing"):
                # finishing: tool radius below the corner radius (0.8x) so the cutter does not wrap the corner
                # room in the width for the entry and the wall pass: at most 0.85 x width (brep2cam@savrasov)
                g["max_d"] = round(min(0.85 * width, 1.6 * r if r > 0 else 0.5 * width), 3)
                g["flute"] = depth
            elif kinds == {"step"}:
                g["max_d"] = round(min(0.9 * width, 20.0), 3)
                g["reach"] = depth
            elif "slot" in kinds and not closed:
                g["max_d"] = round(0.8 * width, 3)
                g["reach"] = depth
            else:
                g["max_d"] = round(0.7 * width if closed else 0.8 * width, 3)  # room for the helical/ramp entry of adaptive clearing
                g["reach"] = depth
                g["op_type"] = "Adaptive" if closed else None
            if r > 0 and r < 0.1 * width:
                g["tight_corners"] = True
        cav = [f for f in fs if f["type"] == "cavity"]
        if cav:
            c = cav[0]
            g.update(depth=float(c["depth"]), reach=float(c["depth"]), width=float(min(c["size"])),
                     max_d=float(c["finish_tool_d"] if process == "surface_finishing" else c["rough_tool_d"]))
        if process == "facing":
            faces = [f for f in fs if f["type"] == "planar_face"]
            if faces:
                g["width"] = min(faces[0]["size"])
            if fused and fused.get("stock_allowance"):
                g["top_allowance"] = fused["stock_allowance"].get("top")
        if process in ("profile_milling", "profile_finishing"):
            op = [f for f in fs if f["type"] in ("outer_profile", "boss")]
            if op:
                g["reach"] = float(op[0]["size"][2]) if len(op[0].get("size", [])) > 2 else float(op[0].get("height") or 10)
                cap = 20.0
                # largest cutter that walks this outline: a wider one cuts a concave notch (measurement, not a ban)
                if os.environ.get("CAM_PROFILE_MAX_D", "1") == "1":
                    limits = [float(f["profile_max_d"]) for f in op if f.get("profile_max_d")]
                    if limits:
                        cap = min(cap, min(limits))
                g["max_d"] = round(cap, 3)
                if process == "profile_finishing":
                    g["flute"] = g.pop("reach")
        if process in ("chamfering", "deburring"):
            ch = [f for f in fs if f["type"] == "chamfer"]
            g["size"] = max(float(f["size"]) for f in ch) if ch else 0.3
            if ch:
                g["angle_deg"] = float(ch[0].get("angle_deg") or 45.0)
        if process == "fillet_milling":
            fl = [f for f in fs if f["type"] == "fillet"]
            if fl:
                g["max_d"] = 2 * min(float(f["radius"]) for f in fl)
        return g

    def candidates(self, process: str, iso: str, geom: dict) -> list[dict]:
        cats = list(self.lib["process_to_tool_category"].get(process, []))
        if process == "chamfering" and abs(float(geom.get("angle_deg") or 45.0) - 45.0) > 3.0:
            cats = ["ball_endmill"]
        out = []
        for t in self.lib["tools"]:
            if t["category"] not in cats or iso not in t["iso"]:
                continue
            c = t["category"]
            if c in ("twist_drill", "reamer", "boring_bar"):
                rng, dd = t.get("diameter_range_mm"), geom.get("diameter")
                if dd is None or not rng or not (rng[0] <= dd <= rng[1]):
                    continue
                if c == "twist_drill" and geom.get("depth") and geom["depth"] > float(t.get("max_depth_x_d", 5)) * dd + 1e-6:
                    continue
            elif c == "tap":
                if not geom.get("thread") or geom.get("series") not in t.get("thread_standards", []):
                    continue
                lo, hi = t["thread_major_range_mm"]
                if not lo <= float(geom.get("major") or 0) <= hi:
                    continue
                ht = t.get("hole_type", "any")
                if ht != "any" and geom.get("through") is not None and (ht == "through") != bool(geom["through"]):
                    continue
            elif c == "thread_mill":
                if not geom.get("thread"):
                    continue
                plo, phi = t.get("pitch_range_mm", [0, 9])
                if not plo <= float(geom.get("pitch") or 0) <= phi or float(t["diameter_mm"]) > float(geom.get("max_d") or 0) + 1e-6:
                    continue
            elif c == "spot_drill":
                if geom.get("hole_diameter") and float(t["diameter_mm"]) < 0.9 * float(geom["hole_diameter"]):
                    continue  # the spot cone must reach the drill's point before the spot drill's shoulder
            elif c in ("countersink",):
                if float(t["diameter_mm"]) < float(geom.get("min_d") or 0):
                    continue
            elif c == "chamfer_mill":
                if float(t.get("flute_length_mm") or 0) < float(geom.get("size") or 0.3) + 0.2:
                    continue
            elif c in ("endmill", "ball_endmill", "bull_endmill"):
                d = float(t["diameter_mm"])
                if geom.get("max_d") and d > float(geom["max_d"]) + 1e-6:
                    continue
                if not NO_REACH and geom.get("reach") and _reach(t) < float(geom["reach"]) - 1e-6:
                    continue
                if not NO_REACH and geom.get("flute") and float(t.get("flute_length_mm") or 0) < float(geom["flute"]) - 1e-6:
                    if _reach(t) < float(geom["flute"]):
                        continue
            elif c == "face_mill":
                if geom.get("width") and float(t["diameter_mm"]) > 2.5 * float(geom["width"]) + 10:
                    continue
            if geom.get("tolerance_it") and t.get("tolerance_it", 12) > geom["tolerance_it"]:
                continue
            out.append(t)
        return out

    def primary(self, process: str, cands: list[dict], geom: dict) -> dict:
        def score(t):
            c, s = t["category"], 0.0
            if c in ("endmill", "bull_endmill", "ball_endmill", "face_mill"):
                s += float(t["diameter_mm"])  # largest that fits: rigid, highest removal rate, best finish on walls
                need = float(geom.get("reach") or geom.get("flute") or 0)
                s -= 0.05 * max(0.0, _reach(t) - need)  # shortest sufficient reach
                if c == "face_mill" and geom.get("width"):
                    s -= abs(float(t["diameter_mm"]) - 1.4 * float(geom["width"])) * 0.2  # D about 1.3-1.5x the width
            elif c == "spot_drill":
                s -= float(t["diameter_mm"])  # smallest sufficient spot drill
            elif c == "thread_mill":
                s += float(t["diameter_mm"])
            elif c == "countersink":
                s -= float(t["diameter_mm"])
            elif c == "chamfer_mill":
                s -= float(t["diameter_mm"])
            elif c == "twist_drill":
                s -= float(t.get("max_depth_x_d", 5)) * 0.1  # shortest drill that reaches
                s += 0.5 if t["tool_material"] == "carbide" else 0
            if t["tool_material"] == "carbide":
                s += 0.3
            return s
        best = dict(max(cands, key=score))
        if best["category"] in ("endmill", "face_mill"):
            best["_why"] = f"largest {best['category']} that fits ({_brief(geom)}), shortest sufficient reach"
            if geom.get("tight_corners") and not process.endswith("finishing"):
                best["_why"] += "; corners left for the finishing tool"
        else:
            best["_why"] = f"matches {_brief(geom)}"
        return best

    # ---------------------------------------------------------------- ReAct
    def _react(self, fused: dict, ps_output: dict, draft: list[dict], mat: dict | None, kr: dict, open_steps: list[int]) -> AgentResult:
        feats = {f["id"]: f for f in fused["features"]}
        steps = {s["step"]: s for s in ps_output["sequence"]}
        iso = (mat or {}).get("iso_group", "P")

        def one_step(step: int) -> dict:
            st = steps.get(step)
            if not st:
                return {"error": f"no step {step}"}
            fs = [feats[i] for i in st.get("feature_ids", []) if i in feats]
            geom = self.geometry(st["process"], fs, fused)
            cands = self.candidates(st["process"], iso, geom)
            keys = ("id", "name", "category", "diameter_mm", "diameter_range_mm", "flutes", "tool_material", "coating", "flute_length_mm", "overall_length_mm", "tolerance_it")
            return {"step": step, "process": st["process"], "geometry": geom,
                    "note": "candidates are the library tools that enter this opening; any other library id can still be named and is reported if it does not fit",
                    "candidates": [{k: c.get(k) for k in keys if k in c} for c in cands[:8]]}

        def tool_selection(step: int | None = None, steps: list | None = None) -> dict:
            if steps:
                return {"results": [one_step(int(x)) for x in steps]}
            return one_step(int(step)) if step is not None else {"error": "pass step or steps"}

        def parameter_estimation(tool_id: str, step: int | None = None, process: str | None = None, diameter_mm: float | None = None) -> dict:
            t = self.by_id.get(tool_id)
            if not t:
                return {"error": f"unknown tool {tool_id}"}
            st = steps.get(step) if step is not None else None
            p = (st or {}).get("process") or process or "pocket_milling"
            geom = self.geometry(p, [feats[i] for i in (st or {}).get("feature_ids", []) if i in feats], fused)
            if diameter_mm:
                geom["diameter"] = diameter_mm
            return {"tool_id": tool_id, **cutting_data(instance(t, geom), p, mat, geom), "model": "Vc table (ISO group), fz(D) + chip thinning, ap/ae by strategy, Kienzle power"}

        registry = ToolRegistry([
            Tool("tool_selection", "Library tools that fit sequence steps (process, ISO group, geometry, reach, thread, tolerance). Batch with steps=[...].", tool_selection,
                 {"step": {"type": "integer"}, "steps": {"type": "array", "items": {"type": "integer"}}}, [], "tool_library"),
            Tool("parameter_estimation", "Cutting data (rpm, feed, ap, ae, power, torque) for a tool on a step.", parameter_estimation,
                 {"tool_id": {"type": "string"}, "step": {"type": "integer"}, "process": {"type": "string"}, "diameter_mm": {"type": "number"}}, ["tool_id"], "tool_library"),
        ])
        keep = ("step", "process", "requires_tool", "tool_id", "tool_diameter_mm", "candidates", "spindle_rpm", "feed_mm_min", "doc_mm", "woc_mm", "warning", "notes", "justification")
        fit = []
        for a in draft:
            if not a.get("requires_tool"):
                continue
            fit.append(f"step {a['step']} {a['process']}: {_brief(a.get('geometry') or {})}; draft {a.get('tool_id')} D{a.get('tool_diameter_mm')}")
        user = (f"MATERIAL: {(mat or {}).get('name', fused['material'].get('input'))} (ISO {iso}, group {(mat or {}).get('group')}), coolant {(mat or {}).get('coolant')}\n"
                f"MACHINE: {json.dumps({k: self.store.machine[k] for k in ('max_rpm', 'max_feed_mm_min', 'spindle_power_kw', 'spindle_torque_nm')})}\n"
                f"FEATURES:\n{describe_features(fused['features'])}\n"
                f"FIT (measurement, not a ban): {'; '.join(fit)}\n"
                f"DRAFT ASSIGNMENTS (hint only): {json.dumps([{k: a[k] for k in keep if k in a} for a in draft], ensure_ascii=False)}\n"
                f"TOOLING STEPS: {open_steps}\nReturn an assignment for every tooling step. A library id is kept; if it does not fit, the fit is reported once and your revision is kept.")
        return SequentialReAct(self.name, TS_SYSTEM, registry, self.llm, max_iterations=min(self.max_iterations, 5)).run(user)

    def _revise_tools(self, fused: dict, draft: list[dict], assignments: list[dict], rejected: list[dict]) -> list | None:
        lines = [f"step {c['step']}: {c['value']} — {'; '.join(c.get('issues') or [])}" for c in rejected]
        brief = [{"step": a.get("step"), "process": a.get("process"), "tool_id": a.get("tool_id"),
                  "spindle_rpm": a.get("spindle_rpm"), "feed_mm_min": a.get("feed_mm_min"),
                  "doc_mm": a.get("doc_mm"), "woc_mm": a.get("woc_mm")} for a in assignments if a.get("requires_tool")]
        user = (f"FEATURES:\n{describe_features(fused['features'])}\nYOUR ASSIGNMENTS:\n{json.dumps(brief)}\n"
                "FIT OBSERVATIONS (measurements, not a ban). Return assignments for the tooling steps. "
                "A library id is kept as you write it.\n- " + "\n- ".join(lines))
        obj = self.llm.json_call(TS_SYSTEM, user, agent=self.name + ".observe")
        got = obj.get("assignments")
        return got if isinstance(got, list) else None

    def _check(self, tool_id, st, feats, iso, fused) -> tuple[bool, list[str], dict]:
        t = self.by_id.get(tool_id or "")
        if not t or not st:
            return False, ["unknown tool or step"], {}
        geom = self.geometry(st["process"], [feats[i] for i in st.get("feature_ids", []) if i in feats], fused)
        issues = []
        if t["category"] not in self.lib["process_to_tool_category"].get(st["process"], []):
            issues.append("tool category does not match the process")
        if iso not in t["iso"]:
            issues.append(f"tool not rated for ISO {iso}")
        if t["id"] not in [c["id"] for c in self.candidates(st["process"], iso, geom)] and not issues:
            issues.append(f"does not fit the geometry {_brief(geom)}")
        return not issues, issues, geom

    def _merge(self, draft: list[dict], llm_assign: list, feats: dict, mat: dict | None, fused: dict, keep_unfit: bool = False) -> tuple[list[dict], list[dict]]:
        steps = {a["step"]: a for a in draft}
        iso = (mat or {}).get("iso_group", "P")
        changes = []
        out = {a["step"]: dict(a) for a in draft}
        for l in llm_assign:
            if not isinstance(l, dict) or l.get("step") not in steps:
                continue
            a = out[l["step"]]
            if not a["requires_tool"]:
                continue
            st = {"step": a["step"], "process": a["process"], "feature_ids": a["feature_ids"]}
            if l.get("tool_id") and l["tool_id"] != a.get("tool_id"):
                ok, issues, geom = self._check(l["tool_id"], st, feats, iso, fused)
                tool = self.by_id.get(l["tool_id"])
                if tool and (ok or keep_unfit):
                    if not ok:
                        geom = self.geometry(a["process"], [feats[i] for i in a.get("feature_ids", []) if i in feats], fused)
                    a = self._assign(a, tool, a["process"], mat, geom, why=str(l.get("justification") or "LLM choice"))
                    changes.append({"step": a["step"], "change": "tool", "value": l["tool_id"], "accepted": True, "fit": ok, "issues": issues})
                else:
                    # If model's tool is rejected, keep the draft tool rather than leaving the step without a tool
                    if not a.get("tool_id") and steps[l["step"]].get("tool_id"):
                        a = dict(steps[l["step"]])
                    changes.append({"step": a["step"], "change": "tool", "value": l["tool_id"], "accepted": False, "issues": issues or ["not in the tool library"]})
            machine = self.store.machine
            src = steps[l["step"]]
            if a.get("process") == "facing" and src.get("tool_category") == "face_mill" and a.get("tool_category") != "face_mill":
                a = dict(src)
                changes.append({"step": a["step"], "change": "tool", "value": src.get("tool_id"), "accepted": True,
                                "helped": "face mill kept: the smaller tool and its stepover do not cover the face"})
                out[a["step"]] = a
                continue
            for k in ("spindle_rpm", "feed_mm_min", "doc_mm", "woc_mm"):
                v = l.get(k)
                if isinstance(v, (int, float)) and v > 0 and a.get("tool_id"):
                    nv = float(v)
                    if k == "spindle_rpm":
                        nv = min(nv, float(machine["max_rpm"]))
                    elif k == "feed_mm_min":
                        nv = min(nv, float(machine["max_feed_mm_min"]))
                    if a.get(k) is None or abs(nv - a[k]) > 1e-6:
                        changes.append({"step": a["step"], "change": k, "value": v, "accepted": round(nv, 2),
                                        "clamped": abs(nv - float(v)) > 1e-6})
                        a[k] = int(nv) if k in ("spindle_rpm", "feed_mm_min") else round(nv, 2)
                        a["llm_adjusted"] = True
            if l.get("justification"):
                a["justification"] = str(l["justification"])[:200]
            out[a["step"]] = a
        return [out[s] for s in sorted(out)], changes


def _reach(t: dict) -> float:
    """Usable reach below the holder (stick-out minus about 25 mm in the collet / shrink chuck)."""
    return float(t.get("overall_length_mm") or (t.get("shape_params") or {}).get("Length") or 60) - 25.0


def _brief(geom: dict) -> str:
    return ", ".join(f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}" for k, v in geom.items() if v is not None and k not in ("op_type",))


def instance(tool: dict, geom: dict) -> dict:
    """A concrete tool: range tools (drills, reamers, boring bars) get the step's diameter."""
    t = dict(tool)
    if t.get("diameter_range_mm") and geom.get("diameter"):
        d = float(geom["diameter"])
        step = float(t.get("diameter_step_mm") or 0.1)
        if os.environ.get("CAM_TOOL_RANGE_FLOOR", "0") == "1":
            # Decimal preserves exact catalog grid points (e.g. 8.2 / 0.1).
            q = (Decimal(str(d)) / Decimal(str(step))).to_integral_value(rounding=ROUND_FLOOR)
            t["diameter_mm"] = round(float(q * Decimal(str(step))), 3)
        else:
            t["diameter_mm"] = round(round(d / step) * step, 3)
        t["shape_params"] = {**t.get("shape_params", {}), "Diameter": t["diameter_mm"]}
    if t["category"] == "tap" and geom.get("major"):
        t["diameter_mm"] = float(geom["major"])
        t["shape_params"] = {**t.get("shape_params", {}), "Diameter": t["diameter_mm"]}
    return t


def tooling_list(assignments: list[dict]) -> list[dict]:
    seen: dict[tuple, dict] = {}
    for a in assignments:
        if a.get("tool_id"):
            key = (a["tool_id"], a.get("tool_diameter_mm"))
            seen.setdefault(key, {"tool_id": a["tool_id"], "name": a.get("tool_name"), "category": a.get("tool_category"),
                                  "diameter_mm": a.get("tool_diameter_mm"), "steps": []})["steps"].append(a["step"])
    for i, t in enumerate(seen.values(), start=1):
        t["tool_number"] = i
    return list(seen.values())
