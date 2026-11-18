"""Process Sequence (PS) Agent.

The catalog draft is a suggestion. Outside deterministic mode the LLM writes the sequence that is machined.
Name normalisation and the split into one FreeCAD operation per hole group stay, because the post cannot emit
a program otherwise. Rule validation is recorded and does not replace the sequence. There is no repair pass.
"""
from __future__ import annotations

import json

from ..knowledge import get_store
from ..llm import LLM
from ..normalizer import Normalizer
from ..planning import sequencing as S
from ..react import AgentResult, SequentialReAct
from ..tools import Tool, ToolRegistry
from .feature_extraction import describe_features

PS_SYSTEM = """SYSTEM PROMPT: Process Sequence Agent
ROLE: Decide the manufacturing process sequence for one setup. Your sequence is machined as you write it.
A draft is attached as a hint: feature ids, catalog process names and one possible order. Keep, drop, reorder or
add steps. You choose what is cut and in what order. Only catalog process names. Every cutting step must list the
feature_ids it machines (use the ids from the feature list; "*" only for setup, cleaning, inspection, coating).
Use process_capability_lookup when you are unsure which process holds a tolerance, and knowledge_search for rationale.
Aim to finish within 3 turns.
OUTPUT SCHEMA (JSON object):
{"sequence": [{"step": <int>, "process": <catalog name>, "feature_ids": [<ids>], "rationale": <short>}],
 "alternatives": [{"description": <str>, "sequence": [<process names>]}],
 "source": "generated", "notes": <str>}"""

HOLE_PROCS = {"spot_drilling", "drilling", "reaming", "boring", "tapping", "thread_milling", "counterboring", "countersinking"}


class ProcessSequenceAgent:
    name = "process_sequence"

    def __init__(self, llm: LLM, mode: str = "parallel", max_iterations: int = 8, sequencing: str = "standard") -> None:
        self.llm, self.mode, self.max_iterations, self.sequencing = llm, mode, max_iterations, sequencing
        self.store = get_store()
        catalog = self.store.templates["process_catalog"]
        aliases = {k: [k.replace("_", " "), v["description"].split(" (")[0]] for k, v in catalog.items()}
        extra = {"drilling": ["drill", "drill holes"], "tapping": ["tap", "thread", "threading holes", "rigid tapping"],
                 "facing": ["face", "face mill", "facing and datum establishment"], "pocket_milling": ["pocket", "pocketing", "rough pocket"],
                 "slot_finishing": ["finish slot", "slot finish"], "cavity_roughing": ["cavity", "z-level roughing", "3d roughing", "area clearing"],
                 "surface_finishing": ["3d finishing", "waterline", "surface finish pass"], "profile_milling": ["contour", "profiling", "profile"],
                 "heat_treatment": ["heat treat", "harden", "quench and temper", "hardening", "stress relieve"],
                 "coating": ["anodize", "anodizing", "plating", "paint", "passivate", "passivation"], "inspection": ["inspect", "cmm", "qc"],
                 "deburring": ["deburr", "break edges"], "countersinking": ["countersink", "csk"], "counterboring": ["counterbore", "cbore"],
                 "reaming": ["ream", "h7"], "thread_milling": ["thread mill"], "step_milling": ["step", "shoulder milling"]}
        for k, v in extra.items():
            aliases.setdefault(k, []).extend(v)
        self.proc_norm = Normalizer(aliases, None if mode == "deterministic" else llm, "process")  # deterministic: no LLM call

    # ------------------------------------------------------------------ public
    def run(self, fused: dict, kr_output: dict, setup: int = 1) -> AgentResult:
        feats = fused["features"]
        grp = fused["material"]["group"]
        moves: list[str] = []
        if fused.get("user_sequence_override"):  # Path 1: the user's order is kept as given, only validated
            seq, source = self._from_names(fused["user_sequence_override"]), "user_override"
        else:
            seq, source = self._generate(feats, grp, kr_output), "generated"
            annotated = [self.proc_norm.normalize(a)[0] for a in fused.get("process_annotations", [])]
            annotated = [a for a in annotated if a and a not in [s["process"] for s in seq]]
            if annotated:  # Path 2
                source = "design_specified"
                seq += [{"process": p, "feature_ids": ["*"], "rationale": "design-specified process annotation"} for p in annotated]
            seq = S.order(seq, feats, self.sequencing)
        draft_seq = [dict(s) for s in seq]
        validation = S.validate(seq, feats, grp, fused, self.sequencing)
        result = {"sequence": S.number(seq, setup), "alternatives": [], "source": source, "notes": ""}
        tool_calls = [{"name": "sequence_retrieval", "arguments": {}, "result_preview": ""}]
        iterations, tokens, llm_used = 1, 0, False
        if self.mode != "deterministic":
            res = self._react(fused, kr_output, result)
            tool_calls += res.tool_calls
            iterations, tokens, llm_used = res.iterations, res.tokens, True
            refined = self._from_llm(res.output.get("sequence") or [], feats) if res.success else []
            if refined:
                seq = self._split_groups(refined, feats)
                obs = self._observations(seq, feats)
                note = res.output.get("notes", "") or "LLM sequence kept"
                if obs:
                    revised = self._revise_sequence(fused, seq, obs)
                    tokens += self.llm.last_usage.total
                    if revised:
                        seq = self._split_groups(revised, feats)
                        note = (note + " Observations were shown once; the revised sequence is kept.").strip()
                    else:
                        note = (note + " Observations were shown once; the first sequence is kept.").strip()
                added = self._restore_finish(seq, draft_seq)
                if added:
                    note = (note + f" Calculated surface finishing kept for {added}.").strip()
                validation = S.validate(seq, feats, grp, fused, self.sequencing)
                result = {"sequence": S.number(seq, setup), "alternatives": res.output.get("alternatives", []) or [],
                          "source": "generated", "notes": note}
        if not result["alternatives"]:
            result["alternatives"] = self._alternatives(seq)
        base = self.store.deps["base_confidence"][result["source"]]
        n_warn = len(validation["warnings"])
        confidence = round(base - 0.3 * (1 if validation["critical"] else 0) - 0.1 * min(n_warn / 5, 1.0), 3)  # Eq. 4
        result.update({"validation": validation, "confidence": confidence, "material_group": grp, "repair_moves": moves,
                       "llm_refined": llm_used, "sequencing": self.sequencing})
        self.llm.trace.log("agent", agent=self.name, source=result["source"], steps=len(seq), confidence=confidence,
                           critical=len(validation["critical"]), warnings=n_warn, llm=llm_used, moves=len(moves))
        return AgentResult(self.name, result, tool_calls=tool_calls, iterations=iterations, tokens=tokens)

    # --------------------------------------------------------------- Path 3
    def _generate(self, feats: list[dict], grp: str, kr_output: dict) -> list[dict]:
        fp = self.store.templates["feature_processes"]
        steps: dict[tuple, dict] = {}

        def add(p, fid, why, group=""):
            st = steps.setdefault((p, group), {"process": p, "feature_ids": [], "rationale": why})
            for one in (fid if isinstance(fid, (list, tuple, set)) else [fid]):
                if isinstance(one, str) and one not in st["feature_ids"]:
                    st["feature_ids"].append(one)

        add("setup", "*", "clamp the stock, set the work offset")
        cavity = any(f["type"] == "cavity" for f in feats)
        for f in feats:
            sub = f.get("subtype", f["type"])
            if cavity and f["type"] in ("pocket", "slot", "step", "boss", "outer_profile", "fillet"):
                continue  # the height-map roughing / finishing of the cavity machines everything reachable from this side
            if f["type"] == "planar_face" and (f.get("role") != "top" or not f.get("needs_facing", True)):
                continue
            if f["type"] == "outer_profile" and not f.get("needs_profile", True):
                continue
            if f["type"] == "hole" and not f.get("vertical", True):
                continue  # not reachable in this setup; reported as a warning
            if sub == "thread" and (grp in ("stainless", "titanium", "superalloy") and (self.store.thread(f.get("thread")) or {}).get("major_mm", 0) >= 10
                                    or grp == "hardened steel"):
                sub = "thread_milled"
            for p in fp.get(sub, fp.get(f["type"], [])):
                group = ""
                if f["type"] == "hole":
                    # one operation per hole pattern: same tool, same start level and depth (FreeCAD drills one depth per op)
                    group = ("spot", round(float(f.get("top_z", 0)), 2)) if p == "spot_drilling" else (
                        round(float(f["diameter"]), 3), bool(f.get("through")), f.get("thread") or "", round(float(f.get("depth", 0)), 2),
                        round(float(f.get("top_z", 0)), 2), f.get("tolerance_grade") or "")
                elif f["type"] == "chamfer":
                    group = "45" if abs(float(f.get("angle_deg") or 45) - 45) <= 3 else f"other{f['id']}"
                elif f["type"] in ("pocket", "slot", "step", "boss", "planar_face"):
                    group = f["id"]
                elif f["type"] in ("outer_profile", "cavity"):
                    group = f["type"]
                why = f"{sub} {f['id']}" + (f" (d={f['diameter']} {'through' if f['through'] else 'blind'}{', ' + f['thread'] if f.get('thread') else ''})" if f["type"] == "hole" else "")
                add(p, f["id"], why, group)
        for st in steps.values():
            if len(st["feature_ids"]) > 1:
                st["rationale"] = f"{st['process']} of {', '.join(st['feature_ids'])}"
        # accuracy beyond milling: an outside face whose IT grade or Ra no planned milling pass holds is ground afterwards
        catalog = self.store.templates["process_catalog"]
        for f in feats:
            if f["type"] not in ("planar_face", "boss", "outer_profile", "step"):
                continue
            it, ra = S._it(f.get("tolerance_grade")), f.get("surface_finish_ra_um")
            if it is None and not ra:
                continue
            own = [k[0] for k, st in steps.items() if f["id"] in st["feature_ids"]]
            best_it = min([catalog.get(p, {}).get("tolerance_it") or 99 for p in own] or [99])
            best_ra = min([float(catalog.get(p, {}).get("ra_um") or 99) for p in own] or [99.0])
            if (it is not None and it < best_it) or (ra and float(ra) < best_ra - 1e-9):
                add("grinding", f["id"], f"{f['id']} {f.get('tolerance_grade') or ''} Ra {ra or '-'}: beyond milling accuracy", "grinding")
        present = {k[0] for k in steps}
        alternatives = {"tapping": "thread_milling", "thread_milling": "tapping", "reaming": "boring", "boring": "reaming"}
        for r in kr_output.get("recommendations", []):
            p = self.proc_norm.normalize(str(r.get("process", "")))[0]
            if p and p in self.store.templates["process_catalog"] and p not in present and alternatives.get(p) not in present \
                    and p not in ("coating", "heat_treatment", "grinding", "setup", "cavity_roughing", "surface_finishing"):
                add(p, r.get("feature_id", "*"), f"KR recommendation ({r.get('source')})")
                present.add(p)
        for p in self.store.deps["mandatory_final"]:
            add(p, "*", "mandatory final step")
        add("cleaning", "*", "remove chips and coolant before inspection")
        incompatible = self.store.templates["material_process_compatibility"].get(grp, {}).get("incompatible", [])
        return [s for s in steps.values() if s["process"] not in incompatible]

    def _split_groups(self, seq: list[dict], feats: list[dict]) -> list[dict]:
        """A step must be one FreeCAD operation: holes of different diameter / depth / start level and several
        pockets or slots are split into consecutive steps (the order is kept)."""
        by_id = {f["id"]: f for f in feats}
        out = []
        for st in seq:
            fids = S._fids(st)
            own = [by_id[x] for x in fids if x in by_id]
            if st["process"] in HOLE_PROCS and len(own) > 1:
                groups: dict = {}
                for f in own:
                    k = (round(float(f.get("top_z", 0)), 2),) if st["process"] == "spot_drilling" else (
                        round(float(f.get("diameter", 0)), 3), bool(f.get("through")), f.get("thread") or "", round(float(f.get("depth", 0)), 2), round(float(f.get("top_z", 0)), 2))
                    groups.setdefault(k, []).append(f["id"])
                out += [{**st, "feature_ids": g} for g in groups.values()]
            elif st["process"] in ("pocket_milling", "pocket_finishing", "slot_milling", "slot_finishing", "step_milling", "counterboring") and len(own) > 1:
                out += [{**st, "feature_ids": [f["id"]], "rationale": f"{st['process']} {f['id']}"} for f in own]
            else:
                out.append(st)
        return out

    def _from_names(self, names: list[str]) -> list[dict]:
        seq = []
        for n in names:
            p = self.proc_norm.normalize(n)[0]
            if p:
                seq.append({"process": p, "feature_ids": ["*"], "rationale": f"user specified '{n}'"})
        return seq

    def _from_llm(self, steps: list, feats: list[dict]) -> list[dict]:
        valid_ids = {f["id"] for f in feats}
        seq = []
        for st in steps:
            if not isinstance(st, dict):
                continue
            p = self.proc_norm.normalize(str(st.get("process", "")))[0]
            if not p:
                continue
            raw = st.get("feature_ids") or []
            raw = raw if isinstance(raw, (list, tuple)) else [raw]
            fids = [x for x in raw if isinstance(x, str) and (x in valid_ids or x == "*")] or ["*"]
            seq.append({"process": p, "feature_ids": fids, "rationale": str(st.get("rationale", ""))[:200]})
        return seq

    def _observations(self, seq: list[dict], feats: list[dict]) -> list[str]:
        """Facts for the model, in the role frontier `submit` plays: report, do not rewrite."""
        by_id = {f["id"]: f for f in feats}
        cavities = [f["id"] for f in feats if f.get("type") == "cavity"]
        if not cavities:
            return []
        covered = {"pocket", "slot", "step", "boss", "outer_profile"}
        procs = {"pocket_milling", "pocket_finishing", "slot_milling", "slot_finishing",
                 "profile_milling", "profile_finishing", "step_milling"}
        notes = []
        rough = [fid for st in seq if st["process"] == "cavity_roughing" for fid in (st.get("feature_ids") or [])]
        finished = {fid for st in seq if st["process"] == "surface_finishing" for fid in (st.get("feature_ids") or [])}
        missing = [fid for fid in dict.fromkeys(rough) if fid not in finished and fid != "*"]
        if missing:
            notes.append("cavity roughing on " + ", ".join(missing) + " has no surface_finishing. "
                         "The roughing pass leaves a finish allowance; the calculated surface_finishing is what brings that surface within tolerance.")
        if sum(1 for st in seq if st["process"] == "cavity_roughing") >= 2:
            notes.append("cavity_roughing is listed more than once. A second roughing pass does not replace surface_finishing.")
        for st in seq:
            if st["process"] not in procs:
                continue
            ids = [x for x in st.get("feature_ids") or [] if by_id.get(x, {}).get("type") in covered]
            if ids:
                notes.append(f"{st['process']} on {', '.join(ids)}: cavity {', '.join(cavities)} on this setup already "
                             "machines pockets, slots, steps and the outer profile")
        return notes

    def _revise_sequence(self, fused: dict, seq: list[dict], observations: list[str]) -> list[dict]:
        brief = [{"process": s["process"], "feature_ids": s["feature_ids"], "rationale": s.get("rationale", "")} for s in seq]
        user = (f"FEATURES:\n{describe_features(fused['features'])}\nYOUR SEQUENCE:\n{json.dumps(brief)}\n"
                "OBSERVATIONS (measurements, not an order). Return the full sequence that should be machined. "
                "It is kept as you write it.\n- " + "\n- ".join(observations))
        obj = self.llm.json_call(PS_SYSTEM, user, agent=self.name + ".observe")
        return self._from_llm(obj.get("sequence") or [], fused["features"])

    def _restore_finish(self, seq: list[dict], draft: list[dict]) -> list:
        """The model often replaces the calculated finish pass with a second roughing. Put that finish back."""
        finished = {fid for st in seq if st["process"] == "surface_finishing" for fid in (st.get("feature_ids") or [])}
        rough = {fid for st in seq if st["process"] == "cavity_roughing" for fid in (st.get("feature_ids") or []) if fid != "*"}
        added = []
        for st in draft:
            if st["process"] != "surface_finishing":
                continue
            ids = [fid for fid in (st.get("feature_ids") or []) if fid in rough and fid not in finished]
            if not ids:
                continue
            seq.append({"process": "surface_finishing", "feature_ids": list(st["feature_ids"]),
                        "rationale": "calculated finish pass kept for the cavity the sequence roughed"})
            finished.update(st.get("feature_ids") or [])
            added.append(list(st["feature_ids"]))
        return added

    def _alternatives(self, seq: list[dict]) -> list[dict]:
        names = [s["process"] for s in seq]
        alts = []
        if "tapping" in names:
            alts.append({"description": "Thread milling instead of tapping (no tap breakage, one tool for several sizes, better in hard materials)",
                         "sequence": [("thread_milling" if n == "tapping" else n) for n in names]})
        if "reaming" in names:
            alts.append({"description": "Fine boring instead of reaming (position correction, any diameter)", "sequence": [("boring" if n == "reaming" else n) for n in names]})
        if "spot_drilling" in names:
            alts.append({"description": "No spot drilling with self-centring 140 deg carbide drills on flat faces (position tolerance >= 0.1 mm)",
                         "sequence": [n for n in names if n != "spot_drilling"]})
        return alts[:3]

    # ---------------------------------------------------------------- ReAct
    def _react(self, fused: dict, kr_output: dict, draft: dict) -> AgentResult:
        s, feats = self.store, fused["features"]

        registry = ToolRegistry([
            Tool("process_capability_lookup", "Process catalog: description, standard-practice phase, achievable IT grade and Ra, prerequisites. Omit 'process' to list all.",
                 s.process_capability_lookup, {"process": {"type": "string"}}, [], "process_templates"),
            Tool("sequence_retrieval", "Template sequence, process phases and the processes each feature type needs.",
                 s.sequence_retrieval, {"part_category": {"type": "string"}, "feature_types": {"type": "array", "items": {"type": "string"}}}, [], "process_templates"),
            Tool("knowledge_search", "Text search over machining guidelines.", s.text_retrieval,
                 {"query": {"type": "string"}}, ["query"], "rag_text"),
        ])
        brief = [{k: st[k] for k in ("step", "process", "feature_ids")} for st in draft["sequence"]]
        user = (f"MATERIAL: {fused['material']}\nFEATURES:\n{describe_features(feats)}\n"
                f"PROCESS ANNOTATIONS: {fused.get('process_annotations')}\nUSER OVERRIDE: {fused.get('user_sequence_override')}\n"
                f"RETRIEVED NOTES: {json.dumps({'recommendations': kr_output.get('recommendations', [])}, ensure_ascii=False)[:5000]}\n"
                f"DRAFT (hint only): {json.dumps(brief)}\n"
                "Return the full sequence that should be machined.")
        return SequentialReAct(self.name, PS_SYSTEM, registry, self.llm, max_iterations=min(self.max_iterations, 5)).run(user)
