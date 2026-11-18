"""Knowledge Retrieval (KR) Agent: sequential ReAct (Fig. 7) or parallel ReAct with 3 sub-agents + coordinator (Figs. A1-A4).

Tool registry = Table 1: SQL retrieval, tabular retrieval, text retrieval (RAG), decision tree evaluation,
knowledge graph query, material database lookup, external fallback (generative, guarded).
"""
from __future__ import annotations

import json
import os

from ..knowledge import get_store
from ..knowledge.store import SOURCE_PRIORITY
from ..llm import LLM
from ..react import AgentResult, ParallelReAct, SequentialReAct
from ..tools import Tool, ToolRegistry
from .feature_extraction import describe_features

OUTPUT_SCHEMA = """OUTPUT SCHEMA (JSON object). Keep it COMPACT: one entry per distinct rule, with "feature_id" a list of
all feature ids it applies to (or "*"); never repeat the same rule for each of several identical holes; messages <= 20 words.
{
  "rules": [{"id": <rule id or source-derived id>, "feature_id": <list of feature ids, or "*">, "parameter": <str>, "value_or_range": <str>,
             "severity": <"critical"|"warning"|"info">, "source": <one of rag_text|tabular|decision_tree|sql_database|knowledge_graph|material_database|external>}],
  "violations": [{"feature_id": <id>, "rule_id": <id>, "severity": <"critical"|"warning">, "message": <str>, "recommendation": <str>, "source": <str>}],
  "recommendations": [{"feature_id": <id or "*">, "process": <process name>, "justification": <str>, "source": <str>}],
  "parameters": {<feature_id>: {<parameter>: <resolved value with unit>}},
  "summary": <overall feature assessment, 2-4 sentences>,
  "confidence": <"high"|"medium"|"low">
}"""

SUBAGENT_BUDGET = """ANSWER BUDGET (generation time is the bottleneck): at most 8 rules, 4 violations and 4 recommendations,
only items that change a process, tool or parameter decision for THESE features; "parameters" only for resolved values a
later step needs (tap drill, max depth ratio, cutting speed, coolant); value_or_range <= 6 words; summary 1 sentence."""

SEQUENTIAL_SYSTEM = f"""SYSTEM PROMPT: Manufacturing Knowledge Retrieval Agent

ROLE: Retrieve rules, constraints and recommendations from manufacturing knowledge sources for the given features and material.

PROCESS (ReAct loop): 1 Reason about which features/parameters need evidence -> 2 Act: call a knowledge tool ->
3 Observe the returned evidence -> 4 Repeat for other sources/features -> 5 Synthesize the final structured answer.
Every rule, violation and recommendation you output MUST carry the source tool it came from (provenance).

TOOL SELECTION (by feature type):
- Threads: sql_retrieval (thread specs, engagement), tabular_retrieval, decision_tree_evaluation, material_database_lookup
- Holes:   sql_retrieval, tabular_retrieval, text_retrieval, decision_tree_evaluation
- Pockets/slots/steps: sql_retrieval, tabular_retrieval, decision_tree_evaluation, text_retrieval
- Faces/chamfers/fillets: text_retrieval, tabular_retrieval, knowledge_graph_query
- Process relationships and ordering: knowledge_graph_query
- Any material question: material_database_lookup (always call it once)

SOURCE PRIORITY (high to low): 1 rag_text (text_retrieval)  2 tabular (tabular_retrieval)  3 decision_tree
4 sql_database (sql_retrieval)  5 knowledge_graph  6 material_database

CONFLICT RESOLUTION: material-specific values override generic values; on a same-priority conflict take the
conservative value; text ranges override fixed SQL values (e.g. tap drill 5.0-5.1 -> choose 5.0, conservative);
always keep the source for traceability.

FALLBACK GUARD: external_fallback (the generative model's own knowledge) is BLOCKED until all six local sources have
been called. Use it only when the local sources returned no data for the question, and label its output source "external".

Be efficient: issue several tool calls in the same turn, batch all features into ONE decision_tree_evaluation call,
call sql_retrieval / tabular_retrieval once per feature type, material_database_lookup once, and never repeat a call
with the same arguments. Aim to finish within 3 turns.
{OUTPUT_SCHEMA}"""

STRUCTURED_SUBAGENT = f"""SUB-AGENT: Structured Data Retrieval
ROLE: Query the SQL rules database, the decision-tree evaluator and the tabular CSV for rules and constraints.
TASK: dimensional constraints, process limits, thread specs, standards (ISO), if-then rules for EVERY feature given.
TOOLS: sql_retrieval (design rules, thread specs), tabular_retrieval (feature-material compatibility, coating),
decision_tree_evaluation (pass/fail checks with the exact dimensions of each feature).
SOURCE PROVENANCE: include "source" in every rule; prefer tabular over SQL when both give a value; if values differ,
include BOTH and mark them for coordinator resolution.
Turn 1: in the SAME turn call decision_tree_evaluation ONCE with all features batched (real dimensions), sql_retrieval
once per feature type and tabular_retrieval once per feature type. Turn 2: answer. Never repeat a call.
{OUTPUT_SCHEMA}
{SUBAGENT_BUDGET}"""

TEXT_SUBAGENT = f"""SUB-AGENT: Text Knowledge Retrieval
ROLE: Search natural-language DFM documents and the knowledge graph for qualitative guidance.
TASK: DFM guidelines, process relationships (which process is required / precedes another), material compatibility,
design insights for the given features.
TOOLS: text_retrieval (guidelines, best practices), knowledge_graph_query (process recommendations, material-feature relations).
SOURCE PRIORITY: rag_text has the HIGHEST priority of all sources; preserve full ranges from the text (e.g. 1.0-1.5x)
and include specific numeric values quoted from the text.
Turn 1: in the SAME turn issue one text_retrieval per feature type/topic (max 4) and one knowledge_graph_query per
feature type (max 4). Turn 2: answer. Never repeat a call.
{OUTPUT_SCHEMA}
{SUBAGENT_BUDGET}"""

MATERIAL_SUBAGENT = f"""SUB-AGENT: Material Constraints
ROLE: Find material-specific manufacturing constraints for the given features.
TASK: retrieve machinability, minimum feature size, maximum depth/diameter ratio, thread engagement, minimum fillet
radius, achievable surface finish, cutting speed ranges and coolant for the material, then translate them into rules
for each feature.
TOOLS: material_database_lookup (single tool). Call it once with the material name, then answer.
SOURCE PROVENANCE: material constraint values may be overridden by text, tabular or SQL sources; always set source
"material_database".
{OUTPUT_SCHEMA}
{SUBAGENT_BUDGET}"""

COORDINATOR = """COORDINATOR: Manufacturing Knowledge Coordinator
INPUT: the feature description and "merged_result": the outputs of 3 sub-agents (structured data, text knowledge, material),
already merged in code: rules/violations/recommendations de-duplicated, conflicts on the same parameter resolved by
source priority (rag_text > tabular > decision_tree > sql_database > knowledge_graph > material_database), every item
carries its provenance, "agreement" lists the parameters confirmed by several sources.
TASK: do NOT repeat the rules. Return only
{"summary": <overall feature assessment: key constraints, violations and recommended processes, 2-4 sentences>,
 "confidence": <"high" if the key values are confirmed by 3 sources, "medium" by 2, "low" by 1 or if a sub-agent failed>}"""


SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}


def merge_subagent_outputs(subs: list) -> dict:
    """Deterministic coordinator steps 1-4: merge + de-duplicate, resolve conflicts by source priority, keep provenance."""
    def prio(x: dict) -> int:
        src = str(x.get("source", ""))
        return SOURCE_PRIORITY.index(src) if src in SOURCE_PRIORITY else len(SOURCE_PRIORITY)

    def fids(x) -> tuple:
        v = x if isinstance(x, list) else [x]
        return tuple(sorted(str(i) for i in v if i is not None)) or ("*",)

    items = {k: [x for s in subs for x in (s.output.get(k) or []) if isinstance(x, dict)] for k in ("rules", "violations", "recommendations")}
    rules: dict = {}
    sources: dict = {}
    for r in sorted(items["rules"], key=prio):
        key = (str(r.get("parameter") or r.get("id")).lower(), fids(r.get("feature_id")))
        sources.setdefault(key[0], set()).add(r.get("source"))
        rules.setdefault(key, r)  # highest-priority source wins a conflict
    violations: dict = {}
    for v in sorted(items["violations"], key=lambda v: (SEVERITY_RANK.get(v.get("severity"), 1), prio(v))):
        violations.setdefault((str(v.get("feature_id")), str(v.get("rule_id"))), v)  # most severe kept
    recs: dict = {}
    for r in sorted(items["recommendations"], key=prio):
        key = (fids(r.get("feature_id")), str(r.get("process", "")).lower())
        if key in recs:
            if r.get("source") and r.get("source") not in str(recs[key].get("source")):
                recs[key] = {**recs[key], "source": f"{recs[key].get('source')}+{r.get('source')}"}
        else:
            recs[key] = r
    params: dict = {}
    for s in subs:  # sub-agents are ordered by source priority: the first value wins
        for fid, vals in (s.output.get("parameters") or {}).items():
            if isinstance(vals, dict):
                for k, val in vals.items():
                    params.setdefault(str(fid), {}).setdefault(k, val)
    return {"rules": list(rules.values()), "violations": list(violations.values()), "recommendations": list(recs.values()),
            "parameters": params, "agreement": sorted(p for p, src in sources.items() if len(src) > 1),
            "sub_agent_summaries": {s.agent: s.output.get("summary", "") for s in subs},
            "summary": "", "confidence": "low"}


def build_registry(llm: LLM) -> ToolRegistry:
    s = get_store()
    fb_calls: list[str] = []

    def decision_tree_evaluation(feature_type: str | None = None, material_group: str = "*", features: list | None = None, **dims) -> dict:
        """Single feature (feature_type + dims) or a batch: features=[{feature_id, feature_type, diameter, depth, ...}]."""
        if features:
            out = []
            for f in features:
                if not isinstance(f, dict):
                    continue
                f = dict(f)
                fid, ft = f.pop("feature_id", None), f.pop("feature_type", feature_type or "*")
                mg = f.pop("material_group", material_group)
                nums = {k: v for k, v in f.items() if isinstance(v, (int, float))}
                r = s.decision_tree_evaluation(ft, mg, **nums)
                out.append({"feature_id": fid, "feature_type": ft, "violations": r["violations"],
                            "passed": [c["rule_id"] for c in r["checks"] if c["pass"]], "evaluated": r["evaluated"]})
            return {"results": out, "count": len(out)}
        nums = {k: v for k, v in dims.items() if isinstance(v, (int, float))}
        return s.decision_tree_evaluation(feature_type or "*", material_group, **nums)

    def external_fallback(question: str) -> dict:
        r = llm.json_call("You are a senior manufacturing engineer. Answer the question with specific numbers and cite the "
                          "standard or rule of thumb. JSON: {\"answer\": str, \"values\": {..}, \"caveats\": str}", question,
                          agent="kr.external_fallback", thinking=False)
        fb_calls.append(question)
        return {"source": "external", **r}

    return ToolRegistry([
        Tool("sql_retrieval", "Relational rules DB: quantitative constraints (min/max values per feature type and material group), "
             "tolerance mappings, thread specs (tap drill sizes). Use feature_type in {hole, thread, pocket, slot, planar_face, chamfer, fillet, boss, *} "
             f"and material_group in {{{', '.join(s.material_groups())}, *}}. Optional raw SELECT via 'query'.",
             s.sql_retrieval, {"feature_type": {"type": "string"}, "material_group": {"type": "string"}, "parameter": {"type": "string"},
                               "query": {"type": "string", "description": "optional SELECT over tables rules / thread_specs"}}, ["feature_type"], "sql_database"),
        Tool("tabular_retrieval", "Structured CSV: feature-material compatibility, machinability score, recommended coating and tool material.",
             s.tabular_retrieval, {"feature_type": {"type": "string"}, "material_group": {"type": "string"}}, [], "tabular"),
        Tool("text_retrieval", "RAG over engineering guidelines (DFM rules, process rationale, material practice). Free-text query.",
             s.text_retrieval, {"query": {"type": "string"}, "top_k": {"type": "integer"}}, ["query"], "rag_text"),
        Tool("decision_tree_evaluation", "Rule-based pass/fail evaluation of feature dimensions against the applicable rules. BATCH all features in one call: "
             "features=[{feature_id, feature_type, diameter, depth, width, min_width, corner_radius, size, radius (mm), tolerance_it_grade, surface_finish_ra}]. "
             "Single-feature form: feature_type + the numbers as top-level arguments.",
             decision_tree_evaluation, {"features": {"type": "array", "items": {"type": "object"}}, "feature_type": {"type": "string"}, "material_group": {"type": "string"}, "diameter": {"type": "number"},
                                          "depth": {"type": "number"}, "width": {"type": "number"}, "min_width": {"type": "number"},
                                          "corner_radius": {"type": "number"}, "size": {"type": "number"}, "radius": {"type": "number"},
                                          "thread_engagement_x_d": {"type": "number"}, "tolerance_it_grade": {"type": "number"},
                                          "surface_finish_ra": {"type": "number"}, "flatness": {"type": "number"}}, [], "decision_tree"),
        Tool("knowledge_graph_query", "Graph of relations between features, processes, materials, tools and coatings "
             "(REQUIRES_PROCESS, PRECEDES, PREFERRED_FOR, AVOID_FOR, USES_TOOL, ACHIEVES_TOLERANCE, ACHIEVES_FINISH...). Entity e.g. 'thread', 'stainless', 'drilling'.",
             s.knowledge_graph_query, {"entity": {"type": "string"}, "relation": {"type": "string"}, "depth": {"type": "integer"}}, ["entity"], "knowledge_graph"),
        Tool("material_database_lookup", "Material properties: ISO 513 group, machinability, hardness, Kienzle kc1.1/mc, min feature size, max hole depth ratio, "
             "thread engagement, min fillet radius, cutting speeds by tool material and operation, coolant.", s.material_database_lookup, {"material": {"type": "string"}}, ["material"], "material_database"),
        Tool("external_fallback", "Generative-model knowledge for questions the local knowledge base cannot answer. Only allowed after all local sources were queried.",
             external_fallback, {"question": {"type": "string"}}, ["question"], "external"),
    ])


LOCAL_TOOLS = ["sql_retrieval", "tabular_retrieval", "text_retrieval", "decision_tree_evaluation", "knowledge_graph_query", "material_database_lookup"]


def fallback_guard(name: str, args: dict, called: list[str]) -> str | None:
    if name == "external_fallback":
        missing = [t for t in LOCAL_TOOLS if t not in called]
        if missing:
            return f"external_fallback is blocked until all local sources are queried. Not yet called: {missing}"
    return None


class KnowledgeRetrievalAgent:
    name = "knowledge_retrieval"

    def __init__(self, llm: LLM, mode: str = "parallel", max_iterations: int = 8) -> None:
        self.llm, self.mode, self.max_iterations = llm, mode, max_iterations
        self.registry = build_registry(llm)

    def build_input(self, fused: dict) -> str:
        mat = fused["material"]
        feats = describe_features(fused["features"])
        specs = [{"feature": f["id"], "specs": f["specs"]} for f in fused["features"] if f.get("specs")]
        return (f"MATERIAL: {mat.get('name')} (id {mat.get('id')}, group {mat.get('group')}"
                f"{', default: the material is not given in the inputs' if mat.get('source') == 'default' else ''})\n"
                f"PART: size {fused['part']['size']} mm, {len(fused['features'])} recognized features\n"
                f"GENERAL TOLERANCE: {fused.get('general_tolerance') or 'not specified'}{' (default)' if fused.get('general_tolerance_source') == 'default' else ''}; SURFACE FINISH: {fused.get('surface_finish_ra_um') or 'not specified'} um "
                f"(finish/tolerance requirements apply ONLY to the features listed under DRAWING SPECS; other features have general tolerance and as-machined finish)\n"
                f"FEATURES:\n{feats}\n"
                f"DRAWING SPECS MAPPED TO FEATURES: {json.dumps(specs, ensure_ascii=False)}\n"
                f"PROCESS ANNOTATIONS: {fused.get('process_annotations')}\n"
                "Retrieve the applicable rules, violations, recommendations and resolved parameters (e.g. tap drill sizes, "
                "max depth ratios, coating) for every feature type present.")

    def run(self, fused: dict) -> AgentResult:
        if os.environ.get("CAM_KR_SKIP_FACING_ONLY", "0") == "1":
            # A03 restores these faces for their own setup's facing template.
            # Global KR would turn them into wildcard recommendations in setup 1.
            fused = {**fused, "features": [f for f in fused["features"]
                     if not (f.get("facing_only") and f.get("setup", 1) > 1)]}
        user_input = self.build_input(fused)
        if self.mode == "sequential":
            agent = SequentialReAct(self.name, SEQUENTIAL_SYSTEM, self.registry, self.llm, max_iterations=self.max_iterations, guard=fallback_guard)
            res = agent.run(user_input)
        elif self.mode == "deterministic":
            res = self._deterministic(fused)
        else:
            subs = [  # ordered by source priority (rag_text first): merge_subagent_outputs relies on it
                SequentialReAct("kr.text", TEXT_SUBAGENT, self.registry.subset(["text_retrieval", "knowledge_graph_query"]), self.llm, max_iterations=min(self.max_iterations, 5)),
                SequentialReAct("kr.structured", STRUCTURED_SUBAGENT, self.registry.subset(["sql_retrieval", "tabular_retrieval", "decision_tree_evaluation"]), self.llm, max_iterations=min(self.max_iterations, 5)),
                SequentialReAct("kr.material", MATERIAL_SUBAGENT, self.registry.subset(["material_database_lookup"]), self.llm, max_iterations=3),
            ]
            res = ParallelReAct(self.name, subs, COORDINATOR, self.llm, merge=merge_subagent_outputs).run(user_input)
            # fallback detection (parallel mode): missing local coverage (unknown material, or no evidence at all)
            # triggers one guarded external_fallback call whose answer is merged with provenance "external"
            unknown_material = not fused["material"].get("id")
            if unknown_material or (not res.output.get("rules") and not res.output.get("recommendations")):
                q = (f"Material '{fused['material'].get('input')}' is not in the local material database. Give machinability (% of B1112), "
                     f"recommended cutting speeds (m/min) for carbide and HSS, max hole depth/diameter ratio, coolant and tooling/coating advice, "
                     f"for these features:\n{describe_features(fused['features'])}") if unknown_material else user_input[:2000]
                fb = self.registry.get("external_fallback")(question=q)
                res.output.setdefault("recommendations", []).append({"feature_id": "*", "process": "material guidance (external)", "justification": str(fb.get("answer", ""))[:1200], "source": "external"})
                res.output.setdefault("parameters", {}).setdefault("*", {}).update({k: v for k, v in (fb.get("values") or {}).items()} if isinstance(fb.get("values"), dict) else {})
                res.output["confidence"] = "low"
                res.tool_calls.append({"name": "external_fallback", "arguments": {"question": q[:200]}, "result_preview": str(fb)[:300]})
        res.output = _normalize_output(res.output)
        self.llm.trace.log("agent", agent=self.name, mode=self.mode, tools=res.tools_called, iterations=res.iterations, tokens=res.tokens)
        return res

    def _deterministic(self, fused: dict) -> AgentResult:
        """Mode 1 of Sec. 3.4.2: fixed retrieval without an LLM loop."""
        s = get_store()
        grp = fused["material"]["group"]
        out = {"rules": [], "violations": [], "recommendations": [], "parameters": {}, "summary": "", "confidence": "medium"}
        calls = []
        for f in fused["features"]:
            ft = "thread" if f.get("subtype") == "thread" else f["type"]
            dims = {k: f[k] for k in ("diameter", "depth", "min_width", "corner_radius", "size", "radius") if k in f and isinstance(f[k], (int, float))}
            spec = get_store().thread(f.get("thread")) if ft == "thread" else None
            if spec and f.get("depth"):
                dims["thread_engagement_x_d"] = round(float(f["depth"]) / spec["major_mm"], 2)
            ev = s.decision_tree_evaluation(ft, grp, **dims)
            calls.append({"name": "decision_tree_evaluation", "arguments": {"feature_type": ft, **dims}})
            for vio in ev["violations"]:
                out["violations"].append({"feature_id": f["id"], "rule_id": vio["rule_id"], "severity": vio["severity"], "message": vio["message"], "recommendation": "", "source": "decision_tree"})
            for c in ev["checks"]:
                out["rules"].append({"id": c["rule_id"], "feature_id": f["id"], "parameter": c["parameter"], "value_or_range": f"{c['range']} {c['unit']}", "severity": c["severity"], "source": "decision_tree"})
        for ft in sorted({("thread" if f.get("subtype") == "thread" else f["type"]) for f in fused["features"]}):
            tab = s.tabular_retrieval(ft, grp)
            calls.append({"name": "tabular_retrieval", "arguments": {"feature_type": ft, "material_group": grp}})
            for r in tab["rows"]:
                out["recommendations"].append({"feature_id": "*", "process": ft, "justification": f"{r['recommended_tool_material']} {r['recommended_coating']}: {r['notes']}", "source": "tabular"})
        m = s.material_database_lookup(fused["material"]["id"] or "")
        calls.append({"name": "material_database_lookup", "arguments": {"material": fused["material"]["id"]}})
        if "error" not in m:
            out["parameters"]["*"] = {"iso_group": m["iso_group"], "machinability_pct": m["machinability_pct"], "coolant": m["coolant"],
                                      "cutting_speed_m_min": m["cutting_speed_m_min"]}
        out["summary"] = f"Deterministic retrieval: {len(out['rules'])} rules, {len(out['violations'])} violations."
        return AgentResult(self.name, out, tool_calls=[{**c, "result_preview": ""} for c in calls], iterations=1)


def _normalize_output(o: dict) -> dict:
    o = dict(o or {})
    for k in ("rules", "violations", "recommendations"):
        v = o.get(k)
        o[k] = [x for x in v if isinstance(x, dict)] if isinstance(v, list) else []
    if not isinstance(o.get("parameters"), dict):
        o["parameters"] = {}
    o.setdefault("summary", "")
    o.setdefault("confidence", "low")
    for v in o["violations"]:
        v.setdefault("severity", "warning")
    return o


__all__ = ["KnowledgeRetrievalAgent", "build_registry", "SOURCE_PRIORITY"]
