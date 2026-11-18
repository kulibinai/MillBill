"""Report Generation Agent: template-based report with the paper's seven sections (Sec. 3.7) plus the CAM and
verification results. Detail levels: executive | standard | detailed. Only the executive summary is LLM-written."""
from __future__ import annotations

import json
from pathlib import Path

from ..llm import LLM
from .feature_extraction import describe_features

NON_CAM_PROCESSES = frozenset({"setup", "fixturing", "cleaning", "inspection", "coating", "plating", "heat_treatment", "stress_relief", "marking"})


def cam_steps(sequence: list[dict]) -> list[dict]:
    return [s for s in sequence if s.get("process") not in NON_CAM_PROCESSES]


SUMMARY_PROMPT = """You write the executive summary of a manufacturing process plan report. Given the structured plan,
write 4-6 sentences: part and material, main features, process route (setups, roughing/finishing/hole making), tooling
highlights, verification result (surface share within tolerance, cycle time) and the main risks. Plain prose, no lists,
no markdown. Only mention cutting operations, not fixture setup, washing or inspection. JSON: {"summary": <text>}"""


def _v(x, nd=2):
    return "-" if x is None else (round(x, nd) if isinstance(x, float) else x)


class ReportGenerationAgent:
    name = "report_generation"

    def __init__(self, llm: LLM, detail: str = "standard") -> None:
        self.llm, self.detail = llm, detail

    def run(self, session: dict) -> str:
        fused, kr, ps, ts, cam = session["fused"], session["kr"], session["ps"], session["ts"], session.get("cam") or {}
        ver = cam.get("verification") or {}
        stats = (ver.get("deviation") or {}).get("stats") or {}
        multi = (session.get("multi_setup") or {}).get("final")
        budget_dropped = any(e.get("budget") for e in session.get("setups", []))
        cam_ok = False if budget_dropped else ver.get("ok")
        payload = {"material": fused["material"], "part_size": fused["part"]["size"], "n_features": len(fused["features"]),
                   "setups": [e["up"] for e in session.get("setups") or []] or ["+Z"], "sequence": [s["process"] for s in cam_steps(ps["sequence"])],
                   "tools": [t["tool_id"] for t in ts["tooling_list"]], "critical": ps["validation"]["critical"][:5],
                   "surface_within_tol_pct": (multi or stats).get("surface_within_tol_pct"), "tol_mm": session.get("tol_mm"),
                   "cycle_time_min": (ver.get("cycle_time") or {}).get("total_min"), "cam_ok": cam_ok,
                   "issues": (["Time budget exhausted; machining incomplete"] if budget_dropped else []) + ver.get("issues", [])[:5]}
        try:
            summary = self.llm.json_call(SUMMARY_PROMPT, json.dumps(payload, ensure_ascii=False, default=str), agent=self.name, thinking=False).get("summary", "")
        except Exception as e:  # noqa: BLE001  (LLM unreachable: plan, G-code and session must still be saved)
            summary = (f"_LLM summary unavailable ({type(e).__name__})._ {payload['n_features']} features in {(fused['material'] or {}).get('name')}, "
                       f"{len(payload['sequence'])} cutting steps, {len(payload['tools'])} tools, {payload['surface_within_tol_pct']} % of the surface within "
                       f"+-{payload['tol_mm']} mm, cycle time {payload['cycle_time_min']} min.")
        L = [f"# Manufacturing Process Plan: {session['part_name']}\n",
             f"_cam_agent - model {session['model']}, agent mode {session['mode']}, sequencing {session.get('sequencing')}, tolerance +-{session.get('tol_mm')} mm_\n",
             "## 1. Executive Summary\n", summary or "-"]
        L.append(f"\nConfidence: knowledge retrieval **{kr.get('confidence')}**, process sequence **{ps['confidence']}** (Eq. 4), tooling **{ts['confidence']}** (Eq. 7)"
                 + (f", CAM verification **{'passed' if cam_ok else 'issues'}**" if cam.get("available") else "") + ".\n")

        L.append("## 2. Design Specifications\n")
        m, md = fused["material"], fused.get("material_data") or {}
        L.append(f"- Material: {m.get('name')} (`{m.get('id')}`, ISO 513 group {md.get('iso_group', '?')}, {m.get('group')}, normalized by {m.get('normalized_by')}{'; default: not given in the inputs' if m.get('source') == 'default' else ''})")
        p = fused["part"]
        L.append(f"- Part envelope: {p['size'][0]} x {p['size'][1]} x {p['size'][2]} mm, volume {round(p.get('volume', 0) / 1000, 2)} cm3")
        stock = session.get("stock") or {}
        if stock.get("kind") == "user":
            L.append(f"- Stock: user blank `{Path(stock['source']).name}`, {stock.get('stock_cm3')} cm3, to remove {stock.get('removal_cm3')} cm3, "
                     f"part outside the blank {stock.get('part_outside_stock_pct')} %")
        else:
            mg = stock.get("margins") or {}
            L.append(f"- Stock: bounding box + {mg.get('xy_margin')} mm per side in XY, + {mg.get('z_top')} mm on top, + {mg.get('z_bottom')} mm below")
        if stock.get("allowance"):
            L.append(f"- Allowance around the part (setup 1): {stock['allowance']} mm")
        L.append(f"- General tolerance: {fused.get('general_tolerance') or 'not specified'}{' (default)' if fused.get('general_tolerance_source') == 'default' else ''}; surface finish: {fused.get('surface_finish_ra_um') or 'not specified'} um Ra")
        L.append(f"- Process annotations: {', '.join(fused.get('process_annotations') or []) or 'none'}")
        L.append("\nRecognised features:\n")
        L.append("```\n" + describe_features(fused["features"]) + "\n```")
        if fused.get("mappings"):
            L.append("\nDrawing annotation mapping (Context Fusion):\n")
            L.append("| # | annotation | features | confidence | method | review |\n|---|---|---|---|---|---|")
            anns = session["extracted"]["drawing"].get("annotations", [])
            for mp in fused["mappings"]:
                a = anns[mp["annotation_index"]] if mp["annotation_index"] < len(anns) else {}
                L.append(f"| {mp['annotation_index']} | {a.get('text', '')} | {', '.join(mp['feature_ids']) or '-'} | {mp['confidence']} | {mp['method']} | {'yes' if mp['review'] else ''} |")

        L.append("\n## 3. Manufacturing Rules\n")
        if kr.get("summary"):
            L.append(kr["summary"] + "\n")
        rules = kr.get("rules", [])
        if rules:
            L.append("| id | feature | parameter | value / range | severity | source |\n|---|---|---|---|---|---|")
            for r in rules[: (80 if self.detail == "detailed" else 25)]:
                L.append(f"| {r.get('id', '')} | {r.get('feature_id', '')} | {r.get('parameter', '')} | {r.get('value_or_range', '')} | {r.get('severity', '')} | {r.get('source', '')} |")

        L.append("\n## 4. Process Plan\n")
        self._plan(L, ps, ts)
        L.append("\n## 5. Tool Selection and Cutting Data\n")
        self._tools(L, ts)
        if cam.get("available"):
            L.append("\n## 5b. CAM Job and Verification\n")
            self._cam(L, cam, session.get("tol_mm"))
        if len(session.get("setups") or []) > 1:
            L.append("\n## 5c. Multi-setup (indexed) machining\n")
            sp = session.get("setup_plan", {})
            L.append(f"Setups: {', '.join('S%d %s' % (e['index'], e['up']) for e in session['setups'])}. Void not reachable from these sides: "
                     f"{sp.get('unreachable_cm3')} cm3 (from no axis direction: {sp.get('unreachable_from_any_axis_cm3')} cm3). Sections 4-5b describe setup 1.\n")
            for e in session["setups"][1:]:
                if e.get("dropped"):
                    reason = "time budget exhausted; machining incomplete" if e.get("budget") else "nothing left to machine from this side"
                    L.append(f"### Setup {e['index']} ({e['up']}): dropped - {reason}\n")
                    continue
                L.append(f"### Setup {e['index']}: machining side {e['up']}\n")
                self._plan(L, e["ps"], e["ts"])
                self._tools(L, e["ts"])
                if (e.get("cam") or {}).get("available"):
                    self._cam(L, e["cam"], session.get("tol_mm"))
            if multi:
                L.append(f"\n**All setups combined:** {multi['surface_within_tol_pct']} % of the target surface within +-{multi['tol_mm']} mm "
                         f"({multi['surface_within_tol_pct_excl_expected']} % without drill-point cones; another {multi.get('band_pct')} % out by less than "
                         f"{multi.get('band_factor')}x the tolerance); material left {multi['excess_cm3']} cm3 "
                         f"(on {multi['excess_area_pct']} % of the surface), overcut {multi['gouge_cm3']} cm3 (on {multi['gouge_area_pct']} %).")

        L.append("\n## 6. Risk Analysis\n")
        risks = [f"- [critical] {c['message']}" for c in ps["validation"]["critical"]]
        risks += [f"- [{v.get('severity', 'warning')}] {v.get('feature_id', '')}: {v.get('message', '')} ({v.get('source', '')})" for v in kr.get("violations", [])]
        risks += [f"- [warning] {w['message']}" for w in ps["validation"]["warnings"]]
        risks += [f"- [tooling] step {a['step']} {a['process']}: {a['warning']}" for a in ts["assignments"] if a.get("warning")]
        risks += [f"- [cutting data] step {a['step']}: {n}" for a in ts["assignments"] for n in a.get("notes") or [] if "exceeds" in n]
        risks += [f"- [review] annotation {mp['annotation_index']} could not be mapped confidently" for mp in fused.get("needs_review", [])]
        coverage = (multi or stats).get("brep_coverage") or {}
        risks += [f"- [surface sampling] {w}" for w in coverage.get("warnings", [])]
        if coverage.get("incomplete"):
            risks.append(f"- [surface sampling] Unsampled area {coverage['unsampled_area_mm2']:.6g} mm2; "
                         f"zero-triangle face IDs (1-based): {[r['face_id'] for r in coverage['zero_triangle_faces']]}; "
                         f"invalid face IDs: {coverage['invalid_face_ids']}.")
        L.append("\n".join(risks) if risks else "No violations or warnings were detected.")
        L.append("\n## 7. Recommendations\n")
        recs = [f"- {r.get('feature_id', '*')}: {r.get('process', '')} - {r.get('justification', '')} ({r.get('source', '')})" for r in kr.get("recommendations", [])]
        recs += [f"- {v.get('recommendation')}" for v in kr.get("violations", []) if v.get("recommendation")]
        recs += [f"- Alternative: {a.get('description')}" for a in ps.get("alternatives", [])]
        if ts.get("notes"):
            recs.append(f"- Tooling note: {ts['notes']}")
        if ps.get("notes"):
            recs.append(f"- Sequencing note: {ps['notes']}")
        L.append("\n".join(recs[:30]) if recs else "None.")

        u = session.get("usage", {})
        L.append(f"\n---\nTokens: {u.get('total_tokens')} in {u.get('calls')} LLM calls; wall time {session.get('seconds')} s.")
        L.append("\nTimings (s): `" + json.dumps(session.get("timings", {})) + "`")
        return "\n".join(L) + "\n"

    @staticmethod
    def _plan(L: list, ps: dict, ts: dict) -> None:
        v = ps["validation"]
        st = v.get("stats", {})
        L.append(f"Source: **{ps['source']}**; LLM refinement: {'yes' if ps.get('llm_refined') else 'no'}; validation: {len(v['critical'])} critical, "
                 f"{len(v['warnings'])} warnings; tool changes {st.get('tool_changes', '-')} (avoidable {st.get('avoidable_tool_changes', '-')}).\n")
        for mv in ps.get("repair_moves") or []:
            L.append(f"- constraint repair: {mv}")
        if ps.get("tool_regrouping"):
            L.append(f"- {ps['tool_regrouping']}")
        L.append("\n| step | process | features | rationale |\n|---|---|---|---|")
        for s in cam_steps(ps["sequence"]):
            L.append(f"| {s['step']} | {s['process']} | {', '.join(s.get('feature_ids', []))} | {s.get('rationale', '')} |")

    @staticmethod
    def _tools(L: list, ts: dict) -> None:
        L.append("\n| step | process | tool | D | Vc m/min | n rpm | fz / f mm | vf mm/min | ap | ae | P kW | coolant | notes |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in ts["assignments"]:
            if a.get("process") in NON_CAM_PROCESSES or not a.get("requires_tool"):
                continue
            L.append(f"| {a['step']} | {a['process']} | {a.get('tool_id') or '**UNASSIGNED**'} | {_v(a.get('tool_diameter_mm'))} | {_v(a.get('cutting_speed_m_min'))} | "
                     f"{_v(a.get('spindle_rpm'))} | {_v(a.get('feed_per_tooth_mm') or a.get('feed_per_rev_mm'), 3)} | {_v(a.get('feed_mm_min'))} | {_v(a.get('doc_mm'))} | "
                     f"{_v(a.get('woc_mm'))} | {_v(a.get('power_kw'))} | {a.get('coolant_mode', '')} | {'; '.join(a.get('notes') or [])[:120]} |")
        if ts.get("llm_changes"):
            L.append("\nLLM changes to the draft: " + "; ".join(f"step {c['step']} {c['change']} -> {c.get('value')} ({'accepted' if c.get('accepted') else 'rejected'})"
                                                          for c in ts["llm_changes"][:12]))
        L.append("\nTool list: " + ", ".join(f"T{t['tool_number']} {t['tool_id']} D{t.get('diameter_mm')}" for t in ts["tooling_list"]))

    @staticmethod
    def _cam(L: list, cam: dict, tol) -> None:
        v = cam.get("verification", {})
        ct = v.get("cycle_time") or {}
        L.append(f"- Job `{cam.get('job_file')}`, G-code `{cam.get('gcode_file')}` ({cam.get('gcode_lines')} lines, post `{cam.get('post')}`)")
        L.append(f"- Cycle time estimate: {ct.get('total_min')} min (cutting {ct.get('cutting_min')}, rapids {ct.get('rapid_min')}, tool changes {ct.get('tool_change_min')})")
        pf = cam.get("postfix") or {}
        if pf.get("changes"):
            L.append(f"- Cycle fixes: {pf['changes']} line(s) rewritten (tapping {pf.get('tap_cycle')}, reaming/boring G85)")
        st = (v.get("deviation") or {}).get("stats") or {}
        cavity_obs = (v.get("deviation") or {}).get("cavity_region_observations")
        if cavity_obs:
            L.append(f"- Cavity/region observations: {len(cavity_obs['associations'])} projected XY bbox overlaps; "
                     f"{len(cavity_obs['unbound'])} unbound observations. Z unconstrained; "
                     "no path contact, reach, causal owner or repair authorization.")
        exact_obs = (v.get("deviation") or {}).get("exact_observations")
        if exact_obs:
            L.append(f"- Exact probe observations: owner unknown; needs review {exact_obs['needs_review']}; artifact `{exact_obs['artifact']}`. "
                     "No causal operation attribution or authorization for destructive repair.")
            for kind, item in exact_obs["kinds"].items():
                L.append(f"  - {kind}: {item['count']} probes / {item['area_mm2']:.3f} mm2, {item['cluster_count']} clusters; "
                         f"normal probe integral {item['probe_integral_mm3']:.3f} mm3 (not removed-body volume); "
                         f"uncovered unexpected {item['uncovered_unexpected']['count']} probes / {item['uncovered_unexpected']['area_mm2']:.3f} mm2")
            for limit in exact_obs["limits"]:
                L.append(f"  - observation limit: {limit}")
        if st:
            scope = " (later setups assumed ideal)" if st.get("later_setups") == "ideal" else ""
            L.append(f"- Simulation vs target (exact swept volumes, {st.get('samples')} points on the B-rep faces){scope}: **{st.get('surface_within_tol_pct')} %** of the surface "
                     f"within +-{tol} mm ({st.get('surface_within_tol_pct_excl_expected')} % without drill-point cones; another {st.get('band_pct')} % out by less than "
                     f"{st.get('band_factor')}x the tolerance); material left {st.get('excess_cm3')} cm3 "
                     f"(max {st.get('max_excess_mm')} mm), overcut {st.get('gouge_cm3')} cm3 (max {st.get('max_gouge_mm')} mm)")
            for reg in (v.get("deviation") or {}).get("regions", [])[:6]:
                L.append(f"  - [{reg['klass']}] {reg['kind']} {reg['max_mm']} mm / {reg['volume_mm3']} mm3 on {reg['feature']} at {reg['at']}" + (f": {reg['note']}" if reg.get("note") else ""))
        for rd in cam.get("rounds", []):
            for fx in rd.get("fixes") or []:
                L.append(f"  - repair round {rd['round']} ({rd.get('fix_source')}): step {fx.get('step')} {fx.get('action')} {fx.get('value')} ({fx.get('reason')})")
        for i in v.get("issues", [])[:15]:
            L.append(f"  - issue: {i}")
        for w in v.get("warnings", [])[:10]:
            L.append(f"  - warning: {w}")
        for s in cam.get("skipped", []):
            L.append(f"  - skipped step {s['step']} {s['process']}: {s['reason']}")
