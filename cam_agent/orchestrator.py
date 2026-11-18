"""Tier-1 Orchestrator: session management, dispatch, HITL hooks, aggregation (no domain reasoning here).

Pipeline per setup: feature extraction -> context fusion -> knowledge retrieval -> process sequence -> tool selection
-> tool regrouping -> CAM generation + verification -> report. Multi-setup (indexed) jobs share fusion and knowledge
retrieval and are verified together on one workpiece model.

The stock is one object for every stage: the default box (part bounding box + margins in the primary setup frame) or a
user blank (STEP in the part's coordinate system). Its allowances decide whether facing and outer-profile operations
are needed, where roughing starts and what the simulation starts from.
"""
from __future__ import annotations

import json
import shutil
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

from .agents.cam_generation import CAMGenerationAgent, cutter_reach, reach_profile, tooling_limits
from .agents.context_fusion import ContextFusionAgent
from .agents.feature_extraction import FeatureExtractionAgent, describe_features
from .agents.knowledge_retrieval import KnowledgeRetrievalAgent
from .agents.process_sequence import ProcessSequenceAgent
from .agents.report_generation import ReportGenerationAgent
from .agents.tool_selection import ToolSelectionAgent
from .config import SETTINGS, Settings
from .knowledge import get_store
from .llm import LLM, Trace
from .planning import sequencing as SEQ
from .services import FreeCADRunner
from .services.voxel import Frame

HITL = Callable[[str, dict], dict]  # (stage, payload) -> possibly modified payload


class Orchestrator:
    def __init__(self, settings: Settings = SETTINGS, hitl: HITL | None = None, log: Callable[[str], None] = print) -> None:
        self.s = settings
        self.trace = Trace()
        self.llm = LLM(settings, self.trace)
        self.freecad = FreeCADRunner(settings.freecadcmd, settings.freecad_timeout)
        self.hitl, self.log = hitl, log
        self.store = get_store()

    # ======================================================================================== stock
    def _tess(self, path, workdir, **kw) -> dict:
        return self.freecad.run("tessellate.py", {"path": str(path), "deflection": kw.pop("deflection", max(0.01, self.s.tol_mm / 4)), **kw}, workdir)

    def _user_stock(self, session: dict, stock: str | Path, cad_path: Path, session_dir: Path) -> dict:
        from .services.stock import fit_check, save_mesh
        src = Path(stock).expanduser().resolve()
        if not src.exists():
            raise FileNotFoundError(f"stock file not found: {src}")
        dst = session_dir / "stock" / src.name  # not in the session root: tools take the first STEP found there as the part
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        part = self._tess(cad_path, session_dir / "fc", raw=True, deflection=0.05)
        blank = self._tess(dst, session_dir / "fc", raw=True, deflection=0.05)
        fit = fit_check(part, blank)
        info = {"kind": "user", "source": str(src), "solid": str(dst), "bbox": [round(v, 3) for v in blank["bbox"]],
                "mesh_raw": save_mesh(dst.parent / "stock_raw_mesh.npz", blank), **fit}
        self.log(f"[stock] user blank {src.name}: {fit['stock_cm3']} cm3, part {fit['part_cm3']} cm3, to remove {fit['removal_cm3']} cm3, "
                 f"part outside the blank {fit['part_outside_stock_pct']} %")
        limit = float(__import__("os").environ.get("CAM_STOCK_MAX_OUTSIDE_PCT", "5"))
        if fit["part_outside_stock_pct"] > limit:
            raise ValueError(f"the stock {src.name} does not contain the part: {fit['part_outside_stock_pct']} % of the part volume is outside it "
                             f"(limit {limit} %, CAM_STOCK_MAX_OUTSIDE_PCT). The stock STEP must be in the part's coordinate system.")
        if fit["part_outside_stock_pct"] > 0.5:
            self.log(f"[stock] WARNING: {fit['part_outside_stock_pct']} % of the part is outside the blank; that material stays as it is")
        session["stock"] = info
        return info

    def _default_stock(self, session: dict, part_raw: dict, primary: Frame, session_dir: Path) -> dict:
        from .services.stock import default_stock, save_mesh
        mesh = default_stock(part_raw, primary, self.s.stock_margins)
        info = {"kind": "box", "margins": self.s.stock_margins, "frame_up": primary.up, "bbox": [round(v, 3) for v in mesh["bbox"]],
                "mesh_raw": save_mesh(session_dir / "stock" / "stock_raw_mesh.npz", mesh)}
        session["stock"] = info
        return info

    @staticmethod
    def _stock_mesh(info: dict) -> dict:
        from .services.stock import load_mesh
        return load_mesh(info["mesh_raw"])

    def _apply_allowances(self, features: list[dict], part_bbox_k, stock_k: dict) -> dict:
        """Facing only where the stock stands above the top face; outer contour only where there is side allowance."""
        from .services.stock import allowances
        a = allowances(stock_k, part_bbox_k)
        for f in features:
            if f["type"] == "planar_face" and f.get("role") == "top":
                f["needs_facing"] = a["top"] > 0.05
            if f["type"] == "outer_profile":
                f["needs_profile"] = a["side"] > 0.05
        return a

    # ======================================================================================== orientation
    def orient(self, cad_path: Path, session_dir: Path, up: str, stock_info: dict | None) -> tuple[Path, dict]:
        from .services.access import accessibility
        info: dict = {"requested": up}
        tri = self.freecad.run("triage.py", {"path": str(cad_path)}, session_dir / "fc")
        info.update({"solids": tri["solids"], "size": tri["size"], "area_share": tri["area_share"]})
        if tri["solids"] == 0:
            raise ValueError("the file contains no solid body (surfaces only): nothing to machine")
        acc = accessibility(tri["vertices"], tri["tri"], tri["volume"])
        info["accessible_by_dir"] = acc
        chosen = max(acc, key=acc.get) if up == "auto" else up
        if up == "auto" and acc.get("+Z", 0) >= acc[chosen] - 0.02:
            chosen = "+Z"
        if os.environ.get("CAM_SIDE_SWITCH", "1") == "1" and acc.get(chosen, 0) < 0.6:
            better = max(acc, key=acc.get)
            if acc.get(better, 0) >= 0.6 and better != chosen:
                info["switched_from"] = chosen
                chosen = better
        info.update({"up": chosen, "accessible": acc[chosen]})
        if chosen == "+Z" and tri["solids"] == 1:
            return cad_path, info
        out = session_dir / f"{cad_path.stem}_oriented.step"
        req = {"path": str(cad_path), "up": chosen, "out": str(out)}
        if stock_info and stock_info.get("kind") == "user":
            src = Path(stock_info["solid"])
            req.update(stock=str(src), stock_out=str(src.parent / f"{src.stem}_oriented.step"))
        r = self.freecad.run("reorient.py", req, session_dir / "fc")
        if stock_info and stock_info.get("kind") == "user":
            from .services.stock import save_mesh
            stock_info["solid"] = r["stock_out"]
            blank = self._tess(r["stock_out"], session_dir / "fc", raw=True, deflection=0.05)
            stock_info["mesh_raw"] = save_mesh(Path(r["stock_out"]).parent / "stock_oriented_mesh.npz", blank)
            info["stock"] = r["stock_out"]
        return out, info

    # ======================================================================================== run
    def run(self, cad_path: str | Path, material: str | None = None, notes: str = "", part_name: str | None = None, skip_cam: bool = False,
            detail: str = "standard", up: str = "+Z", setups: str = "1", stock: str | Path | None = None) -> dict:
        t0 = time.time()
        cad_path = Path(cad_path).resolve()
        part_name = part_name or cad_path.stem
        session_dir = self.s.out_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{part_name}"
        session_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(cad_path, session_dir / cad_path.name)
        session: dict = {"part_name": part_name, "cad_path": str(cad_path), "model": self.s.model, "mode": self.s.mode, "sequencing": self.s.sequencing,
                         "tol_mm": self.s.tol_mm, "session_dir": str(session_dir), "agents": {}, "timings": {}}
        self.log(f"[orchestrator] session {session_dir}")
        stock_info = self._user_stock(session, stock, cad_path, session_dir) if stock else None
        if setups != "1":
            from .agents.setup_planning import SetupPlanningAgent
            t = time.time()
            sm = self._stock_mesh(stock_info) if stock_info else None
            groups, info = FeatureExtractionAgent(self.llm, self.freecad, self.s.recognizer, self.log).learned_groups(str(cad_path), session_dir / "fc")
            if info:
                session["recogniser_info"] = info
            fit = os.environ.get("CAM_REACH_FIT", "1") == "1"
            plan = SetupPlanningAgent(self.freecad, max_setups=6 if setups == "auto" else int(setups), stock_mesh=sm,
                                      reach=reach_profile(self.store, None, self.s.holder_d),
                                      cutters=cutter_reach(self.store) if fit else None, holder_d=self.s.holder_d).run(
                cad_path, session_dir / "fc", primary=up if up in ("-Z", "+X", "-X", "+Y", "-Y") else "auto", log=self.log, groups=groups)
            session["timings"]["setup_planning"] = round(time.time() - t, 1)
            session["setup_plan"] = {k: v for k, v in plan.items() if k != "setups"}
            if len(plan["setups"]) > 1:
                return self._run_multi(session, plan, cad_path, material, notes, skip_cam, detail, t0, stock_info)
            up = plan["primary"]
            self.log(f"[setup_planning] one setup from {up}: single-setup pipeline")
        if up != "+Z":
            cad_path, session["orientation"] = self.orient(cad_path, session_dir, up, stock_info)
            o = session["orientation"]
            self.log(f"[orientation] side {o['up']} (reachable share {o['accessible']:.0%}; by side {o['accessible_by_dir']})")
            if o["accessible"] < 0.75:
                self.log("[orientation] WARNING: large undercuts from every side: not a single-setup 3-axis part")

        # 1. features (3D + notes, concurrently)
        t = time.time()
        extracted = FeatureExtractionAgent(self.llm, self.freecad, self.s.recognizer, self.log).run(cad_path, notes, session_dir / "fc")
        session["timings"]["feature_extraction"] = round(time.time() - t, 1)
        cad = extracted["cad"]
        frame = Frame("+Z", cad["translation"])
        part_raw = self._tess(cad_path, session_dir / "fc", raw=True, deflection=0.2)  # only its bounding box is used here
        if stock_info is None:
            stock_info = self._default_stock(session, part_raw, frame, session_dir)
        from .services.stock import transform
        stock_k = transform(self._stock_mesh(stock_info), frame)
        session["stock"]["allowance"] = self._apply_allowances(cad["features"], [0, 0, 0, *cad["part"]["size"]], stock_k)
        self.log(f"[feature_extraction] {cad['summary']} ({cad.get('recogniser', 'rules')}) | annotations: {len(extracted['drawing'].get('annotations', []))} | "
                 f"stock allowance {session['stock']['allowance']}")
        cav = self._detect_cavity(cad_path, cad["features"], cad["part"]["size"], session_dir / "fc", stock_info=stock_info)
        if cav:
            cav["id"] = "F%02d" % (len(cad["features"]) + 1)
            cad["features"].append(cav)
            self.log(f"[feature_extraction] cavity: {cav['residual_mm3'] / 1000:.1f} cm3 ({cav['residual_share']:.0%} of the removal) not explained by "
                     f"features -> height-map roughing D{cav['rough_tool_d']:g} / finishing D{cav['finish_tool_d']:g}")
        extracted = self._hitl("features", extracted)
        session["extracted"] = extracted

        # 2-5. fusion, knowledge, sequence, tools
        fused, kr = self._fusion_and_knowledge(session, extracted, material, session_dir)
        fused["stock_allowance"] = session["stock"]["allowance"]
        ps_out, ts_out = self._plan_setup(session, fused, kr.output, setup=1)
        self._dump(session_dir / "plan.json", {"kr": kr.output, "ps": ps_out, "ts": ts_out})

        # 6. CAM
        acc = (session.get("orientation") or {}).get("accessible")
        if acc is not None and acc < 0.6 and not skip_cam:
            self.log(f"[cam_generation] skipped: only {acc:.0%} of the part is reachable from the best side (multi-setup, 5-axis or a casting)")
            session["cam"] = {"available": False, "skipped": True, "reason": f"reachable share {acc:.2f} < 0.60"}
            skip_cam = True
        if not skip_cam:
            t = time.time()
            self.log("[cam_generation] FreeCAD job, toolpaths, G-code, verification")
            agent = CAMGenerationAgent(self.llm, self.freecad, self.s.post_processor, mode=self.s.mode, settings=self.s)
            cam = agent.run(str(cad_path), session_dir, fused, ps_out, ts_out, stock_info, frame)
            session["cam"] = cam
            session["timings"]["cam_generation"] = round(time.time() - t, 1)
            v = cam.get("verification", {})
            st = (v.get("deviation") or {}).get("stats") or {}
            self.log(f"[cam_generation] ok={v.get('ok')} ops={len(cam.get('operations') or [])} surface within +-{self.s.tol_mm} mm: {st.get('surface_within_tol_pct')} % "
                     f"(excl. drill points {st.get('surface_within_tol_pct_excl_expected')} %) cycle {((v.get('cycle_time') or {}).get('total_min'))} min issues={len(v.get('issues', []))}")
            for rd in cam.get("rounds", []):
                self.log(f"[cam_generation] round {rd['round']}: ok={rd['ok']} within={rd.get('deviation', {}).get('surface_within_tol_pct')} "
                         f"fixes({rd.get('fix_source', '-')})={[(f.get('step'), f.get('action'), f.get('value')) for f in rd.get('fixes') or []]}")
        else:
            session.setdefault("cam", {"available": False, "skipped": True})
        return self._finish(session, session_dir, detail, t0)

    def _worse_with(self, verifier, k: int, margin: float = 0.5):
        """(without, with) exact within % of the built setups when setup k makes them worse by more than `margin`, else None."""
        if verifier is None or k not in verifier.sweeps:
            return None
        try:
            from .services import exact
            from .services.verify import surface_metric
            built = [kk for kk in sorted(verifier.sweeps)]
            def score(ids):
                wp = exact.blank_solid(verifier.stock_raw, [verifier.sweeps[i] for i in ids])
                return float(surface_metric(verifier.brep, wp.inside, verifier.tol)["stats"]["surface_within_tol_pct"])
            with_k = score(built)
            without_k = score([i for i in built if i != k])
        except Exception as e:  # noqa: BLE001  (a check must not break the pipeline)
            self.log(f"[setup {k}] setup comparison failed: {type(e).__name__}: {e}")
            return None
        return (round(without_k, 2), round(with_k, 2)) if without_k > with_k + margin else None

    # ======================================================================================== shared stages
    def _detect_cavity(self, cad_path, features, size, workdir, up="+Z", translation=None, primary=True, stock_surface=None, stock_info=None, iso=None):
        from .services.zlevel import detect_cavity, cavity_tool_reaches
        ds = sorted({float(t["diameter_mm"]) for t in self.store.tool_library["tools"] if t["category"] == "endmill"}, reverse=True)
        reaches = cavity_tool_reaches(self.store.tool_library["tools"], ds, iso) if os.environ.get("CAM_CAVITY_TOOL_REACH", "0") == "1" else None
        mesh = self._tess(cad_path, workdir, deflection=0.08, up=up, translation=translation)
        feats = [f for f in features if f.get("type") != "hole" or f.get("vertical", True)]
        smesh = None
        if stock_info and stock_info.get("kind") == "user":
            smesh = self.freecad.run("tessellate.py", {"path": stock_info["solid"], "deflection": 0.08, "up": up, "translation": translation,
                                                       "frame_of": str(cad_path)}, workdir)
        return detect_cavity(mesh["vertices"], mesh["faces"], size, feats, ds, primary=primary, stock_surface=stock_surface,
                             min_mm3=1000.0 if primary else 80.0, min_share=0.05 if primary else 0.15, stock_mesh=smesh, margins=self.s.stock_margins,
                             drill_tools=self.store.tool_library["tools"], tool_reaches=reaches)

    def _fusion_and_knowledge(self, session: dict, extracted: dict, material, session_dir: Path):
        t = time.time()
        # material: --material > drawing notes > default; general tolerance: drawing notes > default (the source is kept,
        # so that the knowledge agents and the report know a value was assumed)
        source = "user" if material else ("drawing" if (extracted.get("drawing") or {}).get("material") else "default")
        fused = ContextFusionAgent(self.llm).run(extracted, material or (None if source == "drawing" else self.s.material))
        fused["material"]["source"] = source
        fused["general_tolerance_source"] = "drawing" if fused.get("general_tolerance") else "default"
        fused["general_tolerance"] = fused.get("general_tolerance") or self.s.general_tolerance
        fused = self._hitl("fusion", fused)
        session["fused"] = fused
        self._dump(session_dir / "features.json", fused)
        self.log(f"[context_fusion] material={fused['material']['id']} ({fused['material']['source']}) general tolerance={fused['general_tolerance']} "
                 f"({fused['general_tolerance_source']}) mappings={len(fused['mappings'])} review={len(fused['needs_review'])}")
        if not fused["material"]["id"]:
            self.log(f"[context_fusion] WARNING: material {fused['material']['input']!r} is not in the database: C45 cutting data assumed; pass --material")
        session["timings"]["context_fusion"] = round(time.time() - t, 1)
        t = time.time()
        kr = KnowledgeRetrievalAgent(self.llm, self.s.mode, self.s.max_iterations).run(fused)
        session["agents"]["knowledge_retrieval"] = kr.as_dict()
        session["kr"] = kr.output
        session["timings"]["knowledge_retrieval"] = round(time.time() - t, 1)
        self.log(f"[knowledge_retrieval] mode={self.s.mode} rules={len(kr.output['rules'])} violations={len(kr.output['violations'])} tokens={kr.tokens} "
                 f"({session['timings']['knowledge_retrieval']} s)")
        return fused, kr

    def _plan_setup(self, session: dict, fused: dict, kr_output: dict, setup: int, drop: set | None = None, key: str = "") -> tuple[dict, dict]:
        t = time.time()
        ps = ProcessSequenceAgent(self.llm, self.s.mode, self.s.max_iterations, self.s.sequencing).run(fused, kr_output, setup=setup)
        ps_out = ps.output
        if drop:
            ps_out["sequence"] = SEQ.number([st for st in ps_out["sequence"] if st["process"] not in drop], setup)
        ps_out = self._hitl("sequence", ps_out)
        tp = time.time()
        ts = ToolSelectionAgent(self.llm, self.s.mode, self.s.max_iterations).run(fused, ps_out, kr_output)
        ts_out = ts.output
        if ps_out.get("source") != "user_override":
            seq, asg, saved = SEQ.regroup_by_tool(ps_out["sequence"], ts_out["assignments"], fused["features"])
            if saved:
                ps_out["sequence"], ts_out["assignments"] = seq, asg
                from .agents.tool_selection import tooling_list
                ts_out["tooling_list"] = tooling_list(asg)
                ps_out["tool_regrouping"] = f"{saved} tool change(s) saved by grouping same-tool steps within a phase"
        final_setup = not (drop and drop & set(SEQ.FINAL))  # intermediate setups leave the final steps to the last one
        ps_out["validation"] = SEQ.validate(ps_out["sequence"], fused["features"], fused["material"]["group"], fused, self.s.sequencing,
                                            ts_out["assignments"], final_setup=final_setup)
        agents = session["agents"] if not key else session.setdefault("setup_agents", {}).setdefault(key, {})
        agents["process_sequence"], agents["tool_selection"] = ps.as_dict(), ts.as_dict()
        if not key:
            session["ps"], session["ts"] = ps_out, ts_out
        timings = session["timings"].setdefault(f"setup{setup}", {})
        timings.update(process_sequence=round(tp - t, 1), tool_selection=round(time.time() - tp, 1))
        v = ps_out["validation"]
        self.log(f"[setup {setup}] sequence: {' -> '.join(s['process'] for s in ps_out['sequence'])}")
        self.log(f"[setup {setup}] source={ps_out['source']} llm={ps_out.get('llm_refined')} critical={len(v['critical'])} warnings={len(v['warnings'])} "
                 f"tooled {ts_out['n_tooled']}/{ts_out['n_required']} tool changes={v['stats'].get('tool_changes')} "
                 f"(PS {timings['process_sequence']} s, TS {timings['tool_selection']} s)")
        return ps_out, ts_out

    def _finish(self, session: dict, session_dir: Path, detail: str, t0: float) -> dict:
        session["usage"] = self.llm.usage.as_dict()
        session["seconds"] = round(time.time() - t0, 1)
        report = ReportGenerationAgent(self.llm, detail).run(session)
        (session_dir / "report.md").write_text(report, encoding="utf-8")
        session["usage"] = self.llm.usage.as_dict()
        session["report_path"] = str(session_dir / "report.md")
        self.trace.dump(session_dir / "trace.jsonl")
        if any(e.get("budget") for e in session.get("setups", [])):
            session["partial"] = True
        else:
            session.pop("partial", None)
        self._dump(session_dir / "session.json.tmp", session)
        (session_dir / "session.json.tmp").replace(session_dir / "session.json")
        self.log(f"[orchestrator] done in {session['seconds']} s, {session['usage']['total_tokens']} tokens -> {session_dir / 'report.md'}")
        return session

    # ======================================================================================== multi-setup
    def _run_multi(self, session: dict, plan: dict, cad_path: Path, material, notes: str, skip_cam: bool, detail: str, t0: float, stock_info) -> dict:
        from .services.multi_verify import MultiSetupVerifier
        from .services.stock import transform
        session_dir = Path(session["session_dir"])
        ups = [s["up"] for s in plan["setups"]]
        session["orientation"] = {"up": ups[0], "setups": ups}
        frames = {s["index"]: Frame(s["up"], s["translation"]) for s in plan["setups"]}
        part_raw = self._tess(cad_path, session_dir / "fc", raw=True)
        if stock_info is None:
            stock_info = self._default_stock(session, part_raw, frames[plan["setups"][0]["index"]], session_dir)
        stock_raw = self._stock_mesh(stock_info)
        sp0 = plan["setups"][0]
        cav = self._detect_cavity(cad_path, sp0["features"], sp0["part"]["size"], session_dir / "fc", up=sp0["up"], translation=sp0["translation"],
                                  primary=True, stock_info=stock_info)
        if cav:
            cav.update(id=f"S{sp0['index']}C01", setup=sp0["index"])
            sp0["features"].append(cav)
        all_feats = [f for s in plan["setups"] for f in s["features"]]
        drawing = FeatureExtractionAgent(self.llm, self.freecad, self.s.recognizer, self.log).extract_2d(notes)
        extracted = {"cad": {"features": all_feats, "part": sp0["part"], "hole_patterns": [],
                             "summary": {t: sum(1 for f in all_feats if f["type"] == t) for t in ("hole", "pocket", "slot", "step", "chamfer", "fillet")}}, "drawing": drawing}
        session["extracted"] = extracted
        fused, kr = self._fusion_and_knowledge(session, extracted, material, session_dir)
        stuck = plan.get("unreachable_vs_part")
        if stuck is not None and stuck > 0.15 and not skip_cam:
            reason = f"Stock volume unreachable from the selected setups is {stuck:.0%} of the finished part volume"
            self.log(f"[cam_generation] skipped: {reason}")
            session["cam"] = {"available": False, "skipped": True, "reason": reason}
            skip_cam = True
        verifier = None
        if not skip_cam:
            verifier = MultiSetupVerifier(part_raw, plan["setups"], stock_raw, self.s.tol_mm, self.s.sim_max_cells,
                                          tooling_limits(self.store, (fused.get("material_data") or {}).get("iso_group")), self.s.surface_samples,
                                          part_path=cad_path, freecad=self.freecad, workdir=session_dir / "fc")
        session["setups"] = []
        last = plan["setups"][-1]["index"]
        budget = float(os.environ.get("CAM_WALL_BUDGET_S", "0") or 0)
        for sp in plan["setups"]:
            k, side, fr = sp["index"], sp["up"], frames[sp["index"]]
            if budget and k > 1 and time.time() - t0 > budget:
                # past the time budget: deliver the setups built so far rather than be killed with no session (evolution A12)
                self.log(f"[setup {k} {side}] time budget {budget:.0f} s spent: setup dropped")
                session["setups"].append({"index": k, "up": side, "translation": sp["translation"], "features": [], "dropped": True, "budget": True,
                                          "ps": {"sequence": []}, "ts": {"assignments": [], "tooling_list": [], "n_tooled": 0, "n_required": 0}})
                continue
            fused_k = {**fused, "features": [dict(f) for f in fused["features"] if f.get("setup") == k], "part": sp["part"]}
            stock_k = transform(stock_raw, fr)
            fused_k["stock_allowance"] = self._apply_allowances(fused_k["features"], [0, 0, 0, *sp["part"]["size"]], stock_k)
            allowance_val = float(sp.get("allowance_mm") or 0.0)
            if k > 1 and verifier is not None and 0.0 < allowance_val <= 0.5:
                self.log(f"[setup {k} {side}] thin allowance skim {allowance_val} mm: facing only, no cavity on this side")
            elif k > 1 and verifier is not None:
                sp["stock_surface"] = verifier.stock_surface(k, session_dir / "fc" / f"stock_actual_setup{k}.npy")
                cav = self._detect_cavity(cad_path, fused_k["features"], sp["part"]["size"], session_dir / "fc", up=side, translation=sp["translation"],
                                          primary=False, stock_surface=sp["stock_surface"], stock_info=stock_info,
                                          iso=(fused.get("material_data") or {}).get("iso_group", "P"))
                if cav:
                    cav.update(id=f"S{k}C01", setup=k, subtype="cavity", material=fused["material"].get("id"))
                    fused_k["features"].append(cav)
                    self.log(f"[setup {k} {side}] cavity: {cav['residual_mm3'] / 1000:.1f} cm3 still standing on this side")
                if not any(f["type"] in ("hole", "pocket", "slot", "step", "cavity") for f in fused_k["features"]) and not any(
                        f.get("needs_facing") for f in fused_k["features"]):
                    self.log(f"[setup {k} {side}] nothing left to machine from this side: setup dropped")
                    session["setups"].append({"index": k, "up": side, "translation": sp["translation"], "features": [], "dropped": True,
                                              "ps": {"sequence": []}, "ts": {"assignments": [], "tooling_list": [], "n_tooled": 0, "n_required": 0}})
                    continue
            kr_k = kr.output
            if k > 1:  # drawing-level process notes belong to the primary setup
                fused_k.update(process_annotations=[], user_sequence_override=[])
                kr_k = {**kr.output, "recommendations": []}
            drop = set()
            if k != last:
                drop |= {"cleaning", "inspection", "coating", "deburring"}
            ps_out, ts_out = self._plan_setup(session, fused_k, kr_k, setup=k, drop=drop, key=f"setup{k}")
            entry = {"index": k, "up": side, "translation": sp["translation"], "features": fused_k["features"], "ps": ps_out, "ts": ts_out,
                     "stock_allowance": fused_k["stock_allowance"]}
            if not skip_cam:
                t = time.time()
                out_k = session_dir / f"setup{k}_{side.replace('+', 'p').replace('-', 'm')}"
                agent = CAMGenerationAgent(self.llm, self.freecad, self.s.post_processor, mode=self.s.mode, settings=self.s)
                req = agent.build_request(str(cad_path), out_k, fused_k, ps_out, ts_out, stock_info, side, sp["translation"])
                req["stock_surface"] = sp.get("stock_surface")
                if k > 1:
                    req["stock"].update(xy_margin=0.0, z_top=0.0, z_bottom=0.0)
                agent.deviation_fn = lambda result, rq, k=k, fk=fused_k: verifier.check_setup(k, result["gcode_file"], rq, fk["features"])
                cam = agent.run_request(req, out_k, fused_k, frame=fr)
                entry["cam"] = cam
                session["timings"].setdefault(f"setup{k}", {})["cam_generation"] = round(time.time() - t, 1)
                v = cam.get("verification", {})
                st = (v.get("deviation") or {}).get("stats") or {}
                self.log(f"[setup {k} {side}] CAM ok={v.get('ok')} ops={len(cam.get('operations') or [])} within={st.get('surface_within_tol_pct')} % "
                         f"issues={len(v.get('issues', []))}")
                worse = self._worse_with(verifier, k) if (k > 1 and os.environ.get("CAM_DROP_WORSE_SETUP", "1") == "1") else None
                # the exact metric of the setups built so far, with and without this one: a setup that leaves the part
                # worse than not machining that side at all is not delivered
                if worse is not None:
                    prev_within, now = worse
                    self.log(f"[setup {k} {side}] built setups within {now} % with it, {prev_within} % without it: setup dropped")
                    verifier.sweeps.pop(k, None)
                    verifier.programs.pop(k, None)
                    verifier.results.pop(k, None)
                    verifier.states[k] = type(verifier.states[k])(fr, ideal=verifier.states[k].ideal)
                    session["setups"].append({"index": k, "up": side, "translation": sp["translation"], "features": [], "dropped": True,
                                              "worse": {"before": prev_within, "after": now},
                                              "ps": {"sequence": []}, "ts": {"assignments": [], "tooling_list": [], "n_tooled": 0, "n_required": 0}})
                    continue
            session["setups"].append(entry)
            if budget:
                self._checkpoint_multi(session, session_dir, verifier, t0)
        if os.environ.get("CAM_TWO_SIDED_DRILL", "0") == "1":
            from .services.two_sided_drill import deliver_suffix
            deliver_suffix(self, session, plan, cad_path, fused, stock_info, stock_raw, verifier)
        first = session["setups"][0]
        session["ps"], session["ts"], session["cam"] = first["ps"], first["ts"], first.get("cam", {"available": False, "skipped": True})
        if verifier is not None:
            session["multi_setup"] = {"final": verifier.final(),
                                      "all_ok": not any(e.get("budget") for e in session["setups"]) and
                                      all((e.get("cam", {}).get("verification") or {}).get("ok") for e in session["setups"] if not e.get("dropped"))}
            verifier.save_surface(session_dir / "surface_final.npz")
            self.log(f"[verification] all setups: {session['multi_setup']['final']}")
        self._dump(session_dir / "plan.json", {"kr": kr.output, "setups": [{"index": e["index"], "up": e["up"], "ps": e["ps"], "ts": e["ts"]} for e in session["setups"]]})
        return self._finish(session, session_dir, detail, t0)

    def _checkpoint_multi(self, session: dict, session_dir: Path, verifier, t0: float) -> None:
        """Keep completed setups recoverable if the next setup is interrupted (A12)."""
        first = session["setups"][0]
        session.update(ps=first["ps"], ts=first["ts"], cam=first.get("cam", {"available": False, "skipped": True}),
                       partial=True, seconds=round(time.time() - t0, 1), usage=self.llm.usage.as_dict())
        if verifier is not None:
            # final() evaluates only built setups, unlike the repair loop's optimistic check_setup().
            session["multi_setup"] = {"final": verifier.final(), "all_ok": False}
        self._dump(session_dir / "session.json.tmp", session)
        (session_dir / "session.json.tmp").replace(session_dir / "session.json")

    # ======================================================================================== misc
    def _hitl(self, stage: str, payload: dict) -> dict:
        if self.hitl is None:
            return payload
        out = self.hitl(stage, payload)
        self.trace.log("hitl", stage=stage, changed=out is not payload)
        return out or payload

    @staticmethod
    def _dump(path: Path, obj) -> None:
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist() if o.size <= 64 else f"<array {o.shape}>"
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)


def interactive_hitl(stage: str, payload: dict) -> dict:
    """Minimal terminal HITL: shows the stage output and accepts JSON corrections (or Enter to accept)."""
    if stage == "features":
        print("\n=== HITL: recognised features ===")
        print(describe_features(payload["cad"]["features"]))
        print("annotations:", json.dumps(payload["drawing"].get("annotations", []), ensure_ascii=False)[:2000])
    elif stage == "fusion":
        print("\n=== HITL: material / mappings ===")
        print("material:", payload["material"])
        for m in payload["mappings"]:
            print(m)
        print('Enter e.g. {"material": "SS304"} to override the material, or an empty line to accept.')
    elif stage == "sequence":
        print("\n=== HITL: process sequence ===")
        print(" -> ".join(s["process"] for s in payload["sequence"]))
        print('Enter e.g. {"sequence": ["facing", "drilling", ...]} to override, or an empty line to accept.')
    try:
        line = input("> ").strip()
    except EOFError:
        line = ""
    if not line:
        return payload
    try:
        corr = json.loads(line)
    except json.JSONDecodeError:
        print("not JSON, ignored")
        return payload
    if stage == "fusion" and "material" in corr:
        from .normalizer import Normalizer
        st = get_store()
        mid, _ = Normalizer(st.material_aliases(), None, "material").normalize(corr["material"])
        m = st.material(mid)
        if m:
            payload["material"] = {"id": mid, "name": m["name"], "group": m["group"], "input": corr["material"], "normalized_by": "user"}
            payload["material_data"] = m
            for f in payload["features"]:
                f["material"] = mid
    if stage == "sequence" and "sequence" in corr:
        payload["user_override_sequence"] = corr["sequence"]
        payload["sequence"] = [{"step": i + 1, "setup": 1, "process": p, "feature_ids": ["*"], "rationale": "user override"} for i, p in enumerate(corr["sequence"])]
        payload["source"] = "user_override"
    return payload
