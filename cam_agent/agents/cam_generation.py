"""CAM Generation Agent (extension beyond the paper's process planning): validated sequence + tool assignments ->
FreeCAD CAM job -> toolpaths -> post-processed G-code -> cycle fixes (G84 / G85) -> verification -> bounded repair.

Closed loop (max `repair_rounds`):
  1. build the job in FreeCAD (Custom operations for the toolpaths FreeCAD has no op for: height-map cavity roughing and
     finishing, open slots / steps / nested pockets, chamfer faces, helical counterbores, thread milling);
  2. rewrite tapping / reaming / boring cycles (services/postfix.py) and run the safety checks;
  3. measure the G-code with the surface metric on the exact swept-volume workpiece (services/verify.py, exact.py):
     surface share within tolerance; the dexel replay segments the defects into regions attributed to features and
     steps (a region counts only where the metric's points are out of tolerance);
  4. deterministic repair heuristics first; the LLM is asked only when they have nothing to offer (and not in
     deterministic mode); the loop stops when a round does not improve, and the best round is kept.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

from ..config import SETTINGS
from ..knowledge import get_store
from ..llm import LLM
from ..planning.cutting_data import cutting_data, strategy_for
from ..services import FreeCADError, FreeCADRunner
from ..services.voxel import Frame

REPAIR_PROMPT = """You are the programmer of this setup. The program was simulated on the stock. The numbers below are
measurements: material left, overcut, and empty toolpaths, each tied to a feature and to the step that cut there last.
Nothing has been changed for you. Decide the correction yourself and return JSON only:
{"fixes": [{"step": <int>, "action": <"switch_tool"|"set_finish_stock"|"set_depth_below_top"|"set_final_depth"|"set_doc_mm"|"set_woc_mm"|"set_peck_mm"|"drop_operation">,
            "value": <number or a tool_id listed in that step's candidates>, "reason": <short>}], "notes": <str>}
switch_tool can only name a candidate of that step, because those are the tools that fit the opening. An empty path
usually means the tool is too large. Material left after a roughing step that still has finishing stock means the
allowance was never removed. Overcut on a floor or around a hole is a depth that went past the surface. You may change
only the steps the measurements point at, and you may return no fixes if the program should stand."""

FINISHING = ("pocket_finishing", "slot_finishing", "profile_finishing", "surface_finishing")
ROUGH_OF = {"pocket_finishing": "pocket_milling", "slot_finishing": "slot_milling", "profile_finishing": "profile_milling", "surface_finishing": "cavity_roughing"}


NO_REACH = os.environ.get("CAM_NO_REACH", "1") == "1"  # a cutter goes as deep as the opening lets it: no stick-out limit
UNLIMITED = 1e6


def tool_reach(t: dict) -> float:
    """Usable depth below the holder: stick-out minus the length held, or unlimited with CAM_NO_REACH."""
    return UNLIMITED if NO_REACH else float(t.get("overall_length_mm") or 60) - 25.0


def _custom_depth_limit(op: dict) -> float | None:
    """An explicit repair depth still applies when catalog stick-out is disabled."""
    depth = op.get("repair_depth_mm")
    return depth if depth is not None else (None if NO_REACH else op.get("reach_limit_mm"))


def _face_depth_allowed(op: dict, depth: float) -> bool:
    floor = op.get("face_floor")
    return floor is None or os.environ.get("CAM_FACE_STOP", "1") != "1" or depth >= float(floor) - 1e-6


def tooling_limits(store, iso: str | None) -> dict:
    """Smallest endmill and longest reach the library offers for an ISO material group."""
    ems = [t for t in store.tool_library["tools"] if t["category"] == "endmill" and (not iso or iso in t["iso"])]
    ems = ems or [t for t in store.tool_library["tools"] if t["category"] == "endmill"]
    return {"min_d": min(float(t["diameter_mm"]) for t in ems), "max_reach": max(tool_reach(t) for t in ems)}


def reach_profile(store, iso: str | None, holder_d: float = 40.0) -> list[tuple[float, float]]:
    """Pareto set of (holder overhang radius, reach) over the library's milling cutters.

    A cutter of diameter d and overall length L reaches L - 25 mm below its holder (25 mm held in the collet); the
    holder nose (diameter holder_d) overhangs the cutter axis by holder_d/2 - d/2. A tool is useful at a point only if
    the part around it within that overhang is not higher than the reach: a point is reachable if one tool of the set
    satisfies it. Dominated pairs (larger overhang and not longer) are dropped."""
    cut = ("endmill", "bull_endmill", "ball_endmill")
    ts = [t for t in store.tool_library["tools"] if t["category"] in cut and (not iso or iso in t["iso"])]
    ts = ts or [t for t in store.tool_library["tools"] if t["category"] in cut]
    pts = sorted({(round(max(0.0, holder_d / 2 - float(t["diameter_mm"]) / 2), 3), tool_reach(t)) for t in ts},
                 key=lambda p: (p[0], -p[1]))
    out: list[tuple[float, float]] = []
    for rho, reach in pts:
        if not out or reach > out[-1][1]:
            out.append((rho, reach))
    return out


def cutter_reach(store, iso: str | None = None) -> list[tuple[float, float]]:
    """(diameter, reach below the holder) of every milling cutter diameter of the library, the longest of each."""
    cut = ("endmill", "bull_endmill", "ball_endmill")
    ts = [t for t in store.tool_library["tools"] if t["category"] in cut and (not iso or iso in t["iso"])]
    ts = ts or [t for t in store.tool_library["tools"] if t["category"] in cut]
    best: dict[float, float] = {}
    for t in ts:
        d = float(t["diameter_mm"])
        best[d] = max(best.get(d, 0.0), tool_reach(t))
    return sorted(best.items())


def better_round(a: tuple, b: tuple, slack: float = 0.05) -> bool:
    """Is round rank a better than b? Ranks are (hard problems, gouge area beyond the tolerance %, (-within %, issues)).

    Hard problems first (operation errors, G-code safety, plan) - a program with one is not a candidate against one
    without; then the share of the surface within tolerance minus GOUGE_WEIGHT x the gouged share (differences below
    `slack` points are sampling noise); the number of messages only breaks ties."""
    (ha, ga, ka), (hb, gb, kb) = a, b
    # a round that cuts most of the part away is never better than one that does not, whatever its other problems
    if (ga > 50.0) != (gb > 50.0):
        return ga < gb
    if ha != hb:
        return ha < hb
    # A gouge costs five times the surface of material left: 0.06 points of gouge no longer outweigh
    # 20 points of surface brought into tolerance.
    sa, sb = -ka[0] - GOUGE_WEIGHT * ga, -kb[0] - GOUGE_WEIGHT * gb
    if abs(sa - sb) > slack:
        return sa > sb
    return ka < kb


GOUGE_WEIGHT = float(os.environ.get("CAM_GOUGE_WEIGHT", "5"))


class CAMGenerationAgent:
    name = "cam_generation"

    def __init__(self, llm: LLM, freecad: FreeCADRunner | None = None, post: str = "linuxcnc", repair_rounds: int | None = None,
                 mode: str = "parallel", settings=SETTINGS) -> None:
        self.llm, self.freecad, self.post, self.mode, self.s = llm, freecad or FreeCADRunner(), post, mode, settings
        self.repair_rounds = settings.repair_rounds if repair_rounds is None else repair_rounds
        self.deviation_fn = None  # multi-setup: callable(result, req) -> verification dict (replaces the single-setup check)
        self.store = get_store()

    # ======================================================================================== request
    def build_request(self, cad_path: str, out_dir: Path, fused: dict, ps_output: dict, ts_output: dict, stock: dict,
                      up: str = "+Z", translation=None) -> dict:
        """stock: {"kind": "box"|"user", "margins": {...}, "solid": <STEP path for a user blank>, "mesh_raw": <npz of the blank
        in the cad file's frame>}."""
        feats = {f["id"]: f for f in fused["features"]}
        mat = self.store.material(fused["material"]["id"])
        by_id = {t["id"]: t for t in self.store.tool_library["tools"]}
        tools, controllers, ops, skipped = {}, {}, [], []
        seq_ids = {s["step"]: s for s in ps_output["sequence"]}
        for a in ts_output["assignments"]:
            p = a["process"]
            if not a.get("requires_tool"):
                continue
            if not a.get("tool_id"):
                skipped.append({"step": a["step"], "process": p, "reason": "no tool"})
                continue
            fs = [feats[i] for i in a.get("feature_ids", []) if i in feats]
            phys = (a["tool_id"], a.get("tool_diameter_mm"))
            if phys not in tools:
                lib = dict(by_id.get(a["tool_id"], {}))
                sp = {**(lib.get("shape_params") or {}), "Diameter": a.get("tool_diameter_mm") or (lib.get("shape_params") or {}).get("Diameter")}
                tools[phys] = {"tool_number": len(tools) + 1, "tool_id": a["tool_id"], "name": a.get("tool_name") or lib.get("name"), "category": lib.get("category"),
                               "diameter_mm": a.get("tool_diameter_mm"), "flutes": lib.get("flutes", 2), "freecad_shape": lib.get("freecad_shape") or "endmill.fcstd",
                               "shape_params": sp, "corner_radius_mm": lib.get("corner_radius_mm", 0.0), "overall_length_mm": lib.get("overall_length_mm")}
            tn = tools[phys]["tool_number"]
            ckey = (tn, a.get("spindle_rpm"), a.get("feed_mm_min"), a.get("plunge_feed_mm_min"))
            if ckey not in controllers:  # one controller per speed/feed set: finishing does not inherit roughing feeds
                controllers[ckey] = {"tc": len(controllers) + 1, "tool_number": tn, "spindle_rpm": a.get("spindle_rpm"),
                                     "feed_mm_min": a.get("feed_mm_min"), "plunge_feed_mm_min": a.get("plunge_feed_mm_min")}
            op = self._op_for(a, fs, fused, ts_output, seq_ids.get(a["step"], {}))
            if op is None:
                skipped.append({"step": a["step"], "process": p, "reason": "no CAM mapping in this harness (e.g. fillet milling outside a cavity)"})
                continue
            if op.get("op_type") in ("Drilling", "Pocket", "Adaptive", "MillFace") and not op.get("faces"):
                skipped.append({"step": a["step"], "process": p, "reason": "no machinable faces in this setup"})
                continue
            op.update(step=a["step"], process=p, tool_number=tn, tc=controllers[ckey]["tc"], feature_ids=[i for i in a.get("feature_ids", []) if i in feats],
                      candidates=[c for c in a.get("candidates", []) if c != a["tool_id"]], doc_mm=a.get("doc_mm"), woc_mm=a.get("woc_mm"),
                      peck_mm=a.get("peck_mm"), spindle_rpm=a.get("spindle_rpm"), feed_mm_min=a.get("feed_mm_min"), plunge_feed_mm_min=a.get("plunge_feed_mm_min"),
                      coolant_mode=a.get("coolant_mode"), tool_diameter_mm=a.get("tool_diameter_mm"), label=f"{a['step']:02d} {p} ({a['tool_id']})")
            ops.append(op)
        if os.environ.get("CAM_BLIND_FLOOR", "0") == "1":
            from ..services.blind_floor import floor_tool, tip_height
            for op in list(ops):
                hs = [feats[i] for i in op["feature_ids"] if feats[i].get("type") == "hole"]
                if op["process"] != "drilling" or len(hs) != 1 or hs[0].get("through"):
                    continue
                h = hs[0]
                drill = next(t for t in tools.values() if t["tool_number"] == op["tool_number"])
                tip = tip_height(drill, float(drill.get("diameter_mm") or 0))
                if tip is None:
                    continue
                cert = self.freecad.run("certify_blind_floor.py", dict(path=cad_path, up=up,
                    translation=translation, stock=stock.get("solid"), hole=h), out_dir / ("floor_"+h["id"]))
                mill = floor_tool(by_id.values(), cert, (mat or {}).get("iso_group"))
                if mill is None:
                    continue
                cd = cutting_data(mill, "counterboring", mat, {"depth": cert["depth"]})
                if cd.get("error"):
                    continue
                cd.pop("strategy", None)
                tn = max(t["tool_number"] for t in tools.values()) + 1
                tc = max(c["tc"] for c in controllers.values()) + 1
                tools[(mill["id"], mill["diameter_mm"], h["id"])] = dict(mill, tool_number=tn,
                    tool_id=mill["id"], shape_params=dict(mill["shape_params"]))
                controllers[(tn, h["id"])] = dict(tc=tc, tool_number=tn,
                    **{k:cd[k] for k in ("spindle_rpm", "feed_mm_min", "plunge_feed_mm_min")})
                # The native drilling endpoint is the tip, not the full-diameter shoulder.
                op.update(final_depth=cert["floor_z"], a35_tip_endpoint=True,
                          a35_tip_height_mm=tip, a35_floor=cert)
                floor_step = max(o["step"] for o in ops) + 1
                ops.insert(ops.index(op)+1, dict(op_type="Custom", strategy="counterbore",
                    process="counterboring", step=floor_step, feature_ids=op["feature_ids"],
                    tool_number=tn, tc=tc, candidates=[],
                    label=f"{floor_step:02d} flat floor ({mill['id']})", a35_floor=cert,
                    cbores=[{k:cert[k] for k in ("x","y","top_z","diameter","depth")}], **cd))
                h["a35_floor_milled"] = True
        self._finish_stock(ops)
        holes = [{k: f[k] for k in ("diameter", "top", "top_z", "depth") if k in f} for f in fused["features"]
                 if f["type"] == "hole" and f.get("vertical", True) and any(f["id"] in o["feature_ids"] for o in ops if o["op_type"] == "Drilling")]
        for o in ops:
            if o["op_type"] == "Custom":
                o["drilled_holes"] = holes
        m = stock.get("margins") or SETTINGS.stock_margins
        st = {"xy_margin": m.get("xy_margin", 2.0), "z_top": m.get("z_top", 1.0), "z_bottom": m.get("z_bottom", 0.0), "kind": stock.get("kind", "box")}
        if stock.get("solid"):
            st["solid"] = str(stock["solid"])
        if stock.get("mesh_raw"):
            st["mesh_raw"] = str(stock["mesh_raw"])
        return {"path": cad_path, "out_dir": str(out_dir), "post": self.post, "stock": st, "tools": sorted(tools.values(), key=lambda t: t["tool_number"]),
                "controllers": sorted(controllers.values(), key=lambda c: c["tc"]), "operations": ops, "skipped": skipped, "up": up, "translation": translation,
                "material": (mat or {}).get("id"), "iso_group": (mat or {}).get("iso_group")}

    def _op_for(self, a: dict, fs: list[dict], fused: dict, ts: dict, step: dict) -> dict | None:
        p = a["process"]
        feats = fused["features"]
        if p in ("spot_drilling", "drilling", "reaming", "boring", "tapping", "countersinking"):
            holes = [f for f in fs if f["type"] == "hole" and f.get("vertical", True)]
            # faces[0] of a hole is its narrowest cylinder (counterbore / countersink faces come after it)
            op = {"op_type": "Drilling", "faces": [h["faces"][0] for h in holes if h.get("faces")]}
            op["blind"] = bool(holes) and not all(h["through"] for h in holes)
            op["tapping"] = p == "tapping"
            if holes:
                spec = self.store.thread(holes[0].get("thread"))
                if p == "tapping":
                    op["thread"] = spec["thread"] if spec else holes[0].get("thread")
                    op["pitch_mm"] = spec["pitch"] if spec else None
                top = max(float(h.get("top_z", h["top"][2])) for h in holes)
                start = self._drill_start(holes, fused)
                if start is not None:
                    op["start_depth"] = start
                    top = start
                if p == "spot_drilling":
                    final = top - 0.45 * min(min(float(h["diameter"]) for h in holes), float(a.get("tool_diameter_mm") or 6.0))
                    # a shallow blind hole: the cone stops at its flat bottom, not below it (brep2cam@savrasov)
                    bottoms = [float(h.get("top_z", h["top"][2])) - float(h["depth"]) for h in holes if not h.get("through") and h.get("depth")]
                    op["final_depth"] = round(max([final] + bottoms), 3)
                elif p == "countersinking":
                    d = max(float(h.get("countersink_diameter") or h["diameter"]) for h in holes)
                    op["final_depth"] = round(top - d / 2.0, 3)  # 90 deg countersink: depth = diameter / 2 at the tool tip
                elif p == "tapping":
                    thread_depth = min(float(h["depth"]) for h in holes) if not all(h["through"] for h in holes) else None
                    if thread_depth and spec:
                        op["final_depth"] = round(top - max(thread_depth - 2.5 * spec["pitch"], spec["pitch"]), 3)  # DIN 76 run-out
                if (os.environ.get("CAM_TWO_SIDED_DRILL", "0") == "1" and p == "drilling"
                        and len(holes) == 1 and holes[0].get("two_sided_drill")):
                    c = holes[0]["two_sided_drill"]
                    op.update(start_depth=c["start_depth"], final_depth=c["final_depth"],
                              a34_tip_endpoint=True)
            return op
        if p == "counterboring":
            holes = [f for f in fs if f["type"] == "hole" and f.get("counterbore_diameter")]
            return {"op_type": "Custom", "strategy": "counterbore",
                    "cbores": [{"x": h["top"][0], "y": h["top"][1], "top_z": float(h["top_z"]), "diameter": float(h["counterbore_diameter"]),
                                "depth": float(h["counterbore_depth"])} for h in holes]}
        if p == "thread_milling":
            holes = [f for f in fs if f["type"] == "hole"]
            spec = self.store.thread(holes[0].get("thread")) if holes else None
            if not spec:
                return None
            return {"op_type": "Custom", "strategy": "thread_mill", "pitch_mm": spec["pitch"], "major_mm": spec["major_mm"], "thread": spec["thread"],
                    "threads": [{"x": h["top"][0], "y": h["top"][1], "top_z": float(h["top_z"]),
                                 "thread_depth": float(h["depth"]) if h.get("through") else max(float(h["depth"]) - 2.5 * spec["pitch"], spec["pitch"])} for h in holes]}
        if p == "facing":
            tops = [f for f in fs if f["type"] == "planar_face" and f.get("role") == "top" and f.get("face")]
            op = {"op_type": "MillFace", "faces": [f["face"] for f in tops]}
            zs = [float(f["z"]) for f in tops if f.get("z") is not None]
            if zs and os.environ.get("CAM_FACE_STOP", "1") == "1":
                op["final_depth"] = max(zs)  # the face mill stops on the top face; it does not step down through the body
                op["face_floor"] = max(zs)
            return op
        if os.environ.get("CAM_ALLOWANCE_SKIM", "1") == "1" and self._skim_only(fused) and p not in ("facing", "chamfering", "deburring"):
            return None
        if p in ("pocket_milling", "pocket_finishing"):
            pk = [f for f in fs if f["type"] in ("pocket", "slot", "step")]
            nested = any(self._nested_with(f, feats) for f in pk)
            # FreeCAD Adaptive "profile" on a pocket face walks outside the pocket and cuts the part
            # (p01382 step 9, 45 % gouge). The height-field pass stays on the real floor.
            field_finish = p.endswith("finishing") and os.environ.get("CAM_POCKET_FINISH_FIELD", "1") == "1"
            if (not field_finish and pk and all(f["type"] == "pocket" and not f.get("open") for f in pk)
                    and not nested and all(f.get("face") for f in pk)):
                return {"op_type": "Adaptive", "faces": [f["face"] for f in pk], "adaptive_mode": "profile" if p.endswith("finishing") else "clearing"}
            fc = self._feature_clearing(pk, fused)
            if fc is not None:
                return fc
            # If the step targets a cavity (e.g. LLM called cavity machining pocket_milling/finishing), map to cavity strategy
            cav = next((f for f in fs if f["type"] == "cavity"), None) or next((f for f in feats if f.get("type") == "cavity"), None)
            if cav:
                return {"op_type": "Custom", "strategy": "roughing" if p == "pocket_milling" else "finishing",
                        "reach_limit_mm": a.get("reach_limit_mm"), "cavity": True}
            return None
        if p in ("slot_milling", "slot_finishing", "step_milling"):
            return self._feature_clearing([f for f in fs if f["type"] in ("pocket", "slot", "step")], fused)
        if p in ("profile_milling", "profile_finishing"):
            if os.environ.get("CAM_PROFILE_NEEDS_OUTER", "0") == "1" and fs and not any(f["type"] in ("outer_profile", "boss") for f in fs):
                # Profile follows the whole model's outline (faces []): on sloped faces or fillets it is not their operation,
                # and on a large model FreeCAD times out on it and the setup loses its program (evolution A01)
                return None
            return {"op_type": "Profile", "faces": []}
        if p in ("cavity_roughing", "surface_finishing"):
            if os.environ.get("CAM_FREECAD_3D", "0") == "1":
                # experiment: FreeCAD's own 3D operations instead of the harness's height-map paths (build_cam_job.py):
                # model-aware Adaptive clearing, OCL drop-cutter Surface (+ Waterline with CAM_FC3D_WATERLINE=1)
                return {"op_type": "Adaptive3D" if p == "cavity_roughing" else "Surface3D", "reach_limit_mm": a.get("reach_limit_mm"),
                        "waterline": os.environ.get("CAM_FC3D_WATERLINE", "0") == "1"}
            cav = next((f for f in fs if f["type"] == "cavity"), None)
            return {"op_type": "Custom", "strategy": "roughing" if p == "cavity_roughing" else "finishing", "reach_limit_mm": a.get("reach_limit_mm"),
                    "seal_chamfers": [f["bbox"][:4] + [f["z_min"], f["z_max"]] if f.get("z_min") is not None and f.get("z_max") is not None else f["bbox"]
                                      for f in feats if f.get("type") == "chamfer" and f.get("bbox") and self._chamfer_milled(f, ts)], "cavity": bool(cav)}
        if p == "chamfering":
            ch = [f for f in fs if f["type"] == "chamfer" and f.get("bbox") and f.get("normal")]
            if not ch:
                return None
            return {"op_type": "Custom", "strategy": "chamfer", "chamfer_features": [{k: f.get(k) for k in ("id", "bbox", "z_min", "z_max", "angle_deg", "normal", "size")} for f in ch]}
        if p == "deburring":
            top = [f for f in feats if f["type"] == "planar_face" and f.get("role") == "top" and not f.get("facing_only")]
            return {"op_type": "Deburr", "faces": [top[0]["face"]] if top else [], "deburr_mode": "edge_break",
                    "chamfer_faces": [f["face"] for f in feats if f["type"] == "chamfer"], "chamfer_size": 0.3}
        return None

    @staticmethod
    def _skim_only(fused: dict) -> bool:
        """A side added only to skim a thin blank cap (<= 0.5 mm): no pocket, hole or cavity to machine."""
        feats = fused.get("features") or []
        allow = [float(f.get("allowance_mm") or 0.0) for f in feats if f.get("allowance_mm")]
        if not allow or max(allow) > 0.5:
            return False
        return not any(f.get("type") in ("cavity", "pocket", "slot", "step", "hole") for f in feats)

    @staticmethod
    def _chamfer_milled(f: dict, ts: dict) -> bool:
        """True when a chamfering step of this setup machines this chamfer: a planar chamfer (the chamfer strategy needs
        its normal) named by a chamfering assignment. Other chamfers are left to the cavity passes, not sealed."""
        if not f.get("normal"):
            return False
        ids = {i for x in (ts or {}).get("assignments", []) if x.get("process") == "chamfering" for i in x.get("feature_ids", [])}
        return f.get("id") in ids or "*" in ids

    def _feature_clearing(self, pk: list[dict], fused: dict) -> dict | None:
        if not pk:
            return None
        return {"op_type": "Custom", "strategy": "feature", "slot_features": [self._slot_geom(f, fused) for f in pk if f.get("bbox")]}

    @staticmethod
    def _finish_stock(ops: list[dict]) -> None:
        """A roughing step leaves the finishing allowance only when a finishing step of the same feature follows."""
        allowance = get_store().cutting["finish_allowance_mm"]["wall"]
        for k, o in enumerate(ops):
            o.setdefault("finish_stock", 0.0)
            fin = [x for x in ops[k + 1:] if ROUGH_OF.get(x["process"]) == o["process"] and (set(x["feature_ids"]) & set(o["feature_ids"]) or o["process"] == "cavity_roughing")]
            if fin:
                o["finish_stock"] = allowance

    # ---------------------------------------------------------------------------------- geometry helpers (nesting, holes)
    @staticmethod
    def _area(bb) -> float:
        return abs((float(bb[2]) - float(bb[0])) * (float(bb[3]) - float(bb[1])))

    @staticmethod
    def _overlap(a, b) -> bool:
        return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

    @staticmethod
    def _circle_hits(cx, cy, r, bb) -> bool:
        dx = max(bb[0] - cx, 0.0, cx - bb[2])
        dy = max(bb[1] - cy, 0.0, cy - bb[3])
        return dx * dx + dy * dy <= r * r + 1e-6

    def _inner(self, inner: dict, outer: dict) -> bool:
        ib, ob = inner.get("bbox") or [], outer.get("bbox") or []
        if len(ib) < 4 or len(ob) < 4 or not self._overlap(ib, ob) or inner.get("z") is None or outer.get("z") is None:
            return False
        return float(inner["z"]) < float(outer["z"]) - 0.2 and self._area(ib) < 0.95 * max(self._area(ob), 1e-6)

    def _nested_with(self, f: dict, feats: list[dict]) -> bool:
        for o in feats:
            if o is f or o.get("id") == f.get("id"):
                continue
            if o.get("type") in ("pocket", "slot", "step") and (self._inner(o, f) or self._inner(f, o)):
                return True
            if o.get("type") == "hole" and o.get("top") and f.get("bbox") and self._circle_hits(o["top"][0], o["top"][1], o["diameter"] / 2, f["bbox"]):
                return True
        return False

    def _slot_geom(self, f: dict, fused: dict) -> dict:
        part = (fused.get("part") or {}).get("bbox") or []
        bb = f["bbox"]
        sides = []
        if f.get("open") and len(part) >= 5:
            px0, py0, px1, py1 = float(part[0]), float(part[1]), float(part[3]), float(part[4])
            for cond, side in ((abs(bb[0] - px0) <= 1.0, "-x"), (abs(bb[2] - px1) <= 1.0, "+x"), (abs(bb[1] - py0) <= 1.0, "-y"), (abs(bb[3] - py1) <= 1.0, "+y")):
                if cond:
                    sides.append(side)
        geom = {"bbox": list(bb), "open_sides": sides, "floor_z": float(f.get("z") or 0.0)}
        parents = [o for o in fused["features"] if o.get("type") in ("pocket", "slot", "step") and o.get("id") != f.get("id") and self._inner(f, o)]
        if parents:
            geom["start_z"] = float(min(parents, key=lambda o: self._area(o["bbox"]))["z"])
        geom["holes"] = [{k: h.get(k) for k in ("diameter", "top", "top_z", "depth")} for h in fused["features"]
                         if h.get("type") == "hole" and h.get("vertical", True) and h.get("top") and self._circle_hits(h["top"][0], h["top"][1], h["diameter"] / 2, bb)]
        return geom

    def _drill_start(self, holes: list[dict], fused: dict) -> float | None:
        """A hole that opens from a pocket / slot floor starts at that floor (one value per operation)."""
        floors = []
        pockets = [f for f in fused["features"] if f.get("type") in ("pocket", "slot", "step") and f.get("bbox") and f.get("z") is not None]
        for h in holes:
            top = h.get("top") or []
            top_z = float(h.get("top_z", top[2]))
            # the hole must open AT the floor: a hole that starts higher (e.g. through a boss standing inside the
            # pocket's bbox) is drilled from its own top, otherwise the cycle rapids through material down to R
            hosts = [p for p in pockets if self._circle_hits(top[0], top[1], h["diameter"] / 2, p["bbox"])
                     and float(p["z"]) - 0.2 <= top_z <= float(p["z"]) + 0.5
                     and top_z - float(h.get("depth") or 0) < float(p["z"]) - 0.2]
            if not hosts:
                return None
            floors.append(min(float(p["z"]) for p in hosts))
        if not floors or max(floors) - min(floors) > 0.2:
            return None
        return round(min(floors), 3)

    # ======================================================================================== closed loop
    def run(self, cad_path: str, out_dir: Path, fused: dict, ps_output: dict, ts_output: dict, stock: dict, frame: Frame) -> dict:
        req = self.build_request(cad_path, out_dir, fused, ps_output, ts_output, stock, frame.up, frame.t.tolist())
        return self.run_request(req, out_dir, fused, frame=frame)

    def run_request(self, req: dict, out_dir: Path, fused: dict, frame: Frame | None = None) -> dict:
        out_dir = Path(out_dir)
        self._features = {f["id"]: f for f in fused["features"]}
        self._fused = fused
        if not self.freecad.available():
            return {"available": False, "error": "freecadcmd not found", "request": req, "rounds": []}
        rounds, best, best_rank, prev_rank, best_rank_round = [], None, None, None, 0
        rejected_fixes = []
        part_k, stock_k = self._meshes(req, out_dir, frame)  # before the custom toolpaths: they start from the same stock
        self._part_k, self._stock_k, self._shield_key = part_k, stock_k, None
        import time as _time
        for r in range(self.repair_rounds + 1):
            tm, t0 = {}, _time.time()  # where a round's time goes (seconds per sub-step)
            rest_src = (None, None, None)
            try:
                self._generate_custom(req, out_dir)
                tm["custom_toolpaths_s"] = round(_time.time() - t0, 1)
                t1 = _time.time()
                result = self.freecad.run("build_cam_job.py", req, out_dir / "fc")
                tm["freecad_job_s"] = round(_time.time() - t1, 1)
            except (FreeCADError, ValueError) as e:
                result = {"error": str(e), "verification": {"ok": False, "issues": [str(e)[:500]], "op_errors": []}}
            ver = result.setdefault("verification", {})
            ver.setdefault("issues", [])
            ver.setdefault("warnings", [])
            if not result.get("operations"):
                ver["issues"].append("no machinable operation was produced")
            ver["issues"] += self._lint(req)
            t2 = _time.time()
            if result.get("gcode_file"):
                from ..services import gcode as G
                from ..services.postfix import fix_program, safety_report
                result["postfix"] = fix_program(result["gcode_file"], req["operations"], self.s.tap_cycle, self.s.tap_m29)
                # every round writes the same program / job files: keep this round's copies so that the best round,
                # not the last one, is what gets delivered (and what the multi-setup workpiece holds)
                import shutil as _sh
                keep = out_dir / f"program_round{r}.ngc"
                _sh.copy(result["gcode_file"], keep)
                result["round_gcode"] = str(keep)
                if result.get("job_file") and Path(result["job_file"]).exists():
                    keep_job = out_dir / f"job_round{r}{Path(result['job_file']).suffix}"
                    _sh.copy(result["job_file"], keep_job)
                    result["round_job"] = str(keep_job)
                parsed = G.parse(result["gcode_file"])
                ver["issues"] += safety_report(parsed, req["operations"])
                ver["warnings"] += [f"G-code: {w}" for w in parsed["warnings"]]
                m = self.store.machine
                ver["cycle_time"] = G.cycle_time_min(parsed, m["rapid_mm_min"], m["tool_change_s"])
                try:
                    if self.deviation_fn is not None:
                        dv = dict(self.deviation_fn(result, req))
                    else:
                        from ..services.verify import simulate_setup
                        sim = simulate_setup(result["gcode_file"], part_k, stock_k, req["operations"], req["tools"], fused["features"],
                                             tol=self.s.tol_mm, max_cells=self.s.sim_max_cells, tooling=tooling_limits(self.store, req.get("iso_group")),
                                              samples=self._samples_k(req, out_dir, part_k),
                                              request_frame=Frame(req["up"], req["translation"])
                                              if req.get("up") is not None and req.get("translation") is not None else None)
                        dv = {"stats": sim["stats"], "regions": sim["regions"], "errors": sim["errors"], "_sim": sim}
                        if "exact_observations" in sim:
                            dv["exact_observations"] = sim["exact_observations"]
                        if "cavity_region_observations" in sim:
                            dv["cavity_region_observations"] = sim["cavity_region_observations"]
                        self._save_sim(out_dir, sim, r)
                    if dv.get("_sim") is not None and "exact_observations" in dv:
                        from ..services.exact_issues import save as save_exact_issues
                        save_exact_issues(out_dir / "exact_issues.npz", dv["_sim"])
                        import shutil
                        shutil.copy(out_dir / "exact_issues.npz", out_dir / f"round{r}_exact_issues.npz")
                    rest_src = (dv.pop("_sim", None), dv.pop("_inside", None), dv.pop("_frame", None))
                    ver["deviation"] = dv
                    exact_obs = dv.get("exact_observations")
                    if exact_obs and exact_obs["needs_review"]:
                        ver["needs_review"] = True
                        for kind, summary in exact_obs["kinds"].items():
                            if summary["uncovered_unexpected"]["count"]:
                                ver["issues"].append(f"deviation: exact {kind}: {summary['count']} probes, {summary['area_mm2']:.3f} mm2, "
                                                     f"{summary['cluster_count']} clusters; {summary['uncovered']['count']} outside original column regions; "
                                                     "owner unknown; needs review, no causal operation attribution")
                        ver["warnings"].extend(exact_obs["limits"])
                    for reg in dv["errors"]:
                        what = "material left" if reg["kind"] == "excess" else "overcut"
                        ver["issues"].append(f"deviation: {what} up to {reg['max_mm']} mm ({reg['volume_mm3']} mm3) on {reg['feature']} at {reg['at']}; "
                                             f"last cut by step {reg['last_cut_by_step']} ({reg['last_cut_by']}); steps on this feature: {reg['steps_on_feature']}")
                    for reg in dv["regions"]:
                        if reg["klass"] != "error":
                            ver["warnings"].append(f"deviation ({reg['klass']}): {reg['kind']} {reg['max_mm']} mm on {reg['feature']}: {reg['note']}")
                except Exception as e:  # noqa: BLE001  (the check must never break the pipeline)
                    ver["warnings"].append(f"deviation check failed: {type(e).__name__}: {e}")
            tm["verify_s"] = round(_time.time() - t2, 1)  # post fix, G-code checks, material-removal simulation, deviation
            ver["ok"] = not ver["issues"] and not ver.get("op_errors")
            stats = (ver.get("deviation") or {}).get("stats") or {}
            n_issues = len(ver["issues"]) + len(ver.get("op_errors") or [])
            hard = sum(1 for i in ver["issues"] if not str(i).startswith("deviation:")) + len(ver.get("op_errors") or [])
            rank = (hard, float(stats.get("gouge_area_pct", 0.0)), (-round(float(stats.get("surface_within_tol_pct", 0.0)), 2), n_issues))
            rounds.append({"round": r, "ok": ver["ok"], "issues": ver["issues"], "op_errors": ver.get("op_errors", []), "warnings": ver["warnings"],
                           "deviation": stats, "timing": tm})
            self.llm.trace.log("agent", agent=self.name, round=r, ok=ver["ok"], issues=ver["issues"][:10], deviation=stats)
            if best_rank is None or better_round(rank, best_rank):
                best, best_rank = (json.loads(json.dumps(req, default=str)), result), rank
                best_rank_round = r
            # Small scattered defects never reach the error-region threshold, yet they keep the surface share low.
            short = float(stats.get("surface_within_tol_pct", 100.0) or 100.0) < 99.0
            if (self.s.rest and (not ver["ok"] or short) and r < self.repair_rounds and rest_src[0] is not None
                    and not any(o.get("strategy") == "rest" for o in req["operations"])):
                t3 = _time.time()
                added = self._plan_rest(req, out_dir, *rest_src)
                tm["rest_plan_s"] = round(_time.time() - t3, 1)
                if added:  # the next round builds the job with the rest pass (a deterministic fix)
                    rounds[-1].update(fixes=added, fix_source="rest")
                    prev_rank = rank
                    continue
            if ver["ok"] or r == self.repair_rounds or (prev_rank is not None and not better_round(rank, prev_rank)):
                break
            prev_rank = rank
            fixes = self._deterministic_fixes(req, ver)
            source = "deterministic"
            if not fixes and self.mode != "deterministic":
                try:
                    fixes, source = self._llm_fixes(req, ver, rejected_fixes), "llm"
                except Exception as e:  # noqa: BLE001
                    ver["warnings"].append(f"LLM repair skipped: {type(e).__name__}: {e}")
            if not fixes:
                break
            outcomes = []
            before_fixes = copy.deepcopy(req)
            req = self._apply_fixes(req, fixes, fused, outcomes)
            rounds[-1]["fixes"] = fixes
            rounds[-1]["fix_source"] = source
            if os.environ.get("CAM_REPAIR_FEEDBACK", "0") == "1":
                rounds[-1]["fix_outcomes"] = outcomes
                rejected_fixes = [x for x in outcomes if x["status"] != "applied"]
                if self._repair_parameters(req) == self._repair_parameters(before_fixes):
                    rounds[-1]["repair_stop"] = "no effective change"
                    break
        req, result = best
        if best_rank_round != rounds[-1]["round"] and result.get("round_gcode") and result.get("gcode_file"):
            import shutil as _sh
            _sh.copy(result["round_gcode"], result["gcode_file"])
            if result.get("round_job") and result.get("job_file"):
                _sh.copy(result["round_job"], result["job_file"])
            if self.deviation_fn is not None:  # the verifier's state of this setup = the delivered program
                try:
                    self.deviation_fn(result, req)
                except Exception as e:  # noqa: BLE001
                    result.setdefault("verification", {}).setdefault("warnings", []).append(f"re-check of the best round failed: {e}")
            result["restored_round"] = best_rank_round
            if self.deviation_fn is None:  # the saved simulation / metric points = the delivered program's
                for name in ("simulation.npz", "process_simulation.json", "surface_samples.npz", "exact_issues.npz"):
                    src = out_dir / f"round{best_rank_round}_{name}"
                    if src.exists():
                        _sh.copy(src, out_dir / name)
        for rd in rounds:
            for name in ("simulation.npz", "process_simulation.json", "surface_samples.npz", "exact_issues.npz"):
                (out_dir / f"round{rd['round']}_{name}").unlink(missing_ok=True)
        result.update(request=req, rounds=rounds, available=True, skipped=req.get("skipped", []),
                      best_round={"round": best_rank_round, "hard_issues": best_rank[0], "gouge_area_pct": best_rank[1], "within_pct": -best_rank[2][0],
                                  "issues": best_rank[2][1]})
        return result

    def _samples_k(self, req: dict, out_dir: Path, part_k: dict) -> dict:
        """The surface metric's points on the part's B-rep faces, in this setup's frame (sampled once per part)."""
        from ..services.verify import brep_samples, samples_in
        key = (req["path"], req.get("up", "+Z"), tuple(part_k["translation"]))
        if getattr(self, "_brep", None) is None or self._brep[0] != key:
            raw = brep_samples(req["path"], self.s.surface_samples, Path(out_dir) / "fc", self.freecad)
            self._brep = (key, samples_in(raw, Frame(req.get("up", "+Z"), part_k["translation"])))
        return self._brep[1]

    def _meshes(self, req: dict, out_dir: Path, frame: Frame | None):
        """Target and stock meshes in this setup's frame (target tessellated finer than a quarter of the tolerance)."""
        from ..services.stock import box_mesh, load_mesh, transform
        defl = max(0.01, self.s.tol_mm / 4)
        part_k = self.freecad.run("tessellate.py", {"path": req["path"], "deflection": defl, "up": req.get("up", "+Z"), "translation": req.get("translation")}, out_dir / "fc")
        st = req["stock"]
        if st.get("mesh_raw") and frame is not None:
            stock_k = transform(load_mesh(st["mesh_raw"]), frame)
        else:
            b = part_k["bbox"]
            m = st.get("xy_margin", 2.0)
            stock_k = box_mesh([b[0] - m, b[1] - m, b[2] - st.get("z_bottom", 0.0)], [b[3] + m, b[4] + m, b[5] + st.get("z_top", 1.0)])
        from ..services.stock import save_mesh
        req["stock"]["mesh_file"] = save_mesh(out_dir / "stock_mesh.npz", stock_k)
        return part_k, stock_k

    @staticmethod
    def _save_sim(out_dir: Path, sim: dict, r: int) -> None:
        """This round's dexel replay, metric numbers and metric points (setup frame); the delivered round's are copied
        back after the loop."""
        import shutil
        import numpy as np
        from ..services.verify import save_surface
        dm = sim["dexel"]
        np.savez_compressed(out_dir / "simulation.npz", h=dm.h.astype(np.float32), bottom=dm.bottom.astype(np.float32), owner=dm.owner,
                            target_h=sim["target_h"].astype(np.float32), grid=np.array([dm.g.x0, dm.g.y0, dm.g.res, dm.g.nx, dm.g.ny]))
        (out_dir / "process_simulation.json").write_text(json.dumps(sim["stats"], indent=1))
        save_surface(out_dir / "surface_samples.npz", sim["surface"], "setup")
        for name in ("simulation.npz", "process_simulation.json", "surface_samples.npz"):
            shutil.copy(out_dir / name, out_dir / f"round{r}_{name}")

    # ---------------------------------------------------------------------------------- custom toolpaths
    def _generate_custom(self, req: dict, out_dir: Path) -> None:
        todo = [o for o in req["operations"] if o["op_type"] == "Custom" and not o.get("gcode_file")]
        if not todo:
            return
        from ..services.paths import helical_counterbore, thread_milling
        from ..services.zlevel import chamfer_finishing, feature_clearing, finishing, roughing
        out_dir.mkdir(parents=True, exist_ok=True)
        tools = {t["tool_number"]: t for t in req["tools"]}
        hf = h_open = None
        for o in todo:
            tool = tools[o["tool_number"]]
            d = float(tool["diameter_mm"])
            feed, plunge = float(o.get("feed_mm_min") or 800), float(o.get("plunge_feed_mm_min") or 250)
            if o["strategy"] in ("counterbore", "thread_mill"):
                safe = max(float(h["top_z"]) for h in (o.get("cbores") or o.get("threads") or [{"top_z": 0}])) + 5.0
                if os.environ.get("CAM_CBORE_SAFE_TOP", "0") == "1":
                    # _meshes has already put both bounds in this setup's frame; z_top is only a margin.
                    safe = max(safe, float(self._stock_k["bbox"][5]) + 5.0, float(self._part_k["bbox"][5]) + 5.0)
                lines = helical_counterbore(o["cbores"], d, float(o.get("doc_mm") or d / 2), float(o.get("woc_mm") or 0.4 * d), feed, plunge, safe) \
                    if o["strategy"] == "counterbore" else thread_milling(o["threads"], d, float(o["pitch_mm"]), float(o["major_mm"]), feed, safe)
                info = {"holes": len(o.get("cbores") or o.get("threads") or [])}
            else:
                if hf is None:
                    hf = self._heightfield(req, out_dir, o.get("drilled_holes"))
                lim = _custom_depth_limit(o)
                hf.depth_limit = (hf.part_top - float(lim)) if lim else None
                if o["strategy"] == "chamfer":
                    # CAM_CHAMFER_OPEN=1: the chamfer face as it is, not the field the roughing sealed it in (a sealed
                    # face leaves a few stray points, the shield lifts the pass); savrasov kept a raw field for it
                    # (QUEUE P34). Off - main's shared field
                    hc = hf
                    if h_open is not None and os.environ.get("CAM_CHAMFER_OPEN", "0") == "1":
                        hc = copy.copy(hf)
                        hc.H = h_open
                    parts = [chamfer_finishing(hc, f, tool, feed, plunge) for f in o.get("chamfer_features") or []]
                    lines, info = [x for p in parts for x in p["gcode"]], {"faces": len(parts)}
                elif o["strategy"] == "feature":
                    fin = o["process"] in FINISHING
                    # finishing removes only the allowance: one level at the floor, passes 0.7 D apart (the roughing step
                    # left the walls and floor at a uniform allowance); roughing uses its own depth of cut and stepover
                    doc = 1e3 if fin else float(o.get("doc_mm") or d / 2)
                    step = 0.7 * d if fin else float(o.get("woc_mm") or 0.4 * d)
                    parts = [feature_clearing(hf, f["bbox"], f.get("open_sides") or [], float(f.get("floor_z") or 0.0), d, doc=doc,
                                              stepover=step, leave=float(o.get("finish_stock") or 0.0), feed=feed, plunge=plunge,
                                              top_z=f.get("start_z"), holes=f.get("holes")) for f in o.get("slot_features") or []]
                    lines, info = [x for p in parts for x in p["gcode"]], {"features": len(parts)}
                elif o["strategy"] == "roughing":
                    if h_open is None and o.get("seal_chamfers"):
                        h_open = hf.H.copy()
                    hf.seal_chamfers(o.get("seal_chamfers") or [])
                    r = roughing(hf, d, doc=float(o.get("doc_mm") or d / 2), stepover=float(o.get("woc_mm") or 0.4 * d), leave=float(o.get("finish_stock") or 0.0),
                                 feed=feed, plunge=plunge, top=hf.rough_top)
                    lines, info = r["gcode"], {k: v for k, v in r.items() if k != "gcode"}
                else:
                    step = min(max(float(o.get("woc_mm") or 0.1 * d) * 3, 0.1 * d), 0.45 * d)
                    area = (hf.x1 - hf.x0) * (hf.y1 - hf.y0)
                    wall_res = max(self.s.tol_mm / 2, math.sqrt(area / self.s.sim_max_cells))  # walls at the verification resolution
                    finish_top = not any(x.get("process") == "facing" for x in req["operations"])
                    top_coverage = os.environ.get("CAM_FINISH_TOP_COVERAGE", "0") == "1"
                    if top_coverage:
                        # Custom paths are generated BEFORE MillFace is built. Neither
                        # a facing feature nor its bbox proves actual plateau removal
                        # (especially an A03 facing_only face). Keep the top pass until
                        # such coverage can be proved from an executed facing path.
                        finish_top = True
                    r = finishing(hf, d, stepover=step, z_step=max(0.3, min(1.5, d / 4)), feed=feed, plunge=plunge, wall_res=wall_res,
                                  finish_top=finish_top)
                    lines, info = r["gcode"], {k: v for k, v in r.items() if k != "gcode"}
                    if top_coverage:
                        facing = [x for x in req["operations"] if x.get("process") == "facing"]
                        facing_only = any(getattr(self, "_features", {}).get(fid, {}).get("facing_only")
                                          for x in facing for fid in x.get("feature_ids", []))
                        reason = "no_facing" if not facing else "facing_only" if facing_only else "coverage_unproved"
                        info["top_coverage"] = {"finish_top": True, "reason": reason}
            lines, srep = self._shield(req, o, tool, lines)
            if srep:
                info["shield"] = srep
            path = out_dir / f"custom_step{o['step']:02d}_{o['strategy']}.ngc"
            path.write_text("\n".join(lines) + "\n")
            o["gcode_file"] = str(path.resolve())  # FreeCAD runs with cwd=<session>/fc: a relative CAM_OUT would break it
            o["strategy_info"] = info

    def _shield_grid(self, req: dict, operation: dict | None = None):
        from ..services.shield import Shield
        from ..services.grid_tools import grid_tool_selection, grid_receipt
        operations = req["operations"]
        # _plan_rest shields its new operation immediately before inserting it.
        if operation is not None and operation not in operations:
            operations = operations + [operation]
        radius, selection = grid_tool_selection(req["tools"], operations)
        # Preserve the old cache behavior while off; the selected radius fully
        # determines the lattice while on. Refresh the receipt even on a hit.
        key = ("path", radius) if selection["enabled"] else tuple(sorted(float(t.get("diameter_mm") or 0) for t in req["tools"]))
        if self._shield_key != key:
            self._shield_obj = Shield.for_setup(self._part_k, self._stock_k, req["tools"], self.s.tol_mm, self.s.sim_max_cells,
                                                operations=operations)
            self._shield_key = key
        self._shield_obj.grid_selection = grid_receipt(selection, self._shield_obj.grid)
        req["path_tool_grid"] = self._shield_obj.grid_selection
        return self._shield_obj

    def _plan_rest(self, req: dict, out_dir: Path, sim, inside=None, frame=None) -> list[dict]:
        """Rest finishing from this round's simulation (services/rest.py): residual cells, ball endmills chosen by the
        closing of the target, a gouge-free drop-cutter raster per tool. Adds the tools, controllers and Custom
        operations to the request (after the last milling step) and returns them as fixes."""
        from ..planning.cutting_data import cutting_data
        from ..services.dexel import shape_for
        from ..services.rest import plan_tools, raster, residual_mask
        tol = self.s.tol_mm
        try:
            M = residual_mask(sim, tol, inside, frame)
            if not M.any():
                return []
            sh = self._shield_grid(req)
            g = sim["grid"]
            if sh.T.shape != M.shape or abs(sh.x0 - g.x0) > 1e-6 or abs(sh.y0 - g.y0) > 1e-6 or abs(sh.res - g.res) > 1e-9:
                return []
            iso = req.get("iso_group")
            balls = []
            for cat, want in (("ball_endmill", (10.0, 6.0, 3.0)), ("bull_endmill", (16.0, 10.0)), ("endmill", (20.0, 12.0, 6.0))):
                lib = [t for t in self.store.tool_library["tools"] if t["category"] == cat and (not iso or iso in t["iso"])]
                lib = lib or [t for t in self.store.tool_library["tools"] if t["category"] == cat]
                by_d = {}
                for t in sorted(lib, key=lambda t: -float(t.get("overall_length_mm") or 0)):  # the longest of a diameter
                    by_d.setdefault(float(t["diameter_mm"]), t)
                ds = sorted(by_d, reverse=True)
                balls += [(by_d[d], shape_for(by_d[d])) for d in ([d for d in ds if d in want] or ds[:3])]
            z_floor = float(self._part_k["bbox"][2])
            part_top = float(self._part_k["bbox"][5])
            zlim = {t["id"]: part_top - tool_reach(t) for t, _ in balls}
            plans = plan_tools(sh, M, balls, tol, z_floor, z_limit=zlim)
            if not plans:
                return []
            mat = self.store.material(req.get("material")) if req.get("material") else None
            part_top = float(self._part_k["bbox"][5])
            safe_z = max(float(self._stock_k["bbox"][5]), part_top) + 5.0
            ops = req["operations"]
            milling = [k for k, o in enumerate(ops) if o.get("op_type") in ("Custom", "Pocket", "Adaptive", "Profile", "MillFace", "Adaptive3D", "Surface3D")
                       and o.get("process") not in ("chamfering", "deburring", "thread_milling")]
            at = (milling[-1] + 1) if milling else len(ops)
            added = []
            for p in plans:
                t = p["tool"]
                cd = cutting_data(t, "surface_finishing", mat, {"depth": 0.2, "diameter": t["diameter_mm"]}, strategy="finish")
                if cd.get("error"):
                    continue
                tool = next((x for x in req["tools"] if x.get("tool_id") == t["id"]), None)
                if tool is None:
                    tool = {"tool_number": max([x["tool_number"] for x in req["tools"]] + [0]) + 1, "tool_id": t["id"], "name": t.get("name"),
                            "category": t["category"], "diameter_mm": t["diameter_mm"], "flutes": t.get("flutes", 2),
                            "freecad_shape": t.get("freecad_shape") or "ballend.fcstd", "shape_params": {**(t.get("shape_params") or {}), "Diameter": t["diameter_mm"]},
                            "corner_radius_mm": t.get("corner_radius_mm", 0.0), "overall_length_mm": t.get("overall_length_mm")}
                    req["tools"].append(tool)
                feed = float(cd.get("feed_mm_min") or 800)
                plunge = float(cd.get("plunge_feed_mm_min") or feed / 3)
                tc = max([c["tc"] for c in req["controllers"]] + [0]) + 1
                req["controllers"].append({"tc": tc, "tool_number": tool["tool_number"], "spindle_rpm": cd.get("spindle_rpm"), "feed_mm_min": feed,
                                           "plunge_feed_mm_min": plunge})
                reach = tool_reach(t)
                lines, info = raster(p, sh, tol, part_top - reach, safe_z, feed, plunge)
                step = max(o["step"] for o in ops) + 1
                op = {"op_type": "Custom", "strategy": "rest", "step": step, "process": "surface_finishing", "tool_number": tool["tool_number"], "tc": tc,
                      "feature_ids": [], "candidates": [], "finish_stock": 0.0, "spindle_rpm": cd.get("spindle_rpm"), "feed_mm_min": feed,
                      "plunge_feed_mm_min": plunge, "coolant_mode": cd.get("coolant_mode"), "tool_diameter_mm": t["diameter_mm"],
                      "label": f"{step:02d} rest_finishing ({t['id']})"}
                lines, srep = self._shield(req, op, tool, lines)
                info["shield"] = srep
                path = Path(out_dir) / f"custom_step{step:02d}_rest.ngc"
                path.write_text("\n".join(lines) + "\n")
                op.update(gcode_file=str(path.resolve()), strategy_info=info)
                ops.insert(at, op)
                at += 1
                added.append({"step": step, "action": "add_rest_finishing", "value": t["id"],
                              "reason": f"simulation: {p['area_mm2']} mm2 left above the tolerance that a D{t['diameter_mm']:g} ball reaches "
                                        f"({info['cut_length_mm']:.0f} mm of path, stepover {info['stepover_mm']} mm)"})
            return added
        except Exception as e:  # noqa: BLE001  (rest machining is an improvement, never a failure)
            import traceback
            (Path(out_dir) / "rest_error.txt").write_text(traceback.format_exc())
            return []

    def _shield(self, req: dict, o: dict, tool: dict, lines: list[str]):
        """Gouge shield (services/shield.py): the path is checked against the target on the verifier's own grid; pieces
        that would cut deeper than a quarter of the tolerance (vertically or sideways) are lifted just enough."""
        if not self.s.shield or getattr(self, "_part_k", None) is None:
            return lines, None
        from ..services.dexel import shape_for
        from ..services.shield import shield_gcode
        shape = shape_for(tool, o.get("process") or "")
        if shape is None:
            return lines, None
        try:
            tol = self.s.tol_mm
            return shield_gcode(lines, self._shield_grid(req, o), shape, delta=-tol / 4, eps_xy=tol / 4)
        except Exception as e:  # noqa: BLE001  (the shield must never break generation)
            return lines, {"error": f"{type(e).__name__}: {str(e)[:300]}"}

    def _heightfield(self, req: dict, out_dir: Path, holes):
        from ..services.stock import load_mesh
        from ..services.zlevel import HeightField
        st = req.get("stock", {})
        mesh = self.freecad.run("tessellate.py", {"path": req["path"], "deflection": 0.05, "up": req.get("up", "+Z"), "translation": req.get("translation")}, out_dir / "fc")
        v = mesh["bbox"]
        smesh = load_mesh(st["mesh_file"]) if st.get("mesh_file") and Path(st["mesh_file"]).is_file() and st.get("kind") == "user" else None
        m = float(st.get("xy_margin", 2.0))
        if smesh is not None:
            from ..services.stock import union_bbox
            stock_bb = union_bbox(v, smesh["bbox"])
        else:
            stock_bb = [v[0] - m, v[1] - m, v[2] - float(st.get("z_bottom", 0.0)), v[3] + m, v[4] + m, v[5] + float(st.get("z_top", 0.0))]
        res = max(0.1, max(v[3] - v[0], v[4] - v[1]) / 900)
        return HeightField(mesh["vertices"], mesh["faces"], stock_bb, res, holes=holes, outside="keep" if m == 0 else "cut",
                           stock_surface=req.get("stock_surface"), stock_mesh=smesh)

    # ---------------------------------------------------------------------------------- checks and repairs
    @staticmethod
    def _lint(req: dict) -> list[str]:
        out = []
        ops = req["operations"]
        for k, o in enumerate(ops):
            if float(o.get("finish_stock") or 0.0) > 0 and o.get("op_type") in ("Pocket", "Adaptive", "Profile", "Custom", "Adaptive3D"):
                fids = set(o.get("feature_ids", []))
                later = [x for x in ops[k + 1:] if x["process"] in FINISHING and (set(x.get("feature_ids", [])) & fids or x["process"] == "surface_finishing")]
                if not later:
                    out.append(f"plan: step {o['step']} ({o['process']}) leaves {o['finish_stock']} mm finishing stock but no later step machines it")
        return out

    def _deterministic_fixes(self, req: dict, ver: dict) -> list[dict]:
        ops = {o["step"]: o for o in req["operations"]}
        fixes = []
        for e in ver.get("op_errors") or []:
            o = ops.get(e.get("step"))
            if o and "empty path" in str(e.get("error")) and o.get("candidates"):
                smaller = self._smaller(o)
                if smaller:
                    fixes.append({"step": o["step"], "action": "switch_tool", "value": smaller, "reason": "empty path: tool too large"})
        if os.environ.get("CAM_TIMEOUT_FALLBACK", "1") == "1" and any("timed out" in str(m) and "build_cam_job" in str(m) for m in ver.get("issues", [])):
            # FreeCAD gave up on the job: its own operations (Profile of the whole outline, Adaptive / Pocket on many faces)
            # are what takes it past the timeout; the next round keeps the harness's own paths (evolution A02)
            for o in req["operations"]:
                if o.get("op_type") in ("Profile", "Adaptive", "Pocket", "Adaptive3D", "Surface3D"):
                    fixes.append({"step": o["step"], "action": "drop_operation", "timeout": True,
                                  "reason": f"FreeCAD timed out building the job: {o['op_type']} left out"})
        for msg in ver.get("issues", []):
            if msg.startswith("plan: step "):
                step = int(msg.split()[2])
                fixes.append({"step": step, "action": "set_finish_stock", "value": 0.0, "reason": "finishing stock without a finishing step"})
        for reg in (ver.get("deviation") or {}).get("errors", []):
            steps = [ops[s] for s in reg.get("steps_on_feature") or [] if s in ops]
            last = ops.get(reg.get("last_cut_by_step"))
            if reg["kind"] == "gouge" and last and last["process"] == "spot_drilling":
                f = [self._features.get(i) for i in last["feature_ids"] if self._features.get(i)]
                dmin = min([float(x["diameter"]) for x in f if x.get("diameter")] or [5.0])
                fixes.append({"step": last["step"], "action": "set_depth_below_top", "value": round(0.4 * dmin, 2), "reason": "spot drill cone wider than the hole"})
            elif reg["kind"] == "gouge" and last and last["process"] == "deburring" and os.environ.get("CAM_DEBURR_DROP", "0") == "1":
                # the edge break cuts into the part (thin or small parts): what the LLM repair does by hand (evolution A07)
                fixes.append({"step": last["step"], "action": "drop_operation", "reason": "edge break gouges the part"})
            elif reg["kind"] == "excess" and steps:
                rough = [s for s in steps if s["process"] not in FINISHING and float(s.get("finish_stock") or 0) > 0]
                fin = [s for s in steps if s["process"] in FINISHING]
                if rough and not fin:
                    fixes.append({"step": rough[-1]["step"], "action": "set_finish_stock", "value": 0.0, "reason": "allowance left without finishing"})
                elif fin and fin[-1].get("candidates"):
                    smaller = self._smaller(fin[-1])
                    if smaller:
                        fixes.append({"step": fin[-1]["step"], "action": "switch_tool", "value": smaller, "reason": "material left: smaller finishing tool"})
        uniq = {(f["step"], f["action"]): f for f in fixes}
        return list(uniq.values())

    def _smaller(self, o: dict) -> str | None:
        lib = {t["id"]: t for t in self.store.tool_library["tools"]}
        d0 = float(o.get("tool_diameter_mm") or 0)
        c = sorted((lib[x] for x in o.get("candidates", []) if x in lib and float(lib[x].get("diameter_mm") or 0) < d0 - 1e-6), key=lambda t: -float(t["diameter_mm"]))
        return c[0]["id"] if c else None

    def _llm_fixes(self, req: dict, ver: dict, rejected_fixes: list[dict] | None = None) -> list[dict]:
        payload = {"operations": [{k: o.get(k) for k in ("step", "process", "op_type", "tool_number", "tool_diameter_mm", "doc_mm", "woc_mm", "peck_mm", "finish_stock",
                                                         "final_depth", "feature_ids", "candidates")} for o in req["operations"]],
                   "features": [{k: f.get(k) for k in ("id", "type", "diameter", "depth", "through", "thread", "top_z", "z", "size", "corner_radius") if f.get(k) is not None}
                                for f in self._features.values() if f.get("type") in ("hole", "pocket", "slot", "step", "cavity")],
                   "deviation_stats": (ver.get("deviation") or {}).get("stats"), "deviation_errors": (ver.get("deviation") or {}).get("errors", [])[:8],
                   "issues": ver.get("issues", [])[:20], "op_errors": ver.get("op_errors", [])[:10]}
        exact_obs = (ver.get("deviation") or {}).get("exact_observations")
        if exact_obs:
            from ..services.exact_issues import repair_observation
            # Bounded evidence first, so the existing payload budget cannot hide
            # exact-only issues behind long operation/feature lists.
            payload = {"exact_observations": repair_observation(exact_obs), **payload}
        prompt = REPAIR_PROMPT
        if os.environ.get("CAM_REPAIR_FEEDBACK", "0") == "1" and os.environ.get("CAM_REPAIR_FEEDBACK_PROMPT", "1") == "1":
            # Only the immediately preceding rejected actions; choice/history is A42.
            payload = {"rejected_fixes": rejected_fixes or [], **payload}
            for shown, op in zip(payload["operations"], req["operations"]):
                shown["reach_limit_mm"] = op.get("reach_limit_mm")
                if op["op_type"] == "Custom" and op.get("strategy") in ("roughing", "finishing"):
                    shown["depth_limit_mm"] = _custom_depth_limit(op)
            prompt += ("\nWOC (woc_mm) is width of cut/stepover in mm, not a work offset; it must be >0. "
                       "DOC and peck are positive mm. set_depth_below_top accepts 0 < mm <= 60; "
                       "for Custom roughing/finishing it sets depth_limit_mm below the part top. "
                       "finish_stock accepts 0..1 mm. Rejected/no-op fixes did not change the job; do not repeat them.")
        obj = self.llm.json_call(prompt, json.dumps(payload, ensure_ascii=False, default=str)[:9000], agent=self.name + ".repair", thinking=False)
        fixes = [f for f in obj.get("fixes", []) if isinstance(f, dict) and f.get("action")]
        if exact_obs:
            from ..services.exact_issues import safe_fixes
            fixes = safe_fixes(fixes, ver)
        return fixes

    def _top_z(self, o: dict) -> float | None:
        zs = []
        for fid in o.get("feature_ids", []):
            f = self._features.get(fid, {})
            if f.get("top_z") is not None:
                zs.append(float(f["top_z"]))
            elif f.get("z") is not None:
                zs.append(float(f["z"]) + float(f.get("depth") or 0.0))
        return max(zs) if zs else None

    @staticmethod
    def _repair_parameters(req: dict) -> dict:
        result = copy.deepcopy(req)
        for op in result["operations"]:
            op.pop("gcode_file", None)
            kind = op.get("op_type")
            # Match the actual StepOver payload in fc_scripts/build_cam_job.py,
            # including integer rounding and lower/upper clamps.
            woc_limits = {"MillFace": (0.7, 10, 90), "Pocket": (0.4, 10, 90),
                          "Adaptive": (0.3, 10, 60), "Adaptive3D": (0.3, 10, 60),
                          "Surface3D": (0.1, 5, 50)}
            if kind in woc_limits:
                default, low, high = woc_limits[kind]
                diameter = float(op["tool_diameter_mm"])
                op["woc_mm"] = max(low, min(high, round(100 * float(op.get("woc_mm") or default * diameter) / diameter)))
            elif kind != "Custom":
                op.pop("woc_mm", None)
            if kind != "Drilling":
                op.pop("peck_mm", None)
            if kind in ("MillFace", "Drilling", "Deburr"):
                op.pop("finish_stock", None)
            elif kind in ("Pocket", "Adaptive", "Profile", "Adaptive3D", "Surface3D"):
                op["finish_stock"] = float(op.get("finish_stock") or 0.0)
            if kind == "Custom":
                op.pop("final_depth", None)
                strategy = op.get("strategy")
                diameter = float(op["tool_diameter_mm"])
                if strategy in ("roughing", "counterbore", "feature"):
                    op["doc_mm"] = float(op.get("doc_mm") or diameter / 2)
                    op["woc_mm"] = float(op.get("woc_mm") or 0.4 * diameter)
                    op["finish_stock"] = float(op.get("finish_stock") or 0.0)
                    if strategy == "counterbore":
                        op.pop("finish_stock", None)
                    if strategy == "feature":
                        if op.get("process") in FINISHING:
                            op.pop("doc_mm", None)
                            op.pop("woc_mm", None)
                        else:
                            op["woc_mm"] = min(op["woc_mm"], 0.4 * diameter)
                elif strategy not in ("finishing",):
                    for key in ("doc_mm", "woc_mm", "finish_stock"):
                        op.pop(key, None)
            if kind in ("Adaptive3D", "Surface3D", "Waterline3D"):
                op.pop("final_depth", None)  # overwritten from model bounds/reach
            if kind == "Waterline3D":
                op.pop("doc_mm", None)  # overwritten by CAM_FC3D_WATERLINE_STEP
            if op.get("op_type") == "Custom" and op.get("strategy") in ("roughing", "finishing"):
                op["reach_limit_mm"] = _custom_depth_limit(op)
                op.pop("repair_depth_mm", None)
                if op["strategy"] == "finishing":
                    # Mirror _generate_custom: these fields do not drive finishing,
                    # and distinct WOC values can produce the same clamped stepover.
                    op.pop("doc_mm", None)
                    op.pop("finish_stock", None)
                    diameter = float(op["tool_diameter_mm"])
                    op["woc_mm"] = min(max(float(op.get("woc_mm") or 0.1 * diameter) * 3,
                                           0.1 * diameter), 0.45 * diameter)
        return result

    def _apply_fixes(self, req: dict, fixes: list[dict], fused: dict, outcomes: list[dict] | None = None) -> dict:
        if os.environ.get("CAM_REPAIR_FEEDBACK", "0") != "1":
            return self._apply_fixes_legacy(req, fixes, fused)
        outcomes = outcomes if outcomes is not None else []
        for fix in fixes:
            op = next((o for o in req["operations"] if o["step"] == fix.get("step")), None)
            action, value = fix.get("action"), fix.get("value")
            report = {"fix": copy.deepcopy(fix), "status": "rejected", "reason": "unsupported action or value outside existing limits"}
            try:
                invalid_number = isinstance(value, (int, float)) and (isinstance(value, bool) or not math.isfinite(value))
            except OverflowError:  # JSON ints may exceed the float range consumed below.
                invalid_number = True
            if op is None:
                report["reason"] = "step is not present"
            elif invalid_number:
                report["reason"] = "value must be a finite number, not bool"
            else:
                candidate = copy.deepcopy(req)
                target = next(o for o in candidate["operations"] if o["step"] == op["step"])
                custom_depth = op.get("op_type") == "Custom" and op.get("strategy") in ("roughing", "finishing")
                valid = True
                final_depth = None
                if action in ("set_doc_mm", "set_woc_mm", "set_peck_mm"):
                    valid = isinstance(value, (int, float)) and value > 0
                    report["reason"] = "value must be >0 mm; woc is width of cut, not work offset"
                elif action == "set_finish_stock":
                    valid = isinstance(value, (int, float)) and 0 <= value <= 1
                    report["reason"] = "finish_stock must be 0..1 mm"
                elif action == "set_depth_below_top":
                    valid = isinstance(value, (int, float)) and 0 < value <= 60
                    report["reason"] = "depth must be >0 and <=60 mm; native operations also require a known top"
                    top = op.get("start_depth")
                    if top is None:
                        top = self._top_z(op)
                    if valid and not custom_depth and top is None:
                        valid = False
                        report["reason"] = "operation top is unavailable"
                    if valid and not custom_depth:
                        final_depth = round(float(top) - float(value), 3)
                elif action == "set_final_depth" and custom_depth:
                    valid = False
                    report["reason"] = "Custom roughing/finishing does not read final_depth; use set_depth_below_top"
                elif action == "set_final_depth":
                    valid = isinstance(value, (int, float))
                    report["reason"] = "final_depth must be a finite number in mm"
                    if valid:
                        final_depth = float(value)
                elif action == "switch_tool" and any(t.get("tool_number") == op.get("tool_number") and t.get("tool_id") == value for t in req["tools"]):
                    outcomes.append({**report, "status": "no-op", "reason": "requested tool is already selected"})
                    continue
                elif action == "switch_tool":
                    valid = isinstance(value, str) and any(t["id"] == value for t in self.store.tool_library["tools"])
                    report["reason"] = "tool is unavailable in the catalog or cutting_data rejects this tool/process/material"
                elif action == "drop_operation":
                    valid = op["process"] in ("pocket_finishing", "slot_finishing", "profile_finishing", "deburring", "spot_drilling") or fix.get("timeout") is True
                    report["reason"] = "operation is outside the existing drop whitelist"
                if final_depth is not None and not _face_depth_allowed(op, final_depth):
                    valid = False
                    report["reason"] = "depth is below the facing floor"
                if valid:
                    if action == "set_depth_below_top" and custom_depth:
                        target["repair_depth_mm"] = float(value)
                        target.pop("gcode_file", None)
                    elif final_depth is not None:
                        target["final_depth"] = final_depth
                        target.pop("gcode_file", None)
                    else:
                        self._apply_fixes_legacy(candidate, [fix], fused)
                    try:
                        effective_candidate = self._repair_parameters(candidate)
                    except (OverflowError, ValueError):
                        report["reason"] = "value cannot be represented by the effective consumer payload"
                        outcomes.append(report)
                        continue
                    if effective_candidate != self._repair_parameters(req):
                        req.clear()
                        req.update(candidate)
                        report.update(status="applied", reason="effective request changed")
                        report["operation_after"] = copy.deepcopy(next((o for o in req["operations"] if o["step"] == op["step"]), None))
                    elif valid and action in ("set_doc_mm", "set_woc_mm", "set_peck_mm", "set_finish_stock", "set_final_depth", "set_depth_below_top"):
                        report.update(status="no-op", reason="effective payload unchanged after consumer defaults/rounding/clamping, or parameter is not consumed")
                outcomes.append(report)
                continue
            outcomes.append(report)
        return req

    def _apply_fixes_legacy(self, req: dict, fixes: list[dict], fused: dict) -> dict:
        ops = {o["step"]: o for o in req["operations"]}
        lib = {t["id"]: t for t in self.store.tool_library["tools"]}
        mat = self.store.material(fused["material"]["id"])
        for f in fixes:
            o = ops.get(f.get("step"))
            if not o:
                continue
            act, val = f["action"], f.get("value")
            touched = True
            if act == "drop_operation" and (o["process"] in ("pocket_finishing", "slot_finishing", "profile_finishing", "deburring", "spot_drilling")
                                            or f.get("timeout") is True):
                req["operations"] = [x for x in req["operations"] if x["step"] != o["step"]]
                req.setdefault("skipped", []).append({"step": o["step"], "process": o["process"], "reason": f"dropped by repair loop: {f.get('reason')}"})
            elif act in ("set_doc_mm", "set_woc_mm", "set_peck_mm") and isinstance(val, (int, float)) and val > 0:
                o[{"set_doc_mm": "doc_mm", "set_woc_mm": "woc_mm", "set_peck_mm": "peck_mm"}[act]] = float(val)
            elif act == "set_finish_stock" and isinstance(val, (int, float)) and 0 <= val <= 1.0:
                o["finish_stock"] = float(val)
            elif act == "set_final_depth" and isinstance(val, (int, float)):
                if not _face_depth_allowed(o, float(val)):
                    continue
                o["final_depth"] = float(val)
            elif act == "set_depth_below_top" and isinstance(val, (int, float)) and 0 < val <= 60:
                top = o.get("start_depth")
                if top is None:
                    top = self._top_z(o)
                if top is not None:
                    depth = round(float(top) - float(val), 3)
                    if not _face_depth_allowed(o, depth):
                        continue
                    o["final_depth"] = depth
            elif act == "switch_tool" and isinstance(val, str) and val in lib:
                t = dict(lib[val])
                geom = {"diameter": o.get("tool_diameter_mm")} if t.get("diameter_range_mm") else {}
                if geom:
                    t["diameter_mm"] = geom["diameter"]
                cd = cutting_data(t, o["process"], mat, geom, strategy=strategy_for(o["process"], o.get("op_type")))
                if cd.get("error"):
                    touched = False
                    continue
                tn = max(x["tool_number"] for x in req["tools"]) + 1
                req["tools"].append({"tool_number": tn, "tool_id": val, "name": t["name"], "category": t["category"], "diameter_mm": t["diameter_mm"],
                                     "flutes": t.get("flutes", 2), "freecad_shape": t.get("freecad_shape") or "endmill.fcstd",
                                     "shape_params": {**t.get("shape_params", {}), "Diameter": t["diameter_mm"]}, "corner_radius_mm": t.get("corner_radius_mm", 0.0),
                                     "overall_length_mm": t.get("overall_length_mm")})
                tc = max(c["tc"] for c in req["controllers"]) + 1
                req["controllers"].append({"tc": tc, "tool_number": tn, "spindle_rpm": cd["spindle_rpm"], "feed_mm_min": cd["feed_mm_min"],
                                           "plunge_feed_mm_min": cd["plunge_feed_mm_min"]})
                o.update(tool_number=tn, tc=tc, tool_diameter_mm=t["diameter_mm"], doc_mm=cd["doc_mm"], woc_mm=cd["woc_mm"], spindle_rpm=cd["spindle_rpm"],
                         feed_mm_min=cd["feed_mm_min"], plunge_feed_mm_min=cd["plunge_feed_mm_min"], label=f"{o['step']:02d} {o['process']} ({val})",
                         candidates=[c for c in o.get("candidates", []) if c != val])
            else:
                touched = False
            if touched:
                o.pop("gcode_file", None)  # custom toolpaths are regenerated with the new parameters
        return req
