"""Context Fusion Agent: maps 2D annotations onto 3D features (Sec. 3.3, Eq. 3).

S_map = w_t * S_type + w_d * S_dim + w_s * S_spatial
High-confidence mappings are accepted deterministically; ambiguous ones are resolved by the LLM (reasoning mode);
low-confidence ones are flagged for human review. One-to-many mappings are allowed.
"""
from __future__ import annotations

import json

from ..knowledge import get_store
from ..llm import LLM
from ..normalizer import Normalizer
from .feature_extraction import describe_features

W_TYPE, W_DIM, W_SPATIAL = 0.35, 0.5, 0.15
HIGH, MEDIUM = 0.75, 0.45

FUSION_PROMPT = """You are the Context Fusion agent of a manufacturing planning system. You resolve AMBIGUOUS mappings
between drawing annotations and 3D CAD features. For each ambiguous annotation you get the candidate features with their
scores. Use manufacturing heuristics: a thread callout (e.g. M6) maps to holes whose diameter equals the tap drill
(M6 -> 5.0 mm) or, for clearance, to holes of the clearance diameter (6.6 mm); a diameter symbol reinforces
cylindrical features; a radius indicator favors fillets; hole tolerances (H7) map to holes of that nominal diameter;
pocket depth maps to pockets; surface finish maps to the top face unless a location is given. Answer JSON:
{"resolutions": [{"annotation_index": <int>, "feature_ids": [<ids>], "confidence": <0..1>, "reason": <short>}]}"""


class ContextFusionAgent:
    name = "context_fusion"

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.store = get_store()
        self.material_norm = Normalizer(self.store.material_aliases(), llm, "material")

    def run(self, extracted: dict, material_hint: str | None = None) -> dict:
        features = [dict(f) for f in extracted["cad"]["features"]]
        drawing = extracted["drawing"]
        by_id = {f["id"]: f for f in features}
        # material: user hint > drawing note
        mat_text = material_hint or drawing.get("material")
        mat_id, method = self.material_norm.normalize(mat_text)
        material = self.store.material(mat_id)
        mappings, ambiguous = [], []
        for idx, ann in enumerate(drawing.get("annotations", [])):
            scored = sorted(((self._score(ann, f), f["id"]) for f in features), reverse=True)
            scored = [(s, fid) for s, fid in scored if s > 0.2]
            if not scored:
                mappings.append({"annotation_index": idx, "feature_ids": [], "confidence": 0.0, "method": "none", "review": True})
                continue
            best = scored[0][0]
            near = [fid for s, fid in scored if s >= best - 0.08]  # near-tie filter (one-to-many)
            conf = best
            if best >= HIGH and (len(near) == 1 or ann.get("count") in (None, len(near)) or ann["semantic_type"] in ("thread", "hole_diameter", "hole_tolerance")):
                mappings.append({"annotation_index": idx, "feature_ids": near, "confidence": round(conf, 3), "method": "deterministic", "review": False})
            else:
                ambiguous.append({"annotation_index": idx, "annotation": ann, "candidates": [{"feature_id": fid, "score": round(s, 3), "feature": _brief(by_id[fid])} for s, fid in scored[:6]]})
        if ambiguous:
            obj = self.llm.json_call(FUSION_PROMPT, json.dumps({"features": describe_features(features), "ambiguous": ambiguous}, ensure_ascii=False),
                                     agent=self.name, thinking=False)
            resolved = {r.get("annotation_index"): r for r in obj.get("resolutions", []) if isinstance(r, dict)}
            for a in ambiguous:
                r = resolved.get(a["annotation_index"])
                fids = [f for f in (r or {}).get("feature_ids", []) if f in by_id]
                conf = float((r or {}).get("confidence", 0.0)) if fids else 0.0
                mappings.append({"annotation_index": a["annotation_index"], "feature_ids": fids, "confidence": round(conf, 3),
                                 "method": "reasoning", "review": conf < MEDIUM, "reason": (r or {}).get("reason")})
        mappings.sort(key=lambda m: m["annotation_index"])
        # enrich features with the mapped specifications
        for m in mappings:
            ann = drawing["annotations"][m["annotation_index"]]
            for fid in m["feature_ids"]:
                f = by_id[fid]
                f.setdefault("specs", []).append({"text": ann.get("text"), "semantic_type": ann.get("semantic_type"),
                                                  "value": ann.get("value"), "unit": ann.get("unit"), "thread": ann.get("thread"),
                                                  "tolerance_grade": ann.get("tolerance_grade"), "confidence": m["confidence"], "method": m["method"]})
                if ann.get("semantic_type") == "thread" and ann.get("thread") and f["type"] == "hole":
                    spec = self.store.thread(ann["thread"])
                    f["thread"] = spec["thread"] if spec else ann["thread"]
                    f["subtype"] = "thread"
                if ann.get("tolerance_grade"):
                    f["tolerance_grade"] = ann["tolerance_grade"]  # any feature: the route must hold it (capability check)
                    if f["type"] == "hole" and f.get("subtype") != "thread":
                        f["subtype"] = _precision_subtype(f) or f.get("subtype") or "hole"
                if ann.get("semantic_type") == "surface_finish" and ann.get("value"):
                    f["surface_finish_ra_um"] = ann["value"]
        # infer threads for standard tap-drill blind holes when the notes mention threads without mapping
        self._infer_threads(features)
        for f in features:
            f["subtype"] = _default_subtype(f) if f["type"] == "hole" else f.get("subtype") or f["type"]
            f["material"] = mat_id
        fused = {
            "material": {"id": mat_id, "name": material["name"] if material else mat_text, "group": material["group"] if material else "*",
                         "input": mat_text, "normalized_by": method},
            "general_tolerance": drawing.get("general_tolerance"), "surface_finish_ra_um": drawing.get("surface_finish_ra_um"),
            "material_data": material,
            "process_annotations": drawing.get("process_annotations", []), "user_sequence_override": drawing.get("user_sequence_override", []),
            "features": features, "mappings": mappings, "part": extracted["cad"]["part"], "hole_patterns": extracted["cad"].get("hole_patterns", []),
            "needs_review": [m for m in mappings if m["review"]],
        }
        self.llm.trace.log("agent", agent=self.name, mappings=len(mappings), ambiguous=len(ambiguous), material=mat_id)
        return fused

    # ---------------------------------------------------------------- scoring (Eq. 3)
    def _score(self, ann: dict, f: dict) -> float:
        cat, t = ann.get("feature_category"), f["type"]
        s_type = 1.0 if cat == t or (cat == "thread" and t == "hole") else (0.5 if cat in ("unknown", None) else 0.0)
        s_dim = 0.5
        val = ann.get("value")
        if ann.get("thread") and t == "hole":
            spec = self.store.thread(ann["thread"])
            if spec:
                d = f["diameter"]
                s_dim = 1.0 if abs(d - spec["tap_drill_mm"]) < 0.15 else (0.6 if abs(d - spec["clearance_drill_mm"]) < 0.2 else 0.0)
        elif val is not None and t == "hole" and ann.get("semantic_type") in ("hole_diameter", "hole_tolerance"):
            s_dim = 1.0 if abs(f["diameter"] - float(val)) < 0.15 else 0.0
        elif val is not None and t in ("pocket", "slot", "step") and ann.get("semantic_type") == "pocket_depth":
            s_dim = 1.0 if abs(float(f.get("depth", -1)) - float(val)) < 0.15 else 0.0
        elif val is not None and t == "chamfer":
            s_dim = 1.0 if abs(f["size"] - float(val)) < 0.15 else 0.2
        elif ann.get("semantic_type") == "surface_finish" and t == "planar_face":
            s_dim = 0.9 if f.get("role") == "top" else 0.4
        s_spatial = 0.5
        ctx = (ann.get("spatial_context") or "").lower()
        if ctx:
            if "top" in ctx and (f.get("role") == "top" or (t == "hole" and f.get("from_top_of_part", 0) < 1e-3)):
                s_spatial = 1.0
            elif "blind" in ctx and t == "hole":
                s_spatial = 1.0 if not f["through"] else 0.0
            elif "through" in ctx and t == "hole":
                s_spatial = 1.0 if f["through"] else 0.0
            elif "corner" in ctx and t == "hole":
                s_spatial = 0.8
        if val is None and not ann.get("thread"):  # no dimensional info: weight semantics and spatial cues more heavily
            return 0.6 * s_type + 0.1 * s_dim + 0.3 * s_spatial
        return W_TYPE * s_type + W_DIM * s_dim + W_SPATIAL * s_spatial

    def _infer_threads(self, features: list[dict]) -> None:
        """A blind hole modelled at a standard tap-drill size is a thread CANDIDATE (reported, not machined as a thread
        unless the drawing says so)."""
        for f in features:
            if f["type"] == "hole" and not f.get("thread") and not f["through"]:
                spec = self.store.tap_drill_for_hole(float(f["diameter"]), tol=0.05)
                if spec:
                    f["thread_candidate"] = spec["thread"]


def _default_subtype(f: dict) -> str:
    if f["type"] == "hole":
        if f.get("thread"):
            return "thread"
        if f.get("subtype") in ("counterbore", "countersink"):
            return f["subtype"]
        return _precision_subtype(f) or "hole"
    return f["type"]


def _precision_subtype(f: dict) -> str | None:
    """A hole tighter than drilling holds (IT11, Ra 6.3) is reamed (IT7, Ra 0.8) or, for IT6 or a large diameter, bored."""
    digits = "".join(c for c in str(f.get("tolerance_grade") or "") if c.isdigit())
    it = int(digits) if digits else None
    ra = float(f.get("surface_finish_ra_um") or 99)
    if (it is not None and it <= 10) or ra < 6.3:
        return "large_precision_hole" if (it is not None and it <= 6) or float(f.get("diameter") or 0) >= 20 else "precision_hole"
    return None


def _brief(f: dict) -> dict:
    return {k: v for k, v in f.items() if k in ("type", "diameter", "depth", "through", "size", "corner_radius", "role", "z", "top", "radius", "angle_deg")}
