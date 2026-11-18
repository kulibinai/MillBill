"""Feature Extraction Agent (Tier 2): 3D B-Rep feature recognition (FreeCAD rules, or AAGNet + B-Rep measurement)
and 2D drawing-note parsing (LLM), run concurrently and merged (synchronisation node, Fig. 2 of the paper)."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..llm import LLM
from ..services import FreeCADRunner

ANNOTATION_PROMPT = """You are the 2D-drawing analysis service of a manufacturing planning system.
The user gives free-text drawing notes for a machined part. Extract every manufacturing annotation into JSON:
{
  "material": <string or null>,
  "general_tolerance": <string or null, e.g. "ISO 2768-mK">,
  "surface_finish_ra_um": <number or null, overall Ra requirement>,
  "process_annotations": [<explicit process instructions, e.g. "anodize", "heat treat 40 HRC", "no tapping">],
  "annotations": [
    {"text": <original fragment>, "semantic_type": <"thread"|"hole_diameter"|"hole_tolerance"|"pocket_depth"|"surface_finish"|"flatness"|"position"|"chamfer"|"counterbore"|"countersink"|"quantity"|"other">,
     "interpretation": <what it means for manufacturing>, "feature_category": <"hole"|"thread"|"pocket"|"slot"|"planar_face"|"chamfer"|"outer_profile"|"unknown">,
     "value": <number or null>, "unit": <"mm"|"um"|"deg"|"in"|null>, "thread": <e.g. "M6", "M8x1", "1/4-20 UNC" or null>, "tolerance_grade": <e.g. "H7" or null>,
     "count": <int or null>, "spatial_context": <where on the part, or null>}
  ],
  "user_sequence_override": [<ordered process names only if the user dictates a complete process sequence, else empty>]
}
Rules: quote fragments verbatim in "text"; do not invent annotations that are not in the notes; "M6 tapped holes" is a
thread annotation with feature_category "thread"; "H7" is a tolerance grade; Ra values are in um; inch values keep unit "in"."""


_GROUPS_CACHE: dict = {}  # (STEP path, mtime) -> (groups, info) of the learned recogniser


class FeatureExtractionAgent:
    name = "feature_extraction"

    def __init__(self, llm: LLM, freecad: FreeCADRunner | None = None, recognizer: str = "auto", log=lambda m: None) -> None:
        self.llm = llm
        self.freecad = freecad or FreeCADRunner()
        self.recognizer, self.log = recognizer, log

    def run(self, cad_path: str | Path, notes: str = "", workdir: str | Path | None = None, up: str = "+Z") -> dict:
        cad_path = str(Path(cad_path).resolve())
        with ThreadPoolExecutor(max_workers=2) as ex:
            f3d = ex.submit(self.extract_3d, cad_path, workdir, up)
            f2d = ex.submit(self.extract_2d, notes)
            geo, ann = f3d.result(), f2d.result()
        self.llm.trace.log("agent", agent=self.name, features=len(geo.get("features", [])), annotations=len(ann.get("annotations", [])),
                           recogniser=geo.get("recogniser"))
        return {"cad": geo, "drawing": ann}

    def extract_3d(self, cad_path: str, workdir, up: str = "+Z") -> dict:
        req: dict = {"path": cad_path, "up": up}
        groups, info = self.learned_groups(cad_path, workdir)
        if groups:
            req["groups"] = groups
        geo = self.freecad.run("extract_features.py", req, workdir)
        if info:
            geo["recogniser_info"] = info
        return geo

    def learned_groups(self, cad_path: str, workdir) -> tuple[list | None, dict | None]:
        """Face groups of the learned recogniser (AAGNet) for this STEP, or (None, None) for rule-based recognition."""
        if self.recognizer not in ("auto", "aagnet"):
            return None, None
        from ..recognition import aagnet
        if not aagnet.available():
            if self.recognizer == "aagnet":
                raise RuntimeError("CAM_RECOGNIZER=aagnet but AAGNET_CMD / AAGNET_PYTHON + AAGNET_HOME + AAGNET_WEIGHTS are not set")
            return None, None
        key = (str(Path(cad_path).resolve()), Path(cad_path).stat().st_mtime)
        if key in _GROUPS_CACHE:
            return _GROUPS_CACHE[key]
        try:
            pred = aagnet.predict(cad_path, Path(workdir or ".") / "aagnet")
            table = self.freecad.run("face_table.py", {"path": cad_path}, workdir)["faces"]
            groups, info = aagnet.to_groups(pred, table)
        except Exception as e:  # noqa: BLE001
            if self.recognizer == "aagnet":
                raise
            self.log(f"[feature_extraction] AAGNet unavailable ({type(e).__name__}: {str(e)[:200]}): rule-based recognition")
            return None, None
        self.log(f"[feature_extraction] AAGNet: {info['instances']} instances "
                 f"{sorted({g['class'] for g in groups})}, unmatched faces {info['unmatched_faces']}")
        _GROUPS_CACHE[key] = (groups, info)
        return groups, info

    def extract_2d(self, notes: str) -> dict:
        if not notes.strip():
            return {"material": None, "annotations": [], "process_annotations": [], "user_sequence_override": [],
                    "general_tolerance": None, "surface_finish_ra_um": None}
        obj = self.llm.json_call(ANNOTATION_PROMPT, notes, agent=self.name + ".drawing", thinking=False)
        for k in ("annotations", "process_annotations", "user_sequence_override"):
            obj.setdefault(k, [])
        obj["raw_notes"] = notes
        return obj


def describe_features(features: list[dict]) -> str:
    """Compact, LLM-friendly rendering of the recognised features."""
    lines = []
    for f in features:
        t = f["type"]
        if t == "hole":
            extra = ""
            if f.get("subtype") == "counterbore":
                extra = f" counterbore d{f.get('counterbore_diameter')}x{f.get('counterbore_depth')}"
            elif f.get("subtype") == "countersink":
                extra = f" countersink d{f.get('countersink_diameter')}x{f.get('countersink_angle')}deg"
            lines.append(f"{f['id']}: hole d={f['diameter']} mm depth={f['depth']} mm {'through' if f['through'] else 'blind'}{extra}"
                         f"{' (' + str(f.get('thread')) + ')' if f.get('thread') else ''}{' ' + f['tolerance_grade'] if f.get('tolerance_grade') else ''}"
                         f" at {f['top'][:2]} top_z={f.get('top_z')}{'' if f.get('vertical', True) else ' NOT vertical (axis ' + str(f.get('axis')) + ')'}")
        elif t in ("pocket", "slot", "step"):
            lines.append(f"{f['id']}: {t} {f['size'][0]}x{f['size'][1]} mm depth={f.get('depth')} mm floor_z={f.get('z')} corner_r={f.get('corner_radius')}"
                         f"{' open' if f.get('open') else ' closed'}{' through' if f.get('through') else ''}")
        elif t == "cavity":
            lines.append(f"{f['id']}: cavity (not explained by recognised features) about {f['size'][0]}x{f['size'][1]} mm, depth up to {f['depth']} mm, "
                         f"{round(f['residual_mm3'] / 1000, 1)} cm3 = {round(100 * f['residual_share'])} % of this side's removal; "
                         f"largest tools reaching it: rough D{f['rough_tool_d']:g}, finish D{f['finish_tool_d']:g}")
        elif t == "planar_face":
            lines.append(f"{f['id']}: planar face ({f.get('role')}) {f['size'][0]}x{f['size'][1]} mm z={f['z']}"
                         f"{' needs facing' if f.get('needs_facing') else ''}")
        elif t in ("chamfer", "sloped_face"):
            lines.append(f"{f['id']}: {t} {f['size']} mm x {f['angle_deg']} deg")
        elif t == "fillet":
            lines.append(f"{f['id']}: fillet r={f['radius']} mm")
        elif t == "outer_profile":
            cap = f" contour tool <= D{f['profile_max_d']:g}" if f.get("profile_max_d") and float(f["profile_max_d"]) < 20 else ""
            lines.append(f"{f['id']}: outer profile {f['size']} {'rectangular' if f.get('rectangular') else 'freeform'} ({len(f.get('faces', []))} side faces){cap}")
        else:
            lines.append(f"{f['id']}: {t} {json.dumps({k: v for k, v in f.items() if k not in ('id', 'type', 'faces')}, default=str)[:200]}")
    return "\n".join(lines)
