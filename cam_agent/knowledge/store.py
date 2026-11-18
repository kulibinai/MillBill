"""Knowledge base backing the Table-1 tool registry of the paper, plus the shop data the deterministic core needs.

Sources (priority high -> low, as in Fig. 7): text (RAG) > tabular CSV > decision tree > SQL DB > knowledge graph > material DB.
Shop data: machine profile, tool library (ISO 513 groups), thread table (ISO 261 / ASME B1.1), ISO 2768 general tolerances.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sqlite3
import threading
from functools import lru_cache
from pathlib import Path

from ..config import KB_DIR

SOURCE_PRIORITY = ["rag_text", "tabular", "decision_tree", "sql_database", "knowledge_graph", "material_database"]

_WORD = re.compile(r"[a-z0-9.]+")


def _tokens(s: str) -> list[str]:
    return _WORD.findall(s.lower())


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9./#-]", "", str(s).lower())


class KnowledgeStore:
    def __init__(self, kb_dir: Path = KB_DIR) -> None:
        self.kb_dir = kb_dir
        mats = json.loads((kb_dir / "materials.json").read_text(encoding="utf-8"))
        self.materials = mats["materials"]
        self.iso_groups, self.engagement = mats["iso_groups"], mats["engagement"]
        self.cutting = {k: mats[k] for k in ("finish_allowance_mm", "chip_thinning", "long_reach")}
        rules = json.loads((kb_dir / "rules.json").read_text(encoding="utf-8"))
        self.rules, self.thread_specs = rules["rules"], rules["thread_specs"]
        self.general_tolerances = rules["general_tolerances_iso2768"]
        self.counterbores = rules["counterbores"]
        self.reaming_allowance = rules["reaming_allowance_mm"]["table"]
        self.boring_allowance = float(rules["boring_allowance_mm"])
        with open(kb_dir / "feature_material.csv", encoding="utf-8") as f:
            self.feature_material = list(csv.DictReader(f))
        self.kg = json.loads((kb_dir / "kg.json").read_text(encoding="utf-8"))["triples"]
        self.tool_library = json.loads((kb_dir / "tool_library.json").read_text(encoding="utf-8"))
        self.templates = json.loads((kb_dir / "process_templates.json").read_text(encoding="utf-8"))
        self.deps = json.loads((kb_dir / "dependency_rules.json").read_text(encoding="utf-8"))
        self.machine = json.loads((kb_dir / "machine.json").read_text(encoding="utf-8"))
        self.guideline_chunks = self._load_chunks(kb_dir / "guidelines.md")
        self._db = self._build_db()
        self._lock = threading.Lock()
        self._threads_by_key = {}
        for t in self.thread_specs:
            self._threads_by_key[_key(t["thread"])] = t
            if t["series"] == "M":  # "M8" is also "M8x1.25"
                self._threads_by_key[_key(f"{t['thread']}x{t['pitch']:g}")] = t
            if t["series"] in ("UNC", "UNF"):  # "1/4-20" without the series suffix (the TPI already tells UNC from UNF)
                self._threads_by_key[_key(t["thread"].split()[0])] = t

    # ------------------------------------------------------------ builders
    @staticmethod
    def _load_chunks(path: Path) -> list[dict]:
        chunks, title, buf = [], "", []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                if buf:
                    chunks.append({"title": title, "text": " ".join(buf).strip()})
                title, buf = line[3:].strip(), []
            elif line.strip() and not line.startswith("# "):
                buf.append(line.strip())
        if buf:
            chunks.append({"title": title, "text": " ".join(buf).strip()})
        for c in chunks:
            c["_tokens"] = _tokens(c["title"] + " " + c["text"])
        return chunks

    def _build_db(self) -> sqlite3.Connection:
        db = sqlite3.connect(":memory:", check_same_thread=False)
        db.execute("CREATE TABLE rules (rule_id TEXT, feature_type TEXT, material_group TEXT, parameter TEXT, "
                   "min_value REAL, max_value REAL, unit TEXT, severity TEXT, standard TEXT, message TEXT)")
        db.executemany("INSERT INTO rules VALUES (?,?,?,?,?,?,?,?,?,?)",
                       [(r["rule_id"], r["feature_type"], r["material_group"], r["parameter"], r["min_value"], r["max_value"],
                         r["unit"], r["severity"], r["standard"], r["message"]) for r in self.rules])
        db.execute("CREATE TABLE thread_specs (thread TEXT, standard TEXT, series TEXT, major_mm REAL, pitch REAL, tap_drill_mm REAL, clearance_drill_mm REAL, minor_mm REAL)")
        db.executemany("INSERT INTO thread_specs VALUES (?,?,?,?,?,?,?,?)",
                       [(t["thread"], t["standard"], t["series"], t["major_mm"], t["pitch"], t["tap_drill_mm"], t["clearance_drill_mm"], t["minor_mm"])
                        for t in self.thread_specs])
        db.commit()
        return db

    # ------------------------------------------------------------ materials
    def material(self, material_id: str | None) -> dict | None:
        for m in self.materials:
            if m["id"] == material_id:
                return m
        return None

    def material_group(self, material_id: str | None) -> str:
        m = self.material(material_id)
        return m["group"] if m else "*"

    def material_groups(self) -> list[str]:
        return sorted({m["group"] for m in self.materials})

    def material_aliases(self) -> dict[str, list[str]]:
        return {m["id"]: m["aliases"] + [m["name"]] for m in self.materials}

    # ------------------------------------------------------------ threads / tolerances
    def thread(self, designation: str | None) -> dict | None:
        """ISO metric coarse/fine ("M8", "M8x1", "M10 x 1.25") and unified ("1/4-20 UNC", "#10-32") designations."""
        if not designation:
            return None
        k = _key(designation).replace("unc", "").replace("unf", "").rstrip("-")
        if k in self._threads_by_key:
            return self._threads_by_key[k]
        m = re.match(r"m(\d+(?:\.\d+)?)x(\d+(?:\.\d+)?)", k)
        if m:
            for t in self.thread_specs:
                if t["series"] in ("M", "MF") and abs(t["major_mm"] - float(m.group(1))) < 1e-6 and abs(t["pitch"] - float(m.group(2))) < 1e-6:
                    return t
        m = re.match(r"m(\d+(?:\.\d+)?)", k)
        if m:
            return self._threads_by_key.get(f"m{float(m.group(1)):g}")
        return None

    def tap_drill_for_hole(self, diameter: float, tol: float = 0.06) -> dict | None:
        """Thread whose tap drill is this hole diameter (a modelled tap-drill hole)."""
        best = min(self.thread_specs, key=lambda t: abs(t["tap_drill_mm"] - diameter))
        return best if abs(best["tap_drill_mm"] - diameter) <= tol else None

    def iso2768(self, nominal_mm: float, cls: str = "m") -> float | None:
        gt = self.general_tolerances
        for upper, dev in zip(gt["ranges_mm"], gt.get(cls.lower(), gt["m"])):
            if nominal_mm <= upper:
                return dev
        return None

    def reaming_stock(self, d: float) -> float:
        for upper, stock in self.reaming_allowance:
            if d <= upper:
                return float(stock)
        return float(self.reaming_allowance[-1][1])

    # ------------------------------------------------------------ tools (KR registry, Table 1)
    def sql_retrieval(self, feature_type: str = "*", material_group: str = "*", parameter: str | None = None,
                      query: str | None = None) -> dict:
        """Quantitative constraints from the relational rules DB (read-only SELECT only)."""
        with self._lock:
            if query:
                q = query.strip().rstrip(";")
                if not q.lower().startswith("select"):
                    return {"error": "only SELECT queries are allowed", "tables": ["rules", "thread_specs"]}
                try:
                    cur = self._db.execute(q)
                    cols = [d[0] for d in cur.description]
                    return {"rows": [dict(zip(cols, r)) for r in cur.fetchmany(50)], "query": q}
                except sqlite3.Error as e:
                    return {"error": str(e), "tables": {"rules": "rule_id, feature_type, material_group, parameter, min_value, max_value, unit, severity, standard, message",
                                                        "thread_specs": "thread, standard, series, major_mm, pitch, tap_drill_mm, clearance_drill_mm, minor_mm"}}
            sql = "SELECT * FROM rules WHERE (feature_type=? OR feature_type='*') AND (material_group=? OR material_group='*')"
            args: list = [feature_type, material_group]
            if parameter:
                sql += " AND parameter LIKE ?"
                args.append(f"%{parameter}%")
            cur = self._db.execute(sql, args)
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        threads = self.thread_specs if feature_type in ("thread", "*") else []
        return {"rows": rows, "thread_specs": threads, "count": len(rows)}

    def tabular_retrieval(self, feature_type: str | None = None, material_group: str | None = None) -> dict:
        """Feature-material compatibility, machinability scores and coating recommendations (CSV)."""
        rows = [r for r in self.feature_material
                if (not feature_type or r["feature_type"] == feature_type)
                and (not material_group or r["material_group"] == material_group)]
        return {"rows": rows[:40], "count": len(rows), "columns": list(self.feature_material[0].keys())}

    def text_retrieval(self, query: str, top_k: int = 3) -> dict:
        """Keyword (BM25) retrieval over the DFM / process guideline corpus."""
        q = _tokens(query)
        if not q:
            return {"passages": []}
        n = len(self.guideline_chunks)
        avgdl = sum(len(c["_tokens"]) for c in self.guideline_chunks) / max(n, 1)
        df = {t: sum(1 for c in self.guideline_chunks if t in c["_tokens"]) for t in set(q)}
        scored = []
        for c in self.guideline_chunks:
            score = 0.0
            for t in q:
                tf = c["_tokens"].count(t)
                if tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len(c["_tokens"]) / avgdl))
            if score > 0:
                scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return {"passages": [{"title": c["title"], "text": c["text"], "score": round(s, 2)} for s, c in scored[:top_k]]}

    def decision_tree_evaluation(self, feature_type: str, material_group: str = "*", **dims) -> dict:
        """Rule-based pass/fail checks of a feature's dimensions against the applicable rules."""
        derived = dict(dims)
        d, depth = dims.get("diameter"), dims.get("depth")
        if d and depth:
            derived["depth_to_diameter"] = round(depth / d, 2)
        w = dims.get("width") or dims.get("min_width")
        if w and depth:
            derived["depth_to_width"] = round(depth / w, 2)
            derived.setdefault("min_width", w)
        cr = dims.get("corner_radius")
        if cr is not None and depth:
            derived["corner_radius_to_depth"] = round(cr / depth, 3)
        checks, violations = [], []
        for r in self.rules:
            if r["feature_type"] not in (feature_type, "*") or r["material_group"] not in (material_group, "*"):
                continue
            v = derived.get(r["parameter"])
            if v is None or isinstance(v, (list, dict, str)):
                continue
            ok = r["min_value"] <= float(v) <= r["max_value"]
            item = {"rule_id": r["rule_id"], "parameter": r["parameter"], "value": v, "range": [r["min_value"], r["max_value"]],
                    "unit": r["unit"], "pass": ok, "severity": r["severity"], "message": r["message"], "standard": r["standard"]}
            checks.append(item)
            if not ok:
                violations.append(item)
        applicable = sorted({r["parameter"] for r in self.rules if r["feature_type"] in (feature_type, "*")})
        return {"feature_type": feature_type, "evaluated": derived, "checks": checks, "violations": violations,
                "applicable_parameters": applicable}

    def knowledge_graph_query(self, entity: str, relation: str | None = None, depth: int = 1) -> dict:
        """Relationships between features, processes, materials, tools, coatings and cycles."""
        entity_k = entity.lower().replace(" ", "_")
        seen, frontier, edges = {entity_k}, [entity_k], []
        for _ in range(max(1, min(depth, 2))):
            nxt = []
            for e in frontier:
                for s, r, o in self.kg:
                    if (s == e or o == e) and (not relation or r == relation.upper()):
                        edges.append({"subject": s, "relation": r, "object": o})
                        other = o if s == e else s
                        if other not in seen:
                            seen.add(other)
                            nxt.append(other)
            frontier = nxt
        uniq = [dict(t) for t in {tuple(e.items()) for e in edges}]
        return {"entity": entity_k, "edges": sorted(uniq, key=lambda x: (x["relation"], x["subject"], x["object"])),
                "relations": sorted({s for _, s, _ in self.kg})}

    def material_database_lookup(self, material: str) -> dict:
        """ISO group, machinability, hardness, Kienzle data, dimensional limits, cutting speeds and coolant for a material."""
        from ..normalizer import Normalizer  # local import to avoid cycles
        mid, method = Normalizer(self.material_aliases(), None, "material").normalize(material)
        m = self.material(mid)
        if not m:
            return {"error": f"unknown material '{material}'", "known": [x["id"] for x in self.materials]}
        return {"normalized_by": method, **m, "iso_group_info": self.iso_groups.get(m["iso_group"])}

    # ------------------------------------------------------------ tools (PS / TS)
    def process_capability_lookup(self, process: str | None = None) -> dict:
        cat = self.templates["process_catalog"]
        if process:
            p = cat.get(process)
            return {"process": process, **(p or {"error": "unknown process"}), "requires": self.deps["requires"].get(process, [])}
        return {"processes": cat}

    def sequence_retrieval(self, part_category: str = "machined_prismatic", feature_types: list[str] | None = None) -> dict:
        cats = self.templates["part_categories"]
        cat = cats.get(part_category) or cats["machined_prismatic"]
        fp = self.templates["feature_processes"]
        needed = []
        for ft in feature_types or []:
            for p in fp.get(ft, []):
                if p not in needed:
                    needed.append(p)
        return {"part_category": part_category, "template": cat["template"], "feature_processes": {ft: fp.get(ft, []) for ft in (feature_types or [])},
                "needed_processes": needed, "mandatory_final": self.deps["mandatory_final"],
                "phases": {p: v.get("phase") for p, v in self.templates["process_catalog"].items()}}

    def tool_library_lookup(self, category: str | None = None, iso_group: str | None = None, diameter_mm: float | None = None) -> dict:
        tools = []
        for t in self.tool_library["tools"]:
            if category and t["category"] != category:
                continue
            if iso_group and iso_group not in t["iso"]:
                continue
            if diameter_mm is not None:
                rng = t.get("diameter_range_mm")
                if rng and not (rng[0] <= diameter_mm <= rng[1]):
                    continue
                if not rng and t.get("diameter_mm") and abs(t["diameter_mm"] - diameter_mm) > 1e-6:
                    continue
            tools.append({k: v for k, v in t.items() if k not in ("freecad_shape", "shape_params")})
        return {"tools": tools, "count": len(tools), "categories": sorted({t["category"] for t in self.tool_library["tools"]}),
                "process_to_tool_category": self.tool_library["process_to_tool_category"]}


@lru_cache(maxsize=1)
def get_store() -> KnowledgeStore:
    return KnowledgeStore()
