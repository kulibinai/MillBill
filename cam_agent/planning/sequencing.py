"""Operation sequencing and sequence validation (standard CAPP practice).

Order = a topological sort of hard constraints, taking the step with the best priority whenever several are free:
  * hard constraints (knowledge/data/dependency_rules.json, GOST 3.1109 route principles): setup first; datum facing
    before machining; all roughing before any finishing; hole making after roughing; heat treatment between roughing
    (with fixing holes and tapped threads) and finishing (with reaming, boring, grinding); deburring after all material
    removal, then cleaning, inspection, coating; per-feature pairs (spot -> drill -> chamfer/countersink -> ream/bore/tap,
    rough -> finish); nesting (a container pocket / slot / step is roughed before a deeper feature that opens from its floor).
  * priority ("standard" mode): process phase (datum, roughing, finishing, hole making, chamfers, reaming, threads, deburr),
    then the tool (one tool loaded once per phase; larger tools first), then nesting depth, then a short path.
  * priority ("element" mode, kept for comparisons): finish one element (pocket/slot with its holes) before the next.

The same constraints are used by `validate`, so a sequence produced here never fails its own validation, and an LLM
proposal is only reordered where it breaks a hard constraint (`repair`), never rebuilt from scratch.
"""
from __future__ import annotations

import heapq
import math

from ..knowledge import get_store

FINAL = ("deburring", "cleaning", "inspection", "coating", "marking")
NOT_FORMING = ("spot_drilling", "chamfering", "countersinking", "deburring")  # do not produce the size / finish of a surface
NON_MACHINING = ("setup", "fixturing", "cleaning", "inspection", "coating", "plating", "heat_treatment", "stress_relief", "marking")


# ----------------------------------------------------------------------------------------------- geometry helpers
def _bbox(f: dict) -> list | None:
    bb = f.get("bbox")
    if bb and len(bb) >= 4:
        return [float(v) for v in bb[:4]]
    if f.get("type") == "hole" and f.get("top") and f.get("diameter"):
        r = float(f["diameter"]) / 2
        return [f["top"][0] - r, f["top"][1] - r, f["top"][0] + r, f["top"][1] + r]
    return None


def _area(bb: list) -> float:
    return max(0.0, (bb[2] - bb[0]) * (bb[3] - bb[1]))


def _overlap(a: list, b: list) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def _circle_hits_box(cx: float, cy: float, r: float, bb: list) -> bool:
    dx = max(bb[0] - cx, 0.0, cx - bb[2])
    dy = max(bb[1] - cy, 0.0, cy - bb[3])
    return dx * dx + dy * dy <= r * r + 1e-6


def anchor(f: dict) -> tuple[float, float]:
    bb = _bbox(f)
    if bb:
        return ((bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2)
    return (0.0, 0.0)


def contains(outer: dict, inner: dict) -> bool:
    """True when `inner` opens from the floor of `outer` (a deeper pocket/slot, or a hole that starts at/below its floor).
    Partial overlap counts: the inner feature cannot be machined before the container is cleared."""
    rules = get_store().deps.get("nesting", {})
    if outer.get("type") not in rules.get("container_types", ["pocket", "slot", "step"]) or outer.get("id") == inner.get("id"):
        return False
    ob = _bbox(outer)
    if not ob or outer.get("z") is None:
        return False
    oz = float(outer["z"])
    if inner.get("type") in ("pocket", "slot", "step"):
        ib = _bbox(inner)
        if not ib or inner.get("z") is None:
            return False
        return (float(inner["z"]) < oz - 0.2 and _overlap(ib, ob) >= float(rules.get("min_overlap_mm2", 1.0))
                and _area(ib) < 0.95 * _area(ob))
    if inner.get("type") == "hole":
        top = inner.get("top") or []
        if len(top) < 2 or not inner.get("diameter"):
            return False
        top_z = inner.get("top_z", top[2] if len(top) > 2 else 0.0)
        bottom = float(top_z) - float(inner.get("depth") or 0.0)
        return _circle_hits_box(float(top[0]), float(top[1]), float(inner["diameter"]) / 2, ob) and bottom < oz - 0.2
    return False


def nesting(feats: list[dict]) -> dict[str, str]:
    """inner feature id -> its innermost container id."""
    parent = {}
    for inner in feats:
        cs = [o for o in feats if contains(o, inner)]
        if cs:
            parent[inner["id"]] = min(cs, key=lambda o: _area(_bbox(o) or [0, 0, 0, 0]))["id"]
    return parent


def _depth(fid: str, parent: dict) -> int:
    n, seen = 0, set()
    while fid in parent and fid not in seen:
        seen.add(fid)
        fid = parent[fid]
        n += 1
    return n


# ----------------------------------------------------------------------------------------------- constraints
def phase(process: str) -> int:
    return int(get_store().templates["process_catalog"].get(process, {}).get("phase", 50))


def is_machining(process: str) -> bool:
    cat = get_store().templates["process_catalog"].get(process, {})
    return bool(cat.get("tooling")) and process not in ("grinding",)


def _fids(step: dict) -> list[str]:
    return [x for x in (step.get("feature_ids") or []) if isinstance(x, str)]


def _shares_feature(a: dict, b: dict) -> bool:
    fa, fb = set(_fids(a)), set(_fids(b))
    return bool(fa & fb) or "*" in fa or "*" in fb


def group(token: str) -> set[str]:
    """Processes named by a rule token: a process, '@group' of dependency_rules.json, '@cutting' (tooling processes outside
    the final group, grinding included), '@machining' (@cutting without grinding) or '*' (every catalog process)."""
    store = get_store()
    catalog = store.templates["process_catalog"]
    groups = store.deps.get("process_groups", {})
    if token == "*":
        return set(catalog)
    if token in ("@cutting", "@machining"):
        cut = {p for p, c in catalog.items() if c.get("tooling") and p not in groups.get("final", FINAL)}
        return cut if token == "@cutting" else {p for p in cut if is_machining(p)}
    if token.startswith("@"):
        return set(groups.get(token[1:], []))
    return {token}


def constraints(seq: list[dict], feats: list[dict]) -> list[tuple[int, int, str]]:
    """Hard precedence edges (i must come before j) between the steps of `seq`, all from dependency_rules.json:
    global rules between processes (datum first, all roughing before finishing, holes after roughing, heat treatment
    between the stages, final operations last), per-feature pairs (spot -> drill -> chamfer/countersink -> ream/bore/tap,
    rough -> finish) and nesting (a container is roughed before the features that open from its floor)."""
    deps = get_store().deps
    rules = [(group(r["before"]), group(r["after"]), r.get("reason", f"{r['before']} before {r['after']}")) for r in deps.get("global_precedence", [])]
    pairs = {(p[0], p[1]) for p in deps["precedence"]}
    parent = nesting(feats)
    edges = []
    for i, a in enumerate(seq):
        pa = a["process"]
        for j, b in enumerate(seq):
            if i == j:
                continue
            pb = b["process"]
            why = None
            if pa != pb:
                why = next((reason for before, after, reason in rules if pa in before and pb in after), None)
                if why is None and (pa, pb) in pairs and _shares_feature(a, b):
                    why = f"{pa} before {pb} (same feature)"
            if why is None:
                # nesting: the container's roughing (first milling phase) precedes every step of a contained feature
                fa, fb = _fids(a), _fids(b)
                if phase(pa) <= 20 and any(parent.get(x) in fa for x in fb) and not any(x in fa for x in fb):
                    why = "container before nested feature"
            if why:
                edges.append((i, j, why))
    return edges


# ----------------------------------------------------------------------------------------------- ordering
def order(seq: list[dict], feats: list[dict], mode: str = "standard", tool_of: dict | None = None) -> list[dict]:
    """Constraint-respecting order with standard-practice priorities. tool_of: step index -> (tool_id, diameter) once
    the tools are known (otherwise a provisional tool key is derived from the process and feature)."""
    if not seq:
        return []
    by_id = {f["id"]: f for f in feats}
    parent = nesting(feats)
    edges = constraints(seq, feats)
    succ: dict[int, list[int]] = {i: [] for i in range(len(seq))}
    indeg = [0] * len(seq)
    for i, j, _ in edges:
        succ[i].append(j)
        indeg[j] += 1
    roots: dict[str, int] = {}

    def element(st: dict) -> int:
        for fid in _fids(st):
            r = fid
            while r in parent:
                r = parent[r]
            if r in by_id:
                return roots.setdefault(r, len(roots))
        return 10 ** 6

    def tool_rank(i: int) -> tuple:
        st = seq[i]
        if tool_of and i in tool_of and tool_of[i]:
            tid, d = tool_of[i]
            return (-(float(d or 0)), str(tid))
        fs = [by_id[x] for x in _fids(st) if x in by_id]
        d = max([float(f.get("diameter") or 0) for f in fs] + [0.0])
        return (-d, st["process"])

    last_xy = [0.0, 0.0]

    def dist(i: int) -> float:
        fs = [by_id[x] for x in _fids(seq[i]) if x in by_id]
        if not fs:
            return 0.0
        x, y = anchor(fs[0])
        return math.hypot(x - last_xy[0], y - last_xy[1])

    def key(i: int) -> tuple:
        st = seq[i]
        nest = min([_depth(x, parent) for x in _fids(st)] or [0])
        if mode == "element" and phase(st["process"]) > 10 and st["process"] not in FINAL:
            return (1, element(st), phase(st["process"]), nest, tool_rank(i), dist(i), i)
        return (phase(st["process"]), tool_rank(i), nest, round(dist(i), 1), i)

    ready = [i for i in range(len(seq)) if indeg[i] == 0]
    out = []
    while ready:
        i = min(ready, key=key)  # priorities depend on the current position (path), so re-evaluate each pick
        ready.remove(i)
        out.append(i)
        fs = [by_id[x] for x in _fids(seq[i]) if x in by_id]
        if fs:
            last_xy[:] = list(anchor(fs[0]))
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                ready.append(j)
    if len(out) < len(seq):  # a cycle (contradictory LLM input): keep the rest in their phase order
        rest = sorted((i for i in range(len(seq)) if i not in out), key=lambda i: (phase(seq[i]["process"]), i))
        out += rest
    return [seq[i] for i in out]


def repair(seq: list[dict], feats: list[dict]) -> tuple[list[dict], list[str]]:
    """Minimal fix of a given order: keep it, move only the steps that break a hard constraint (stable topological sort
    with the given position as the priority). Returns (sequence, list of moves)."""
    edges = constraints(seq, feats)
    broken = [(i, j, why) for i, j, why in edges if i > j]
    if not broken:
        return seq, []
    succ: dict[int, list[int]] = {i: [] for i in range(len(seq))}
    indeg = [0] * len(seq)
    for i, j, _ in edges:
        succ[i].append(j)
        indeg[j] += 1
    ready = [i for i in range(len(seq)) if indeg[i] == 0]
    heapq.heapify(ready)
    out = []
    while ready:
        i = heapq.heappop(ready)
        out.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready, j)
    out += [i for i in range(len(seq)) if i not in out]
    moves = [f"step {seq[j]['process']} {_fids(seq[j])} moved after {seq[i]['process']} {_fids(seq[i])}: {why}" for i, j, why in broken]
    return [seq[i] for i in out], moves


# ----------------------------------------------------------------------------------------------- validation
def _it(grade) -> int | None:
    digits = "".join(c for c in str(grade or "") if c.isdigit())
    return int(digits) if digits else None


def validate(seq: list[dict], feats: list[dict], material_group: str, fused: dict | None = None, mode: str = "standard",
             assignments: list[dict] | None = None, final_setup: bool = True) -> dict:
    """final_setup False: an intermediate setup of a multi-setup job, whose deburring / cleaning / inspection are done
    after the last setup (the mandatory final steps are not required here)."""
    store = get_store()
    deps, catalog = store.deps, store.templates["process_catalog"]
    names = [s["process"] for s in seq]
    critical, warnings = [], []
    for n in names:
        if n not in catalog:
            critical.append({"check": "invalid_process", "message": f"'{n}' is not a catalog process"})
    for i, j, why in constraints(seq, feats):
        if i > j:
            critical.append({"check": "dependency", "message": f"step {i + 1} {names[i]} must precede step {j + 1} {names[j]} ({why})",
                             "processes": [names[i], names[j]]})
    for combo in deps["invalid_combinations"]:
        firsts = [k for k, n in enumerate(names) if n == combo["first"]]
        if not firsts:
            continue
        later_set = group("@machining") if combo["later"] == "*machining*" else group(combo["later"])
        if any(k > min(firsts) for k, n in enumerate(names) if n in later_set):
            critical.append({"check": "invalid_combination", "message": combo["message"], "processes": [combo["first"], combo["later"]]})
    for st in seq:
        for r in deps["requires"].get(st["process"], []):
            if not any(o["process"] == r and _shares_feature(o, st) for o in seq):
                critical.append({"check": "dependency", "message": f"'{st['process']}' on {_fids(st)} requires '{r}' on the same feature"})
    compat = store.templates["material_process_compatibility"].get(material_group, {})
    for n in set(names):
        if n in compat.get("incompatible", []):
            critical.append({"check": "material_compatibility", "message": f"'{n}' is not applicable to {material_group}: {compat.get('notes', '')}"})
        elif n in compat.get("discouraged", []):
            warnings.append({"check": "material_compatibility", "message": f"'{n}' is discouraged for {material_group}: {compat.get('notes', '')}"})
    covered: dict[str, set] = {}
    for st in seq:
        for fid in _fids(st):
            covered.setdefault(fid, set()).add(st["process"])
    star = covered.get("*", set())
    cav_ids = [f["id"] for f in feats if f.get("type") == "cavity"]
    cav_procs = set().union(*(covered.get(c, set()) for c in cav_ids)) if cav_ids else set()
    ra_global = (fused or {}).get("surface_finish_ra_um")
    for f in feats:
        t = f["type"]
        procs = covered.get(f["id"], set()) | star
        if cav_ids and t in ("pocket", "slot", "step", "boss", "outer_profile", "fillet", "planar_face"):
            procs |= cav_procs  # the cavity's height-map roughing / finishing machines these surfaces
        if t == "hole" and not f.get("vertical", True):
            warnings.append({"check": "accessibility", "message": f"{f['id']} axis is not vertical: another setup or 4/5-axis", "feature": f["id"]})
            continue
        # accuracy and finish: only the processes that cut the feature's own surface count (economic accuracy of the method);
        # spot drilling, entry chamfers, countersinks and deburring do not size or finish a bore or a face
        forming = [p for p in procs if p not in NOT_FORMING and catalog.get(p, {}).get("tooling")]
        it = _it(f.get("tolerance_grade"))
        best_it = min([catalog.get(p, {}).get("tolerance_it") or 99 for p in forming] or [99])
        if it is not None and best_it > it:
            need = "reaming or boring" if t == "hole" else "a finishing pass or grinding"
            critical.append({"check": "capability", "message": f"{f['id']} {f['tolerance_grade']}: the planned processes hold IT{best_it if best_it < 99 else '-'}, "
                             f"{need} needed", "feature": f["id"]})
        ra = f.get("surface_finish_ra_um")
        best_ra = min([float(catalog.get(p, {}).get("ra_um") or 99) for p in forming] or [99.0])
        if ra and float(ra) < best_ra - 1e-9:
            warnings.append({"check": "capability", "message": f"{f['id']} Ra {ra} um: the planned processes reach Ra {best_ra if best_ra < 99 else '-'} um "
                             f"(finishing / reaming / boring / grinding needed)", "feature": f["id"]})
        # completeness: every recognised feature has the processes that make it
        if t == "hole" and f.get("thread") and not procs & {"tapping", "thread_milling"}:
            critical.append({"check": "missing_process", "message": f"{f['id']} is threaded ({f['thread']}) but has no tapping / thread milling", "feature": f["id"]})
        if t == "hole" and not procs & {"drilling", "boring"}:
            critical.append({"check": "missing_process", "message": f"{f['id']} hole has no drilling step", "feature": f["id"]})
        if t in ("pocket", "slot") and not procs & {"pocket_milling", "slot_milling", "cavity_roughing"}:
            critical.append({"check": "missing_process", "message": f"{f['id']} {t} has no milling step", "feature": f["id"]})
        if t == "step" and not procs & {"step_milling", "profile_milling", "cavity_roughing", "pocket_milling"}:
            critical.append({"check": "missing_process", "message": f"{f['id']} step has no milling step", "feature": f["id"]})
        if t in ("boss", "outer_profile") and f.get("needs_profile", True) and not procs & {"profile_milling", "cavity_roughing"}:
            warnings.append({"check": "missing_process", "message": f"{f['id']} {t} has no contour milling", "feature": f["id"]})
        if t == "planar_face" and f.get("role") == "top" and f.get("needs_facing", True) and not procs & {"facing", "grinding", "surface_finishing"}:
            warnings.append({"check": "missing_process", "message": f"{f['id']} datum face is not faced", "feature": f["id"]})
        if t == "chamfer" and not procs & {"chamfering", "countersinking"}:
            warnings.append({"check": "missing_process", "message": f"{f['id']} modelled chamfer has no chamfering step", "feature": f["id"]})
        if t == "fillet" and not procs & {"fillet_milling", "surface_finishing"}:
            warnings.append({"check": "missing_process", "message": f"{f['id']} modelled fillet has no fillet milling", "feature": f["id"]})
    if ra_global:
        best = min([float(catalog.get(n, {}).get("ra_um") or 99) for n in names if n not in NOT_FORMING] or [99])
        if float(ra_global) < best - 1e-9:
            warnings.append({"check": "capability", "message": f"general Ra {ra_global} um is below what the planned processes reach (Ra {best} um): "
                             "finishing passes, grinding or polishing needed"})
    for p in deps["mandatory_final"] if final_setup else []:
        if p not in names:
            warnings.append({"check": "missing_process", "message": f"mandatory final step '{p}' missing"})
    machining = [k for k, n in enumerate(names) if is_machining(n)]
    stats = {"steps": len(seq), "machining_steps": len(machining)}
    if assignments:
        tools = [a.get("tool_id") for a in assignments if a.get("tool_id")]
        stats["tool_changes"] = sum(1 for k in range(1, len(tools)) if tools[k] != tools[k - 1])
        stats["distinct_tools"] = len(set(tools))
        avoidable = max(0, stats["tool_changes"] - (len(set(tools)) - 1)) if tools else 0
        stats["avoidable_tool_changes"] = avoidable  # a tool loaded more than once
        if avoidable:
            warnings.append({"check": "practice", "message": f"{avoidable} avoidable tool change(s): a tool is loaded more than once"})
    return {"critical": critical, "warnings": warnings, "valid": not critical, "stats": stats}


def number(seq: list[dict], setup: int | None = None) -> list[dict]:
    return [{"step": i + 1, "setup": setup or s.get("setup", 1), **{k: v for k, v in s.items() if k not in ("step", "setup")}} for i, s in enumerate(seq)]


def regroup_by_tool(seq: list[dict], assignments: list[dict], feats: list[dict]) -> tuple[list[dict], list[dict], int]:
    """Merge the steps that use the same tool inside one phase block (first appearance keeps its place), as long as no
    hard constraint breaks. The order chosen by the planner or the LLM is otherwise kept. Returns (sequence, assignments,
    number of tool changes saved)."""
    by_step = {a["step"]: a for a in assignments}

    def tool(st):
        a = by_step.get(st["step"], {})
        return (a.get("tool_id"), a.get("tool_diameter_mm")) if a.get("tool_id") else None

    def changes(s):
        ts = [tool(x) for x in s if tool(x)]
        return sum(1 for k in range(1, len(ts)) if ts[k] != ts[k - 1])

    out, k = [], 0
    while k < len(seq):
        ph = phase(seq[k]["process"])
        block = []
        while k < len(seq) and phase(seq[k]["process"]) == ph:
            block.append(seq[k])
            k += 1
        first: dict = {}
        for pos, st in enumerate(block):
            first.setdefault(tool(st) or ("_", pos), pos)
        prev = next((tool(x) for x in reversed(out) if tool(x)), None)
        if prev in first:  # the tool already in the spindle continues into this phase
            first[prev] = -1
        out += sorted(block, key=lambda st, b=block: (first[tool(st) or ("_", b.index(st))], b.index(st)))
    if any(i > j for i, j, _ in constraints(out, feats)) or changes(out) >= changes(seq):
        return seq, assignments, 0
    saved = changes(seq) - changes(out)
    renum = {st["step"]: i + 1 for i, st in enumerate(out)}
    new_seq = [{**st, "step": renum[st["step"]]} for st in out]
    new_asg = sorted(({**a, "step": renum.get(a["step"], a["step"])} for a in assignments), key=lambda a: a["step"])
    return new_seq, new_asg, saved
