"""A36 experimental ready-round comparator; no production hook or safety admission.

The off arm calls the supplied legacy comparator verbatim. The on arm consumes
only actual cumulative material already available at selection time. Final
material is checked separately after execution, never used as a future oracle.
Evidence attestations must come from a reviewed native replay, not saved stats.
"""
from __future__ import annotations

from decimal import Decimal
import math
import re


CAP_MM3 = 1000  # Q24, 2026-11-06: total 1 cm3, not an incremental budget.
EXCHANGE = 20
GATES = ("native_health", "inputs", "material_binding", "gcode_safety",
         "complete", "attribution", "other_gates")
COMMON = ("source_sha256", "inputs_sha256", "metric_sha256", "samples_sha256",
          "settings_sha256", "budget_sha256")


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("metric must be numeric, not bool/string")
    try:
        if not math.isfinite(value):
            raise ValueError("metric must be finite")
    except OverflowError as exc:
        raise ValueError("unrepresentable metric") from exc
    return Decimal(str(value))


def known_cap_violation(stats: dict) -> bool:
    """A rounded accepted gouge value can prove rejection, never acceptance.

    The accepted display rounds cm3 to 3 decimals. Its lower bound includes
    half a display unit; no guessed raw value or case-ID exception is needed.
    """
    try:
        return (isinstance(stats, dict) and stats.get("metric") == "exact"
                and (_number(stats.get("gouge_cm3")) - Decimal("0.0005")) * 1000 > CAP_MM3)
    except (ValueError, TypeError, OverflowError):
        return False


def validate(evidence: dict, *, phase: str, _reference: bool = False) -> tuple[Decimal, Decimal]:
    """Fail closed; return accepted within/area percentages after total cap.

    gouge_mm3 is the unrounded sum(w * raw_gouge), INCLUDING expected probes,
    with the SAME ladder/tol/cap as services.verify.surface_metric. It is a
    normal-probe integral, not a physical removed-body volume. Rounded display
    gouge_cm3 alone, or dev (which can hide simultaneous gouge), is insufficient.
    """
    if not isinstance(evidence, dict):
        raise ValueError("evidence unavailable")
    if _reference and phase != "round":
        raise ValueError("only a measured round may be a comparison reference")
    if evidence.get("phase") != phase or evidence.get("scope") != "actual-cumulative":
        raise ValueError("actual cumulative phase required; ideal later setups forbidden")
    if phase == "round" and evidence.get("available_before_selection") is not True:
        raise ValueError("future or retrospective metrics cannot select a live round")
    gates = evidence.get("gates")
    if not isinstance(gates, dict) or any(gates.get(k) is not True for k in GATES):
        raise ValueError("hard/health/provenance/material gate unavailable or failed")
    if type(evidence.get("hard")) is not int or evidence["hard"] != 0:
        raise ValueError("hard issues/errors are not exchangeable")
    lineage = evidence.get("lineage")
    if not isinstance(lineage, dict) or any(
        not isinstance(lineage.get(k), str) or re.fullmatch(r"[0-9a-f]{64}", lineage[k]) is None
        for k in (*COMMON, "program_sha256", "raw_gouge_sha256", "prefix_sha256")
    ):
        raise ValueError("immutable source/input/metric/program/raw/prefix hashes required")
    stats = evidence.get("stats")
    if (not isinstance(stats, dict) or stats.get("metric") != "exact"
            or stats.get("later_setups") not in (None, "programmed")):
        raise ValueError("accepted exact metric unavailable")
    within = _number(stats.get("surface_within_tol_pct"))
    area = _number(stats.get("gouge_area_pct"))
    if not 0 <= within <= 100 or not 0 <= area <= 100:
        raise ValueError("percentages outside 0..100")
    volume = _number(evidence.get("gouge_mm3"))
    display = _number(stats.get("gouge_cm3"))
    if volume < 0 or display < 0:
        raise ValueError("negative gouge")
    if round(float(volume) / 1000, 3) != float(display):
        raise ValueError("raw mm3/display cm3 metric mismatch")
    if not _reference and volume > CAP_MM3:
        raise ValueError("total cumulative gouge exceeds 1 cm3 (1000 mm3)")
    return within, area


def better_round(a: tuple, b: tuple, *, legacy, enabled: bool = False,
                 candidate: dict | None = None, baseline: dict | None = None) -> bool:
    """Explicit default-off isolated arm, using the existing round rank contract.

    No issue-count tie breaker can buy damage. Both deltas are percentage
    points of the accepted metric, without extra slack or fitted coefficients.
    """
    if enabled is False:
        return legacy(a, b)
    if enabled is not True:
        return False
    try:
        wa, ga = validate(candidate, phase="round")
        # The unselected measured reference is not an admitted round. Its raw
        # metrics/context must still validate, but only the candidate buys cap.
        wb, gb = validate(baseline, phase="round", _reference=True)
        if any(candidate["lineage"][k] != baseline["lineage"][k] for k in COMMON):
            return False
        # Reject accidental mixing with ranks from a different saved round.
        for rank, ev, within, area in ((a, candidate, wa, ga), (b, baseline, wb, gb)):
            if rank[0] != ev["hard"] or _number(rank[1]) != area or _number(rank[2][0]) != -within:
                return False
        dw, dg = wa - wb, ga - gb
        return dw > 0 and dw >= EXCHANGE * max(Decimal(0), dg)
    except (ValueError, TypeError, KeyError, IndexError, OverflowError):
        return False


def final_gate(evidence: dict) -> bool:
    """Post-run admission prerequisite only; never an input to better_round."""
    try:
        validate(evidence, phase="final")
        return True
    except (ValueError, TypeError, KeyError, OverflowError):
        return False
