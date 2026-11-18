"""A23: shared shield/verifier padding from this request's milling paths.

The selection changes only the dexel lattice. Meshes and exact surface queries
continue to use the complete tool/request lists. Unknown mappings keep legacy
padding rather than silently excluding a potentially large cutter.
"""
import os


# Current CAMGenerationAgent._op_for / _generate_custom and _plan_rest mappings.
NATIVE_PATHS = frozenset(("Pocket", "Adaptive", "Profile", "Adaptive3D", "Surface3D", "Deburr"))
CUSTOM_PATHS = frozenset(("roughing", "finishing", "feature", "chamfer", "counterbore", "thread_mill", "rest"))


def grid_tool_selection(tools, operations=None):
    enabled = os.environ.get("CAM_PATH_TOOL_GRID", "0") == "1"
    chosen, reason = tools, "flag_off_legacy"
    if enabled:
        selected, unknown = set(), False
        by_no = {t.get("tool_number"): t for t in tools}
        for op in operations or []:
            kind = op.get("op_type")
            if kind == "MillFace" or op.get("process") == "facing" or kind == "Drilling":
                continue
            eligible = kind in NATIVE_PATHS or (kind == "Custom" and op.get("strategy") in CUSTOM_PATHS)
            if not eligible or op.get("tool_number") not in by_no:
                unknown = True
                continue
            selected.add(op["tool_number"])
        if unknown:
            reason = "unknown_operation_or_tool_legacy"
        elif not selected:
            reason = "no_eligible_path_tools_legacy"
        else:
            chosen = [t for t in tools if t.get("tool_number") in selected]
            reason = "path_operations"
    radius = max([float(t.get("diameter_mm") or 0) / 2 for t in chosen] + [1.0])
    return radius, {"enabled": enabled, "reason": reason, "tool_numbers": [t.get("tool_number") for t in chosen],
                    "max_radius_mm": radius, "pad_mm": radius + 1.0}


def grid_receipt(selection, grid):
    return {**selection, "x0": grid.x0, "y0": grid.y0, "res": grid.res, "nx": grid.nx, "ny": grid.ny}
