"""A17: projected cavity/column-region overlap, never repair permission.

The cavity bbox is two-dimensional. Neither a contact, an accessible tool pose,
a causal operation owner nor an admissible replacement tool follows from it.
"""
from __future__ import annotations

import re

import numpy as np


def _frame_info(frame):
    return None if frame is None else {"up": frame.up, "translation": frame.t.tolist()}


def points_in_request(points, *, points_frame, request_frame, source_frame=None):
    """Explicit frame conversion; missing frame information fails closed."""
    if request_frame is None:
        raise ValueError("missing request setup frame")
    if points_frame == "original":
        return request_frame.to_setup(points)
    if points_frame != "setup" or source_frame is None:
        raise ValueError("missing/unsupported source points frame")
    if source_frame.up == request_frame.up and np.array_equal(source_frame.t, request_frame.t):
        return points
    return request_frame.to_setup(source_frame.to_original(points))


def bind_native(parsed, operations, tools):
    """Unique explicit program identity, including historical tool rewrites.

    This binding is diagnostic only. It does not replace the metric's legacy
    operation binding or mutate the saved request.
    """
    bound, unbound = [], []
    names = [p.get("name") for p in parsed["ops"]]
    for index, p in enumerate(parsed["ops"]):
        name = p.get("name", "")
        match = re.fullmatch(r"(\d+) (.+) \(([^()]+)\)", name)
        reason = None
        if names.count(name) != 1 or match is None:
            reason = "ambiguous or unsupported native label"
        else:
            step, process, tool_id = match.groups()
            exact = [o for o in operations if o.get("label") == name]
            candidates = exact or [o for o in operations if o.get("step") == int(step)
                                  and o.get("process") == process]
            profiles = [t for t in tools if t.get("tool_id") == tool_id]
            if len(candidates) != 1 or len(profiles) != 1:
                reason = "missing or ambiguous request operation/tool profile"
            else:
                op, tool = candidates[0], profiles[0]
                numbers = [t for t in tools if t.get("tool_number") == tool.get("tool_number")]
                if (op.get("step") != int(step) or op.get("process") != process
                        or len(numbers) != 1 or tool.get("tool_number") is None
                        or not float(tool.get("diameter_mm") or 0) > 0):
                    reason = "inconsistent operation or tool identity"
                elif p.get("tool") not in (None, tool["tool_number"]):
                    reason = "conflicting native T word"
                elif exact and op.get("tool_number") != tool["tool_number"]:
                    reason = "request label/tool conflict"
                else:
                    bound.append({"native_index": index, "native_label": name, "step": int(step),
                                  "process": process, "feature_ids": list(op.get("feature_ids") or []),
                                  "tool_profile": dict(tool),
                                  "binding": "exact_label" if exact else "explicit_tool_id_unique_step_process",
                                  "saved_request_label": op.get("label"),
                                  "saved_request_tool_number": op.get("tool_number")})
        if reason:
            unbound.append({"native_index": index, "native_label": name, "reason": reason})
    return bound, unbound


def observations(regions, labels, grid, target_h, features, parsed, operations, tools, *,
                 points_frame, source_frame, request_frame):
    """Intersect actual retained masks with request cavity bboxes in setup XY.

    Region indices are one-based indices in the unchanged sorted regions list.
    Grid samples are column positions, not exact surface probes. Z is carried
    through explicit frame conversion but does not constrain this 2D relation.
    """
    result = {"source": "retained_column_region_cells", "points_frame": points_frame,
              "source_frame": _frame_info(source_frame), "request_frame": _frame_info(request_frame),
              "relation": "projected_xy_bbox_overlap", "z_scope": "unconstrained", "margin_mm": 0.0,
              "causal_owner": None, "repair_authorized": False, "associations": [], "unbound": [],
              "limits": ["2D cavity bbox only; no path contact, reach or causal operation attribution",
                         "Does not authorize tool switch, operation drop or depth changes"]}
    bound, result["unbound"] = bind_native(parsed, operations, tools)
    if request_frame is None or points_frame != "setup" or source_frame is None:
        result["unbound"].append({"reason": "retained column masks require explicit source and request setup frames"})
        return result
    cavities = {}
    for feature in features:
        if feature.get("type") != "cavity":
            continue
        bbox = feature.get("bbox")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or not np.isfinite(bbox).all() or bbox[0] > bbox[2] or bbox[1] > bbox[3]):
            result["unbound"].append({"cavity_id": feature.get("id"), "reason": "invalid 2D cavity bbox"})
            continue
        fid = feature.get("id")
        if fid in cavities:
            result["unbound"].append({"cavity_id": fid, "reason": "ambiguous cavity id"})
            cavities[fid] = None
        else:
            cavities[fid] = feature
    eligible = [(op, cavities[fid]) for op in bound
                if op["process"] in ("cavity_roughing", "surface_finishing")
                for fid in op["feature_ids"] if cavities.get(fid) is not None]
    # Request-only operations must not masquerade as delivered native paths.
    native_steps = {op["step"] for op in bound}
    for op in operations:
        if (op.get("process") in ("cavity_roughing", "surface_finishing")
                and any(cavities.get(fid) is not None for fid in op.get("feature_ids") or [])
                and op.get("step") not in native_steps):
            result["unbound"].append({"request_step": op.get("step"), "reason": "no uniquely bound native operation"})
    region_ids = {id(region): i + 1 for i, region in enumerate(regions)}
    for kind, (source_labels, emitted) in labels.items():
        from scipy import ndimage
        boxes = ndimage.find_objects(source_labels)
        for label, region in emitted.items():
            sl = boxes[label - 1]
            ii, jj = np.nonzero(source_labels[sl] == label)
            ii, jj = ii + (sl[0].start or 0), jj + (sl[1].start or 0)
            points = np.c_[grid.x0 + jj * grid.res, grid.y0 + ii * grid.res, target_h[ii, jj]]
            points = points_in_request(points, points_frame=points_frame,
                                       source_frame=source_frame, request_frame=request_frame)
            for op, cavity in eligible:
                x0, y0, x1, y1 = cavity["bbox"]
                overlap = ((points[:, 0] >= x0) & (points[:, 0] <= x1)
                           & (points[:, 1] >= y0) & (points[:, 1] <= y1))
                count = int(overlap.sum())
                if count:
                    result["associations"].append({"region_index": region_ids[id(region)], "kind": kind,
                        "feature": region.get("feature"), "cavity_id": cavity["id"],
                        "cavity_bbox_xy": list(cavity["bbox"]), **op,
                        "overlap_cells": count, "region_cells": len(points),
                        "overlap_fraction": count / len(points)})
    return result
