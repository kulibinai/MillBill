"""A20 observations only: exact probes have no causal operation owner.

Sparse occupied 3D cells preserve separated surfaces at the same XY. Face IDs
are evidence, not connectivity: adjacent B-Rep faces may share one cluster.
Cell connectivity is descriptive (sampling gaps can split a physical defect).
"""
from __future__ import annotations

import numpy as np


def clusters(points, cell_mm):
    """Deterministic 6-connected occupied cells; O(N log N), no dense 3D grid."""
    if not len(points):
        return np.empty(0, np.int32)
    cells, inverse = np.unique(np.floor(points / cell_mm).astype(np.int64), axis=0, return_inverse=True)
    lookup = {tuple(c): i for i, c in enumerate(cells)}
    parent = np.arange(len(cells))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, c in enumerate(cells):
        for axis in range(3):
            neighbor = c.copy()
            neighbor[axis] -= 1
            j = lookup.get(tuple(neighbor))
            if j is not None:
                a, b = root(i), root(j)
                parent[max(a, b)] = min(a, b)
    roots = np.array([root(i) for i in range(len(cells))])
    _, ids = np.unique(roots, return_inverse=True)
    return (ids[inverse] + 1).astype(np.int32)


def observations(sd, samples, setup_points, frame, grid, regions, label_receipt, tol):
    P, w = sd["points"], sd["w"]
    model_points = P if frame is not None else samples.get("model_points")
    face = np.asarray(samples.get("face", np.full(len(P), -1)), np.int32)
    gi, gj, ok = grid.index(setup_points[:, 0], setup_points[:, 1])
    cell_mm = max(2 * tol, grid.res)
    arrays = {"points_setup": setup_points, "normals_metric_frame": sd["normals"], "w": w,
              "face_zero_based": face, "excess_mm": sd["exact_excess"], "gouge_mm": sd["exact_gouge"]}
    if model_points is not None:
        arrays["points_model"] = model_points
    coverage = (samples.get("info") or {}).get("brep_coverage")
    result = {"source": "exact_surface_probes", "owner": "unknown", "last_cut_by_step": None,
              "repair_authorized": False, "needs_review": False, "cell_mm": cell_mm,
              "connectivity": "6-connected occupied 3D setup cells; no normal filter or volume cutoff",
              "point_ids": "zero-based artifact row indices; cluster label 0 means not bad for this kind",
              "face_ids": "artifact zero-based B-Rep face indices; JSON IDs one-based",
              "artifact": "exact_issues.npz", "brep_coverage": coverage,
              "model_frame_available": model_points is not None,
              "limits": ["No causal operation owner: overlapping column region is spatial evidence only",
                         "Probe integral is area times normal probe depth, not removed-body volume",
                         "Sampling gaps may split clusters; occupied cell adjacency may join nearby surfaces",
                         "Earlier setup checks include ideal future setups; they are not final job defects"], "kinds": {}}
    if coverage and coverage.get("incomplete"):
        result["limits"].append("A38 incomplete B-Rep sampling: missing faces are not represented or repaired")
    for kind in ("gouge", "excess"):
        depth = sd["exact_" + kind]
        bad = depth >= tol - 1e-9
        ids = np.flatnonzero(bad)
        labels = np.zeros(len(P), np.int32)
        labels[ids] = clusters(setup_points[ids], cell_mm)
        column = np.zeros(len(P), np.int32)
        source_labels, emitted = label_receipt[kind]
        mapping = np.zeros(int(source_labels.max()) + 1, np.int32)
        by_identity = {id(r): i + 1 for i, r in enumerate(regions)}
        for label, reg in emitted.items():
            mapping[label] = by_identity[id(reg)]
        column[ok] = mapping[source_labels[gi[ok], gj[ok]]]
        arrays["cluster_" + kind] = labels
        arrays["column_region_" + kind] = column
        arrays["expected"] = sd["expected"]

        def amount(idx):
            return {"count": len(idx), "area_mm2": float(w[idx].sum()),
                    "probe_integral_mm3": float((depth[idx] * w[idx]).sum())}

        areas = np.bincount(labels[ids], weights=w[ids])
        order = sorted(range(1, len(areas)), key=lambda k: (-areas[k], k))
        summaries = []
        # Full memberships remain in the artifact even when JSON is bounded.
        sorted_ids = ids[np.argsort(labels[ids], kind="stable")]
        bounds = np.searchsorted(labels[sorted_ids], np.arange(len(areas) + 1))
        for label in order[:24]:
            ii = sorted_ids[bounds[label]:bounds[label + 1]]
            overlap = ii[column[ii] > 0]
            unique_regions = np.unique(column[overlap])
            summaries.append({"cluster_id": label, **amount(ii), "max_probe_mm": float(depth[ii].max()),
                              "bbox_setup": [setup_points[ii].min(0).tolist(), setup_points[ii].max(0).tolist()],
                              "bbox_model": None if model_points is None else [model_points[ii].min(0).tolist(), model_points[ii].max(0).tolist()],
                              "face_ids": (np.unique(face[ii][face[ii] >= 0])[:24] + 1).tolist(),
                              "face_count": len(np.unique(face[ii][face[ii] >= 0])),
                              "example_point_ids": ii[:8].tolist(), "owner": "unknown", "repair_authorized": False,
                              "column_overlap": amount(overlap), "column_region_ids": unique_regions[:24].tolist(),
                              "column_region_count": len(unique_regions), "uncovered": amount(ii[column[ii] == 0])})
        result["kinds"][kind] = {**amount(ids), "expected": amount(ids[sd["expected"][ids]]),
                                  "cluster_count": len(areas) - 1 if len(ids) else 0,
                                  "clusters_shown": len(summaries), "clusters": summaries,
                                  "column_overlap": amount(ids[column[ids] > 0]),
                                  "uncovered": amount(ids[column[ids] == 0]),
                                  "uncovered_unexpected": amount(ids[(column[ids] == 0) & ~sd["expected"][ids]]),
                                  "column_region_ids": "one-based indices in unchanged regions; exact cell intersection, no dilation"}
        result["needs_review"] |= bool(result["kinds"][kind]["uncovered_unexpected"]["count"])
    return result, arrays


def save(path, sim):
    if "exact_issue_arrays" in sim:
        np.savez_compressed(path, **sim["exact_issue_arrays"])


def repair_observation(obs):
    """Small LLM view; complete evidence stays in verification and NPZ."""
    result = {k: obs[k] for k in ("source", "owner", "last_cut_by_step", "repair_authorized", "needs_review", "artifact", "limits")}
    result["kinds"] = {}
    for kind, item in obs["kinds"].items():
        result["kinds"][kind] = {k: item[k] for k in ("count", "area_mm2", "probe_integral_mm3", "cluster_count", "uncovered", "uncovered_unexpected")}
        result["kinds"][kind]["clusters"] = [{k: c[k] for k in ("cluster_id", "count", "area_mm2", "bbox_setup", "bbox_model", "example_point_ids")}
                                               for c in item["clusters"][:2]]
    result["repair_limits"] = "Unknown exact owner does not authorize operation drop or depth changes; independent operation evidence is required"
    return result


def safe_fixes(fixes, ver):
    """Do not turn a new spatial observation into a destructive attribution."""
    independently_owned = {r.get("last_cut_by_step") for r in (ver.get("deviation") or {}).get("errors", [])}
    independently_owned.update(e.get("step") for e in ver.get("op_errors", []))
    independently_owned.discard(None)
    destructive = {"drop_operation", "set_final_depth", "set_depth_below_top"}
    return [f for f in fixes if f.get("action") not in destructive or f.get("step") in independently_owned]
