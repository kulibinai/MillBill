"""AAGNet runner for the contract in aagnet.py (runs in the AAGNet environment, not in the harness env).

    $AAGNET_PYTHON aagnet_predict.py part.step out.json
    env: AAGNET_HOME (clone of https://github.com/whjdark/AAGNet), AAGNET_WEIGHTS (default: weights/weight_on_MFInstseg.pth
         in AAGNET_HOME), AAGNET_DEVICE (cpu | cuda, default cpu)

Follows the repository's own inference path (utils/qt5_visualization.py, featureRecog): AAGExtractor with
feature_lists/all.json -> load_one_graph -> standardization with weights/attr_stat.json -> AAGNetSegmentor 'full'
(seg_out, inst_out, bottom_out) -> instance proposals from the thresholded face-face matrix -> class by summed face
logits -> bottom faces. Face index = node index of the AAG + 1 (occwl face order = TopExp face order); the centroid
and area of every face are written so that the harness re-matches faces with FreeCAD.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

CLASSES = ["chamfer", "through_hole", "triangular_passage", "rectangular_passage", "6sides_passage",
           "triangular_through_slot", "rectangular_through_slot", "circular_through_slot",
           "rectangular_through_step", "2sides_through_step", "slanted_through_step", "Oring", "blind_hole",
           "triangular_pocket", "rectangular_pocket", "6sides_pocket", "circular_end_pocket",
           "rectangular_blind_slot", "v_circular_end_blind_slot", "h_circular_end_blind_slot",
           "triangular_blind_step", "circular_blind_step", "rectangular_blind_step", "round", "stock"]
INST_THR, BOTTOM_THR, EPS = 0.5, 0.5, 1e-6


def face_table(step_path: str) -> list[dict]:
    """index (1-based, TopExp order), centroid and area of every face, in the STEP's own units and frame."""
    from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
    from OCC.Core.GProp import GProp_GProps
    from OCC.Core.STEPControl import STEPControl_Reader
    from OCC.Extend.TopologyUtils import TopologyExplorer
    r = STEPControl_Reader()
    if r.ReadFile(step_path) != 1:
        raise RuntimeError(f"cannot read {step_path}")
    r.TransferRoots()
    rows = []
    for i, f in enumerate(TopologyExplorer(r.OneShape()).faces(), start=1):
        p = GProp_GProps()
        brepgprop_SurfaceProperties(f, p)
        c = p.CentreOfMass()
        rows.append({"index": i, "centroid": [c.X(), c.Y(), c.Z()], "area": p.Mass()})
    return rows


def run_network(step_path: str, home: str):
    import torch
    sys.path.insert(0, home)
    from dataset.AAGExtractor import AAGExtractor
    from models.inst_segmentors import AAGNetSegmentor
    from utils.data_utils import load_json_or_pkl, load_one_graph, load_statistics, standardization
    device = os.environ.get("AAGNET_DEVICE", "cpu")
    weights = os.environ.get("AAGNET_WEIGHTS") or os.path.join(home, "weights", "weight_on_MFInstseg.pth")
    model = AAGNetSegmentor(arch="AAGNetGraphEncoder", num_classes=len(CLASSES),
                            edge_attr_dim=12, node_attr_dim=10, edge_attr_emb=64, node_attr_emb=64,
                            edge_grid_dim=0, node_grid_dim=7, edge_grid_emb=0, node_grid_emb=64,
                            num_layers=3, delta=2, mlp_ratio=2, drop=0., drop_path=0.,
                            head_hidden_dim=64, conv_on_edge=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model = model.to(device).eval()
    aag = AAGExtractor(step_path, load_json_or_pkl(os.path.join(home, "feature_lists", "all.json"))).process()
    sample = standardization(load_one_graph(step_path, aag), load_statistics(os.path.join(home, "weights", "attr_stat.json")))
    with torch.no_grad():
        seg_out, inst_out, bottom_out = model(sample["graph"].to(device))
    logits = seg_out.cpu().numpy()
    adj = (inst_out[0].sigmoid() > INST_THR).cpu().numpy().astype("int32")
    bottom = (bottom_out.sigmoid() > BOTTOM_THR).cpu().numpy().reshape(-1)
    return logits, adj, bottom, os.path.basename(weights)


def proposals(adj) -> list[list[int]]:
    """Instance proposals exactly as in the repository: greedy rows of the face-face matrix, each face used once."""
    used = np.zeros(adj.shape[0], dtype=bool)
    out = set()
    for i, row in enumerate(adj):
        if used[i] or row.sum() <= EPS:
            continue
        p = {j for j, v in enumerate(row) if v and not used[j]}
        for j in p:
            used[j] = True
        if p:
            out.add(frozenset(p))
    return [sorted(p) for p in out]


def main() -> None:
    step, out = sys.argv[1], sys.argv[2]
    home = os.environ["AAGNET_HOME"]
    rows = face_table(step)
    logits, adj, bottom, model_id = run_network(step, home)
    if len(logits) != len(rows):
        raise RuntimeError(f"network returned {len(logits)} faces, the STEP has {len(rows)}")
    for r, lg in zip(rows, logits):
        r.update(label=CLASSES[int(np.argmax(lg))], instance=-1, bottom=False)
    k = 0
    for inst in proposals(adj):
        cls = int(np.argmax(sum(logits[f] for f in inst)))
        if CLASSES[cls] == "stock":
            continue
        for f in inst:  # the instance class wins over the per-face label
            rows[f].update(label=CLASSES[cls], instance=k, bottom=bool(bottom[f]))
        k += 1
    with open(out, "w") as fh:
        json.dump({"faces": rows, "model": model_id, "instances": k}, fh)


if __name__ == "__main__":
    main()
