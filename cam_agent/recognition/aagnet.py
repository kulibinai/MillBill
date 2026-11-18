"""Learned machining-feature recognition with AAGNet (Wu et al., 2023, MIT licence, https://github.com/whjdark/AAGNet).

AAGNet segments the faces of a B-Rep into machining-feature instances (MFInstSeg classes: holes, pockets, passages,
slots, steps, chamfers, rounds, O-rings) and marks each instance's bottom faces. The harness uses the network only for
WHICH faces form a feature; every parameter is then measured on the exact B-Rep by fc_scripts/extract_features.py.

The model runs in its own environment (torch + dgl + occwl / pythonocc), usually on the GPU server, through a command
with a fixed contract:

    $AAGNET_CMD <part.step> <out.json>          (default: "$AAGNET_PYTHON recognition/aagnet_predict.py")

    out.json = {"faces": [{"index": <1-based face index>, "centroid": [x, y, z], "area": <mm2>,
                           "label": "<class name>", "instance": <int or -1>, "bottom": <bool>}],
                "model": "<weights id>"}

Face indices of OpenCascade's TopExp face map and FreeCAD's Shape.Faces normally agree for a STEP read the same way;
`to_groups` re-matches every face by centroid and area against FreeCAD's faces, so a different ordering cannot mix
features up (unmatched faces are dropped and reported).
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def available() -> bool:
    return bool(os.environ.get("AAGNET_CMD") or (os.environ.get("AAGNET_PYTHON") and os.environ.get("AAGNET_HOME") and os.environ.get("AAGNET_WEIGHTS")))


def predict(step_path: str | Path, workdir: str | Path, timeout: float = 600.0) -> dict:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    out = workdir / "aagnet_prediction.json"
    cmd = os.environ.get("AAGNET_CMD")
    if cmd:
        args = shlex.split(cmd) + [str(step_path), str(out)]
    else:
        args = [os.environ["AAGNET_PYTHON"], str(HERE / "aagnet_predict.py"), str(step_path), str(out)]
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"AAGNet failed (rc={proc.returncode}): {(proc.stdout + proc.stderr)[-2000:]}")
    return json.loads(out.read_text())


def to_groups(prediction: dict, fc_faces: list[dict], tol_mm: float = 0.05) -> tuple[list[dict], dict]:
    """Instance groups [{"class", "faces": [FreeCAD 1-based indices], "bottom": [...]}] from the network output.
    fc_faces: [{"index", "centroid", "area"}] of the same STEP in FreeCAD (from tessellate/face listing)."""
    import numpy as np
    C = np.array([f["centroid"] for f in fc_faces], float) if fc_faces else np.zeros((0, 3))
    A = np.array([f["area"] for f in fc_faces], float) if fc_faces else np.zeros(0)
    groups: dict[int, dict] = {}
    unmatched = 0
    for f in prediction.get("faces", []):
        if f.get("label") in (None, "stock") or int(f.get("instance", -1)) < 0:
            continue
        idx = None
        if len(C):
            d = np.linalg.norm(C - np.asarray(f["centroid"], float), axis=1)
            k = int(d.argmin())
            if d[k] <= tol_mm + 1e-3 * max(1.0, float(np.abs(C).max())) and abs(A[k] - float(f["area"])) <= 0.01 * max(A[k], 1.0) + 1e-3:
                idx = int(fc_faces[k]["index"])
        if idx is None:
            unmatched += 1
            continue
        g = groups.setdefault(int(f["instance"]), {"class": f["label"], "faces": [], "bottom": []})
        g["faces"].append(idx)
        if f.get("bottom"):
            g["bottom"].append(idx)
    return list(groups.values()), {"instances": len(groups), "unmatched_faces": unmatched, "model": prediction.get("model")}


if __name__ == "__main__":  # quick manual check: python -m cam_agent.recognition.aagnet part.step
    print(json.dumps(predict(sys.argv[1], Path(sys.argv[1]).parent / "aagnet"), indent=1)[:4000])
