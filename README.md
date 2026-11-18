# MillBill: A Machining Agent with Geometric Feedback

MillBill turns a boundary-representation part, its milling stock and its material into G-code for indexed three-axis machining. A locally hosted open model revises a standard-practice plan of operations, tools and cutting parameters. A geometric harness writes the toolpaths, simulates material removal and returns surface deviations so the plan can be refined. The model never writes coordinates or G-code. Setups are chosen by what the available cutters can reach, not by visibility alone.

- Paper: [arXiv:2611.22565](https://arxiv.org/abs/2611.22565)

## What is here

| Path        | Contents                                                                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset/`  | GrabCAD parts paired with milling stocks. The 100 parts scored in the paper are listed in [`dataset/evaluation.json`](dataset/evaluation.json). The full description is in [`dataset/README.md`](dataset/README.md). |
| `cam_agent/` | The harness. Feature recognition, reach-aware setup planning, the planning agents, toolpaths, G-code, material-removal simulation and geometric feedback. Run with `python -m cam_agent`. |
| `examples/` | `make_demo_part.py`: a FreeCAD script that builds a small demo bracket and writes its STEP file.                                                                          |
| `tests/`    | Offline checks of planning, tool selection, CAM generation and the FreeCAD runner. They do not machine the evaluation set.                                                |
| `tools/`    | `exact_metric.py`: recomputes the paper's surface metric on a saved session, without re-running CAM.                                                                      |

## Run

Run the commands from the repository root. The harness needs FreeCAD (`freecadcmd` on `PATH`, or `FREECADCMD`) and an OpenAI-compatible server. By default it calls `http://127.0.0.1:8013/v1` with model `qwen3.8-27b` and API key `EMPTY` (`OPENAI_BASE_URL`, `CAM_MODEL`, `OPENAI_API_KEY`).

```bash
python -m cam_agent check
python -m cam_agent demo
```

`check` calls the server and reads the demo bracket with FreeCAD. `demo` builds that bracket with `examples/make_demo_part.py` and plans it. The session is written under `outputs/` (`CAM_OUT` or `--out` to put it elsewhere): `report.md`, the G-code `program.ngc`, and `session.json`. If AAGNet is not configured, feature recognition uses the rule-based recogniser, and `demo` still runs.

To plan another part:

```bash
python -m cam_agent run part.step --material "AL6061-T6" --notes "2x M6 tapped holes" --stock blank.step
```

`--setups` is `1` by default, `auto` for up to six indexed setups, or a maximum such as `3`. `--mode` is `parallel` (the default), `sequential`, or `deterministic`.

## Evaluation

`evaluate` scores a plan with the harness metrics. Export the task first, so the part, the stock and the conventions stay fixed, then point it at a harness session or at an answer directory:

```bash
python -m cam_agent export-task part.step --out task/ --material "AL6061-T6"
python -m cam_agent evaluate --task task/ --session outputs/<session>
python -m cam_agent evaluate --task task/ --answer answer_dir
```

An answer is `answer.json` plus one G-code file per setup. `task/prompt.md` states the format. The command writes `metrics.json` and `metrics.md`.

The surface figure is the share of the target surface within the tolerance (0.1 mm unless the task sets another). Points lie on the B-rep faces and are probed along the normals into the exact swept volume of the cutters. The same file reports cycle time, tool changes, ordering against standard practice, cutting data outside 0.5–1.5× the reference model, and safety issues: a tap without a rigid tapping cycle, a reamer or boring bar retracting at rapid, a rapid into the stock.

The 100 parts scored in the paper are listed in [`dataset/evaluation.json`](dataset/evaluation.json). Each entry names that part's `blank.step` and `part.json`. How the draw was made is in [`dataset/README.md`](dataset/README.md).

To recompute the surface metric of a saved session without re-running CAM:

```bash
python tools/exact_metric.py <session_dir> <out_dir>
```

This writes `<out_dir>/surface.json`.

## Citation

```bibtex
@article{millbill2026,
  title         = {MillBill: A Machining Agent with Geometric Feedback},
  author        = {Savrasov, Gennadii and Slavin, Artem and Kabisov, Soslan and
                   Kuznetsov, Andrey and Konushin, Anton and Zhemchuzhnikov, Dmitrii},
  year          = {2026},
  eprint        = {2611.22565},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2611.22565}
}
```

## License

MIT — see [LICENSE](LICENSE).
