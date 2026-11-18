# MillBill: A Machining Agent with Geometric Feedback

MillBill turns a boundary-representation part, its milling stock and its material into G-code for indexed three-axis machining. A locally hosted open model revises a standard-practice plan of operations, tools and cutting parameters. A geometric harness writes the toolpaths, simulates material removal and returns surface deviations so the plan can be refined. The model never writes coordinates or G-code. Setups are chosen by what the available cutters can reach, not by visibility alone.

- Paper: [arXiv:2611.22565](https://arxiv.org/abs/2611.22565)

## What is here

| Path       | Contents                                                                                                                                                                      |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `dataset/` | GrabCAD parts paired with milling stocks, including the 100-part evaluation split used in the paper. The full description is in [`dataset/README.md`](dataset/README.md).     |
| `harness/` | [TO BE ADDED LATER TODAY] The harness described in the paper: feature recognition, reach-aware setup planning, toolpaths, material-removal simulation and geometric feedback. |

## License

MIT — see [LICENSE](LICENSE).
