"""Multi-agent CAM planner for FreeCAD (Design-to-Plan style) with a deterministic planning core.

  Tier 1  Orchestrator            -> cam_agent.orchestrator
  Tier 2  Agents                  -> cam_agent.agents.*  (feature extraction, context fusion, knowledge retrieval (ReAct),
                                     process sequence, tool selection, setup planning, CAM generation, report)
  Core    Planning                -> cam_agent.planning  (sequencing + validation, cutting-data model)
  Tier 3  Services                -> cam_agent.services  (headless FreeCAD, dexel simulation, verification, G-code)
          Evaluation              -> cam_agent.evaluate  (same task / same metrics for planners without the harness)
"""

__version__ = "0.2.0"
