"""Slavin llm10 integration through the existing planning and CAM entry points."""
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cam_agent.agents.cam_generation import CAMGenerationAgent
from cam_agent.agents.tool_selection import ToolSelectionAgent
from cam_agent.knowledge import get_store
from cam_agent.orchestrator import Orchestrator


class SlavinLLM10(unittest.TestCase):
    def setUp(self):
        self.store = get_store()
        self.llm = SimpleNamespace(last_usage=SimpleNamespace(total=0), trace=Mock())
        self.cam = CAMGenerationAgent(self.llm, freecad=object())

    def test_skim_only_stops_at_half_millimetre(self):
        for allowance, expected in ((None, False), (0, False), (0.05, True),
                                    (0.5, True), (0.50001, False), (5, False)):
            with self.subTest(allowance=allowance):
                face = {"type": "planar_face", "allowance_mm": allowance}
                self.assertEqual(self.cam._skim_only({"features": [face]}), expected)
                self.assertFalse(self.cam._skim_only({"features": [face, {"type": "cavity"}]}))

    def test_pocket_commands_fall_back_to_cavity(self):
        cavity = {"id": "C", "type": "cavity"}
        fused = {"features": [cavity]}
        for process, strategy in (("pocket_milling", "roughing"), ("pocket_finishing", "finishing")):
            for selected in ([cavity], []):
                with self.subTest(process=process, selected=selected):
                    operation = self.cam._op_for({"process": process, "reach_limit_mm": 7}, selected, fused, {}, {})
                    self.assertEqual(operation, {"op_type": "Custom", "strategy": strategy,
                                                 "reach_limit_mm": 7, "cavity": True})
        self.assertIsNone(self.cam._op_for({"process": "pocket_milling"}, [], {"features": []}, {}, {}))

    def test_rejected_tool_restores_draft_after_empty_assignment(self):
        selector = ToolSelectionAgent(self.llm)
        hole = {"id": "H1", "type": "hole", "diameter": 4, "depth": 10, "through": False}
        fused = {"material": {"id": "AL6061-T6"}, "features": [hole]}
        material = self.store.material("AL6061-T6")
        draft = selector._select({"step": 1, "process": "drilling", "feature_ids": ["H1"]},
                                 {"H1": hole}, material, fused)
        proposals = [{"step": 1, "tool_id": "DR-C8D"}, {"step": 1, "tool_id": "missing-tool"}]
        empty = {**draft, "tool_id": None}
        checks = [(True, [], {}), (False, ["not in the tool library"], {})]
        with patch.object(selector, "_check", side_effect=checks), patch.object(selector, "_assign", return_value=empty):
            assignments, changes = selector._merge([draft], proposals, {"H1": hole}, material, fused)
        self.assertEqual(assignments, [draft])
        self.assertFalse(changes[-1]["accepted"])

    def test_late_setup_with_thick_allowance_rechecks_stock(self):
        class PlanningReached(Exception):
            pass

        runner = object.__new__(Orchestrator)
        runner.llm, runner.freecad, runner.store, runner.log = self.llm, object(), self.store, Mock()
        runner.s = SimpleNamespace(recognizer="rules", tol_mm=0.1, sim_max_cells=10, surface_samples=10)
        runner._tess, runner._stock_mesh = Mock(), Mock()
        runner._apply_allowances = Mock(return_value={})
        runner._plan_setup = Mock(side_effect=PlanningReached)
        for allowance, recheck in ((0.5, False), (0.50001, True), (5, True)):
            with self.subTest(allowance=allowance):
                face = {"id": "F", "type": "planar_face", "needs_facing": True, "setup": 2}
                plan = {"setups": [{"index": 2, "up": "+Z", "translation": [0, 0, 0],
                                     "part": {"size": [1, 1, 1]}, "features": [], "allowance_mm": allowance}]}
                runner._fusion_and_knowledge = Mock(return_value=(
                    {"features": [face], "material": {"id": "AL6061-T6"}}, SimpleNamespace(output={})))
                runner._detect_cavity = Mock(return_value=None)
                with patch.dict(os.environ, {"CAM_WALL_BUDGET_S": "0"}), patch(
                        "cam_agent.services.multi_verify.MultiSetupVerifier") as verifier, patch(
                        "cam_agent.orchestrator.FeatureExtractionAgent"), patch("cam_agent.services.stock.transform"):
                    with self.assertRaises(PlanningReached):
                        runner._run_multi({"session_dir": "/tmp"}, plan, Path("unused.step"),
                                          "AL6061-T6", "", False, "", 0, {})
                self.assertEqual(runner._detect_cavity.call_count, 1 + recheck)
                self.assertEqual(verifier.return_value.stock_surface.call_count, int(recheck))

    def test_micro_tools_have_slavin_dimensions(self):
        tools = {t["id"]: t for t in self.store.tool_library["tools"]}
        for name, diameter, edge in (("EM1-U2", 1, 4), ("EM1-U2-L", 1, 12),
                                     ("BEM1", 1, 3), ("CH90-3", 3, 1.4)):
            with self.subTest(tool=name):
                tool = tools[name]
                self.assertEqual(tool["diameter_mm"], diameter)
                self.assertEqual(tool["shape_params"]["Diameter"], diameter)
                self.assertEqual(tool["flute_length_mm"], edge)
                self.assertEqual(tool["shape_params"]["CuttingEdgeHeight"], edge)
                self.assertEqual(tool["overall_length_mm"], tool["shape_params"]["Length"])


if __name__ == "__main__":
    unittest.main()
