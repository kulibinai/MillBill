"""A19 causal mechanics; no FreeCAD, geometry, network or live LLM needed."""
import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cam_agent.agents.cam_generation import CAMGenerationAgent, REPAIR_PROMPT
from cam_agent.knowledge import get_store


def request(kind="Custom", strategy="roughing"):
    return {"operations": [{"step": 1, "process": "cavity_roughing", "op_type": kind,
                            "strategy": strategy, "tool_number": 1, "tool_diameter_mm": 10,
                            "tc": 1, "doc_mm": 2, "woc_mm": 4, "finish_stock": .2,
                            "feature_ids": [], "candidates": [], "gcode_file": "cached.ngc"}],
            "tools": [{"tool_number": 1, "tool_id": "EM10"}], "controllers": [{"tc": 1}]}


def fix(action, value=None, step=1):
    return {"step": step, "action": action, "value": value}


def agent():
    a = object.__new__(CAMGenerationAgent)
    a.store = get_store()
    a._features = {}
    return a


class RepairFeedback(unittest.TestCase):
    def setUp(self):
        self.a = agent()
        self.env = patch.dict(os.environ, {"CAM_REPAIR_FEEDBACK": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def apply(self, req, fixes):
        outcomes = []
        self.a._apply_fixes(req, fixes, {"material": {"id": "AL6061-T6"}}, outcomes)
        return outcomes

    def test_invalid_and_zero(self):
        for f in [fix("set_woc_mm", 0), fix("set_doc_mm", -1), fix("set_peck_mm", True),
                  fix("set_doc_mm", float("nan")), fix("set_doc_mm", float("inf")),
                  fix("set_doc_mm", "1"), fix("set_depth_below_top", 61),
                  fix("drop_operation"), fix("unknown"), fix("switch_tool", "missing"),
                  fix("set_doc_mm", 1, step=999)]:
            r = request()
            old = copy.deepcopy(r)
            with self.subTest(f=f):
                out = self.apply(r, [f])
                self.assertEqual(out[0]["status"], "rejected")
                self.assertTrue(out[0]["reason"])
                self.assertEqual(r, old)
        r = request()
        self.assertEqual(self.apply(r, [fix("set_finish_stock", 0)])[0]["status"], "applied")
        self.assertEqual(r["operations"][0]["finish_stock"], 0)

    def test_custom_depth_reaches_actual_consumer(self):
        r = request()
        r["operations"][0]["reach_limit_mm"] = 15
        self.assertEqual(self.apply(r, [fix("set_depth_below_top", 10)])[0]["status"], "applied")
        self.assertEqual(r["operations"][0]["repair_depth_mm"], 10)
        self.assertEqual(r["operations"][0]["reach_limit_mm"], 15)
        self.assertNotIn("final_depth", r["operations"][0])
        self.assertNotIn("gcode_file", r["operations"][0])
        self.assertEqual(self.apply(r, [fix("set_depth_below_top", 10)])[0]["status"], "no-op")
        self.assertEqual(self.apply(r, [fix("set_final_depth", -10)])[0]["status"], "rejected")
        # Execute the production generator with only its geometry/path dependencies mocked.
        from cam_agent.services import paths, zlevel
        hf = SimpleNamespace(part_top=30, depth_limit=None, rough_top=30, H=None,
                             seal_chamfers=lambda _: None)
        seen = []
        self.a._heightfield = lambda *args: hf
        self.a._shield = lambda req, op, tool, lines: (lines, None)
        r["tools"][0]["diameter_mm"] = 10
        def rough(*args, **kwargs):
            seen.append(hf.depth_limit)
            return {"gcode": ["G1 Z20"]}
        with tempfile.TemporaryDirectory() as d, patch.object(zlevel, "roughing", rough):
            self.a._generate_custom(r, Path(d))
        self.assertEqual(seen, [20])

    def test_unrepresentable_json_integer_is_rejected(self):
        # JSON integers are unbounded; math.isfinite/float conversion can overflow.
        for action in ("set_doc_mm", "set_woc_mm", "set_final_depth", "set_depth_below_top"):
            r = request()
            old = copy.deepcopy(r)
            value = json.loads("1" + "0" * 400)
            out = self.apply(r, [fix(action, value)])
            self.assertEqual(out[0]["status"], "rejected")
            self.assertTrue(out[0]["reason"])
            self.assertEqual(r, old)

        r = request("Pocket")
        old = copy.deepcopy(r)
        # Finite input can still overflow native percent conversion before clamp.
        out = self.apply(r, [fix("set_woc_mm", 1e308)])
        self.assertEqual(out[0]["status"], "rejected")
        self.assertTrue(out[0]["reason"])
        self.assertEqual(r, old)

    def test_finishing_depth_and_stepover_actual_consumer(self):
        from cam_agent.services import zlevel
        r = request(strategy="finishing")
        r["tools"][0]["diameter_mm"] = 10
        r["operations"][0]["process"] = "surface_finishing"
        out = self.apply(r, [fix("set_depth_below_top", 12), fix("set_woc_mm", 1)])
        self.assertEqual([x["status"] for x in out], ["applied", "applied"])
        hf = SimpleNamespace(part_top=30, depth_limit=None, x0=0, x1=10, y0=0, y1=10)
        self.a.s = SimpleNamespace(tol_mm=.1, sim_max_cells=1000)
        self.a._heightfield = lambda *args: hf
        self.a._shield = lambda req, op, tool, lines: (lines, None)
        seen = []
        def finishing(*args, **kwargs):
            seen.append((hf.depth_limit, kwargs["stepover"]))
            return {"gcode": ["G1 Z18"]}
        with tempfile.TemporaryDirectory() as d, patch.object(zlevel, "finishing", finishing):
            self.a._generate_custom(r, Path(d))
        self.assertEqual(seen, [(18, 3)])

    def test_mixed_and_native_zero_top(self):
        r = request("Pocket")
        r["operations"][0]["start_depth"] = 0
        out = self.apply(r, [fix("set_woc_mm", 0), fix("set_doc_mm", 1), fix("set_doc_mm", 1),
                             fix("set_depth_below_top", 2)])
        self.assertEqual([x["status"] for x in out], ["rejected", "applied", "no-op", "applied"])
        self.assertEqual(r["operations"][0]["final_depth"], -2)
        self.assertEqual(self.apply(request("Pocket"), [fix("set_depth_below_top", 2)])[0]["status"], "rejected")

    def test_facing_floor_applies_to_both_depth_actions(self):
        for feedback in ("0", "1"):
            for action in (fix("set_final_depth", -1), fix("set_depth_below_top", 1)):
                with self.subTest(feedback=feedback, action=action), patch.dict(os.environ, {
                    "CAM_REPAIR_FEEDBACK": feedback, "CAM_FACE_STOP": "1",
                }):
                    r = request("MillFace")
                    r["operations"][0].update(process="facing", face_floor=0, start_depth=0, final_depth=0)
                    old = copy.deepcopy(r)
                    out = self.apply(r, [action])
                    self.assertEqual(r, old)
                    if feedback == "1":
                        self.assertEqual(out[0]["status"], "rejected")
                        self.assertIn("facing floor", out[0]["reason"])

    def test_actual_native_clamp_payload(self):
        # Execute the real FreeCAD property adapter, with fake operation objects.
        source = Path(__file__).resolve().parents[1] / "cam_agent/services/fc_scripts/build_cam_job.py"
        node = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "create_op")
        scope = {"q": lambda value: value, "log": lambda *args: None}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), scope)
        mods = {}
        for name in ("Path", "Path.Op", "Path.Op.Drilling", "Path.Op.MillFace", "Path.Op.Pocket", "Path.Op.Profile", "Path.Op.Adaptive"):
            mods[name] = ModuleType(name)
        def create(name):
            return SimpleNamespace(StepOver=0)
        for m in list(mods.values())[2:]:
            m.Create = create
        for kind in ("MillFace", "Pocket", "Adaptive"):
            r = request(kind)
            r["operations"][0]["woc_mm"] = 100
            old = copy.deepcopy(r)
            out = self.apply(r, [fix("set_woc_mm", 200)])
            self.assertEqual(out[0]["status"], "no-op")
            self.assertEqual(r, old, "no-op must preserve cached paths/raw input")
            def payload(woc):
                op = dict(r["operations"][0], tool_diameter=10, woc_mm=woc)
                with patch.dict(sys.modules, mods):
                    return scope["create_op"](None, None, None, op, {1: None}).StepOver
            self.assertEqual(payload(100), payload(200))
            self.assertNotEqual(payload(100), payload(2))
            self.assertEqual(self.apply(r, [fix("set_woc_mm", 2)])[0]["status"], "applied")
            # Distinguish true integer-percent rounding from raw WOC comparison.
            self.assertEqual(payload(2), payload(2.049))
            self.assertNotEqual(payload(2), payload(2.051))
            self.assertEqual(self.apply(r, [fix("set_woc_mm", 2.049)])[0]["status"], "no-op")
            self.assertEqual(self.apply(r, [fix("set_woc_mm", 2.051)])[0]["status"], "applied")

    def test_custom_clamp_defaults_unused_fields(self):
        r = request(strategy="finishing")
        out = self.apply(r, [fix("set_woc_mm", 5), fix("set_doc_mm", 1), fix("set_peck_mm", 1), fix("set_finish_stock", 0)])
        self.assertEqual([x["status"] for x in out], ["no-op"] * 4)
        r = request()
        r["operations"][0].pop("doc_mm")
        self.assertEqual(self.apply(r, [fix("set_doc_mm", 5)])[0]["status"], "no-op")

    def test_default_off_equality(self):
        fixes = [fix("set_woc_mm", 0), fix("set_doc_mm", 1), fix("set_depth_below_top", 10)]
        base = request()
        legacy = self.a._apply_fixes_legacy(copy.deepcopy(base), fixes, {"material": {"id": "AL6061-T6"}})
        for flag in (None, "0"):
            with patch.dict(os.environ, {}, clear=True):
                if flag is not None:
                    os.environ["CAM_REPAIR_FEEDBACK"] = flag
                out = []
                actual = self.a._apply_fixes(copy.deepcopy(base), fixes, {"material": {"id": "AL6061-T6"}}, out)
                self.assertEqual(actual, legacy)
                self.assertEqual(out, [])

    def test_drop_and_same_tool(self):
        r = request()
        self.assertEqual(self.apply(r, [fix("switch_tool", "EM10")])[0]["status"], "no-op")
        r["operations"][0]["process"] = "deburring"
        self.assertEqual(self.apply(r, [fix("drop_operation")])[0]["status"], "applied")
        self.assertFalse(r["operations"])
        self.assertEqual(self.apply(r, [fix("drop_operation")])[0]["status"], "rejected")

    def test_loop_stop_and_feedback_toggle_same_mechanics(self):
        for prompt_flag in ("0", "1"):
            for fixes, expected_rounds in [([fix("set_woc_mm", 0)], 1),
                                           ([fix("set_woc_mm", 0), fix("set_doc_mm", 1)], 2),
                                           ([fix("set_doc_mm", 1), fix("set_doc_mm", 2)], 1)]:
                with self.subTest(prompt=prompt_flag, fixes=fixes), patch.dict(os.environ, {"CAM_REPAIR_FEEDBACK_PROMPT": prompt_flag}):
                    a = agent()
                    a.repair_rounds, a.mode, a.name = 2, "parallel", "cam_generation"
                    a.s = SimpleNamespace(rest=False)
                    calls, prompts = [], []
                    def build(*args):
                        calls.append(copy.deepcopy(args[1]))
                        return {"operations": [1], "verification": {"issues": ["bad"] * (3 - len(calls))}}
                    def llm(prompt, payload, **kwargs):
                        prompts.append((prompt, json.loads(payload)))
                        return {"fixes": fixes}
                    a.llm = SimpleNamespace(json_call=llm, trace=SimpleNamespace(log=lambda *args, **kwargs: None))
                    a.freecad = SimpleNamespace(available=lambda: True, run=build)
                    a._meshes = lambda *args: (None, None)
                    a._generate_custom = lambda *args: None
                    a._lint = lambda *args: []
                    a._deterministic_fixes = lambda *args: []
                    with tempfile.TemporaryDirectory() as d:
                        result = a.run_request(request(), Path(d), {"features": [], "material": {"id": "AL6061-T6"}})
                    self.assertEqual(len(calls), expected_rounds)
                    self.assertEqual(result["rounds"][-1]["repair_stop"], "no effective change")
                    if prompt_flag == "1":
                        self.assertIn("not a work offset", prompts[0][0])
                        if len(prompts) > 1:
                            self.assertEqual(prompts[1][1]["rejected_fixes"][0]["status"], "rejected")
                    else:
                        self.assertEqual(prompts[0][0], REPAIR_PROMPT)
                        self.assertNotIn("rejected_fixes", prompts[0][1])


if __name__ == "__main__":
    unittest.main()
