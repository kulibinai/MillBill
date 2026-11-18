import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cam_agent.services.freecad_runner import FreeCADError, FreeCADRunner


class FreeCADRunnerTest(unittest.TestCase):
    def test_reused_workdir_requires_new_response_and_successful_process(self):
        def run_success(command, **kwargs):
            Path(kwargs["env"]["CAM_FC_RESPONSE"]).write_text('{"ok": true}')
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as workdir:
            runner = FreeCADRunner(freecadcmd=os.__file__)
            with patch("cam_agent.services.freecad_runner.subprocess.run", side_effect=run_success):
                self.assertTrue(runner.run("extract_features.py", {}, workdir)["ok"])

            with patch(
                "cam_agent.services.freecad_runner.subprocess.run",
                return_value=subprocess.CompletedProcess([], 0, "", ""),
            ):
                with self.assertRaisesRegex(FreeCADError, "produced no response"):
                    runner.run("extract_features.py", {}, workdir)

            def run_failure(command, **kwargs):
                run_success(command, **kwargs)
                return subprocess.CompletedProcess(command, 1, "", "FreeCAD failed")

            with patch("cam_agent.services.freecad_runner.subprocess.run", side_effect=run_failure):
                with self.assertRaisesRegex(FreeCADError, "failed.*rc=1"):
                    runner.run("extract_features.py", {}, workdir)
