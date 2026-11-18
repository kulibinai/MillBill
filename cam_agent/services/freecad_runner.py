"""Tier-3 utility service: runs headless FreeCAD (freecadcmd) scripts and exchanges JSON through files."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from ..config import FC_SCRIPTS_DIR, SETTINGS


class FreeCADError(RuntimeError):
    pass


class FreeCADRunner:
    def __init__(self, freecadcmd: str | None = None, timeout: float | None = None) -> None:
        self.exe = freecadcmd or SETTINGS.freecadcmd
        self.timeout = timeout or SETTINGS.freecad_timeout

    def available(self) -> bool:
        return bool(self.exe) and Path(self.exe).exists()

    def run(self, script_name: str, request: dict, workdir: str | Path | None = None,
            *, env_overrides: dict[str, str] | None = None) -> dict:
        if not self.available():
            raise FreeCADError("freecadcmd not found. Set FREECADCMD=/path/to/freecadcmd")
        script = FC_SCRIPTS_DIR / script_name
        request = dict(request)
        for key in ("path", "out_dir", "out"):
            if request.get(key):
                request[key] = str(Path(request[key]).expanduser().resolve())
        workdir = Path(workdir or tempfile.mkdtemp(prefix="cam_fc_")).resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        req_path, resp_path = workdir / f"{script.stem}.request.json", workdir / f"{script.stem}.response.json"
        resp_path.unlink(missing_ok=True)
        req_path.write_text(json.dumps(request, default=str))
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "PYTHONHOME", "CONDA", "VIRTUAL_ENV"))}
        if env_overrides:
            env.update(env_overrides)
        env["CAM_FC_REQUEST"] = str(req_path)
        env["CAM_FC_RESPONSE"] = str(resp_path)
        env.setdefault("HOME", str(Path.home()))
        try:
            proc = subprocess.run([self.exe, str(script)], cwd=str(workdir), env=env, capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise FreeCADError(f"FreeCAD timed out after {self.timeout}s running {script_name}") from e
        if proc.returncode or not resp_path.exists():
            tail = (proc.stdout + "\n" + proc.stderr)[-4000:]
            reason = "failed" if proc.returncode else "produced no response"
            raise FreeCADError(f"FreeCAD {reason} (rc={proc.returncode}) for {script_name}:\n{tail}")
        resp = json.loads(resp_path.read_text())
        resp.setdefault("_stdout_tail", proc.stdout[-2000:])
        if resp.get("error"):
            raise FreeCADError(f"{script_name}: {resp['error']}\n{resp.get('traceback', '')}")
        return resp
