"""Runtime settings. Everything can be overridden through environment variables."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
KB_DIR = PKG_DIR / "knowledge" / "data"
FC_SCRIPTS_DIR = PKG_DIR / "services" / "fc_scripts"

_FREECAD_CANDIDATES = [
    ROOT_DIR / "apps" / "FreeCAD.app" / "Contents" / "Resources" / "bin" / "freecadcmd",
    Path("/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd"),
]


def find_freecadcmd() -> str | None:
    env = os.environ.get("FREECADCMD")
    if env and Path(env).exists():
        return env
    for c in _FREECAD_CANDIDATES:
        if c.exists():
            return str(c)
    for name in ("freecadcmd", "FreeCADCmd", "freecad-cmd"):
        p = shutil.which(name)
        if p:
            return p
    return None


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass
class Settings:
    # OpenAI-compatible LLM endpoint
    base_url: str = field(default_factory=lambda: os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8013/v1"))
    api_key: str = field(default_factory=lambda: os.environ.get("OPENAI_API_KEY", "EMPTY"))
    model: str = field(default_factory=lambda: os.environ.get("CAM_MODEL", "qwen3.8-27b"))
    temperature: float = field(default_factory=lambda: _env_float("CAM_TEMPERATURE", 0.0))
    thinking: bool = field(default_factory=lambda: os.environ.get("CAM_THINKING", "0") == "1")
    max_tokens: int = field(default_factory=lambda: int(os.environ.get("CAM_MAX_TOKENS", "8192")))
    timeout: float = 600.0
    # Agents: sequential | parallel | deterministic (Sec. 3.4 of the paper)
    mode: str = field(default_factory=lambda: os.environ.get("CAM_MODE", "parallel"))
    max_iterations: int = field(default_factory=lambda: int(os.environ.get("CAM_MAX_ITER", "8")))
    # Process planning
    sequencing: str = field(default_factory=lambda: os.environ.get("CAM_SEQUENCING", "standard"))  # standard | element
    recognizer: str = field(default_factory=lambda: os.environ.get("CAM_RECOGNIZER", "auto"))  # auto | rules | aagnet
    # What the inputs do not say (B-Rep of the part and of the blank only, no drawing): part material and general tolerance
    material: str = field(default_factory=lambda: os.environ.get("CAM_MATERIAL", "AL6061-T6"))
    general_tolerance: str = field(default_factory=lambda: os.environ.get("CAM_GENERAL_TOLERANCE", "ISO 2768-m"))
    # Stock (default = bounding box + margins; --stock = user blank)
    stock_xy_margin: float = field(default_factory=lambda: _env_float("CAM_STOCK_XY", 2.0))
    stock_top_margin: float = field(default_factory=lambda: _env_float("CAM_STOCK_TOP", 1.0))
    stock_bottom_margin: float = field(default_factory=lambda: _env_float("CAM_STOCK_BOTTOM", 0.0))
    # Tool holder nose diameter (collet nut / shrink-fit nose): with the tool's stick-out it decides how deep a setup reaches
    holder_d: float = field(default_factory=lambda: _env_float("CAM_HOLDER_D", 40.0))
    # Gouge shield on the generated (height-map) toolpaths: certify against the target, lift what would cut it
    shield: bool = field(default_factory=lambda: os.environ.get("CAM_SHIELD", "1") != "0")
    # Rest finishing planned from the simulation of the first round (ball endmills chosen by morphological closing)
    rest: bool = field(default_factory=lambda: os.environ.get("CAM_REST", "1") != "0")
    # Verification: tolerance band of the surface metric, simulation cell = tolerance / 2 (bounded by a cell budget)
    tol_mm: float = field(default_factory=lambda: _env_float("CAM_TOL", 0.1))
    sim_max_cells: float = field(default_factory=lambda: _env_float("CAM_SIM_MAX_CELLS", 9e6))
    surface_samples: int = field(default_factory=lambda: int(os.environ.get("CAM_SURFACE_SAMPLES", "600000")))
    repair_rounds: int = field(default_factory=lambda: int(os.environ.get("CAM_REPAIR_ROUNDS", "2")))
    # FreeCAD
    freecadcmd: str | None = field(default_factory=find_freecadcmd)
    freecad_timeout: float = 900.0
    post_processor: str = field(default_factory=lambda: os.environ.get("CAM_POST", "linuxcnc"))
    # canned cycle for rigid tapping written by the post fix: G84 (ISO / Fanuc / Haas / LinuxCNC 2.9+) or G33.1 (LinuxCNC)
    tap_cycle: str = field(default_factory=lambda: os.environ.get("CAM_TAP_CYCLE", "G84").upper())
    tap_m29: bool = field(default_factory=lambda: os.environ.get("CAM_TAP_M29", "0") == "1")  # Fanuc rigid-tap mode
    # Output
    out_dir: Path = field(default_factory=lambda: Path(os.environ.get("CAM_OUT", str(ROOT_DIR / "outputs"))))

    @property
    def stock_margins(self) -> dict:
        return {"xy_margin": self.stock_xy_margin, "z_top": self.stock_top_margin, "z_bottom": self.stock_bottom_margin}


SETTINGS = Settings()
