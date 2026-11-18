"""Verification of multi-setup (indexed) machining on one workpiece model without voxels.

    workpiece W = blank AND for every setup k { what setup k's program leaves }

The surface metric (`verify.surface_metric`) is measured on the EXACT workpiece (`exact.py`): the blank (box or closed
mesh) AND the swept volumes of every programmed setup; a setup that is not programmed yet takes its ideal result from
the target mesh (IdealView: everything above the target's top seen from that side removed), so a setup can be checked
before the others exist. `final()` takes the programmed setups only.

The dexel maps of the setups (h_k, the ideal views at half resolution) are kept beside it for what needs a height map:
the defect regions of the repair loop, rest machining (`_inside`) and the stock surface of later setups.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dexel import DexelMap, stock_dexel
from .raster import Grid, rasterize
from .stock import transform
from .verify import SetupState, Workpiece, brep_samples, mesh_samples, save_surface, sim_grid, simulate_setup, surface_metric
from .voxel import Frame


class MultiSetupVerifier:
    def __init__(self, part_raw: dict, setups: list[dict], stock_raw: dict, tol: float = 0.1, max_cells: float = 9e6,
                 tooling: dict | None = None, samples: int = 600000, part_path=None, freecad=None, workdir=None) -> None:
        """part_raw / stock_raw: meshes in the part's original frame; setups: [{"index", "up", "translation"}] (first =
        primary); part_path: the part's STEP (the metric's points on its B-rep faces, cached in workdir)."""
        from . import exact  # numba
        self.part_raw, self.stock_raw, self.tol, self.max_cells, self.tooling, self.samples = part_raw, stock_raw, tol, max_cells, tooling, samples
        self.frames = {s["index"]: Frame(s["up"], s["translation"]) for s in setups}
        k0 = setups[0]["index"]
        self.frame0 = self.frames[k0]
        st0, pt0 = transform(stock_raw, self.frame0), transform(part_raw, self.frame0)
        bb = [min(st0["bbox"][0], pt0["bbox"][0]), min(st0["bbox"][1], pt0["bbox"][1]), max(st0["bbox"][3], pt0["bbox"][3]), max(st0["bbox"][4], pt0["bbox"][4])]
        self.stock = stock_dexel(sim_grid(bb, 1.0, tol, max_cells), st0, pt0)
        self.states: dict[int, SetupState] = {}
        self.ideal_exact: dict = {}
        for k, fr in self.frames.items():
            self.states[k] = SetupState(fr, ideal=self._ideal(fr))
            self.ideal_exact[k] = exact.IdealView(fr, part_raw)
        self.sweeps: dict = {}  # setup index -> exact.SweepSetup of its last checked (= delivered) program
        if part_path is not None:
            wd = Path(workdir) if workdir is not None else Path(part_path).resolve().parent / "fc"
            self.brep = brep_samples(part_path, samples, wd, freecad)
        else:
            self.brep = mesh_samples(part_raw["vertices"], part_raw["faces"], samples)
        self.results: dict[int, dict] = {}
        self.programs: dict[int, tuple] = {}
        self._final_surface = None

    def _ideal(self, fr: Frame) -> DexelMap:
        pk = transform(self.part_raw, fr)
        b = pk["bbox"]
        g = sim_grid([b[0], b[1], b[3], b[4]], 2.0, 2 * self.tol, self.max_cells / 4)  # ideal views are only fallbacks: half resolution
        top = rasterize(g, pk["vertices"], pk["faces"], "top", empty=b[2] - 10.0)
        return DexelMap(g, top, np.full(g.shape, b[2] - 10.0))

    def workpiece(self, only_built: bool = False) -> Workpiece:
        """The dexel workpiece (rest machining, stock surface); the metric's is exact_workpiece()."""
        if only_built:
            states = [SetupState(s.frame, dexel=s.dexel) for s in self.states.values() if s.dexel is not None]
        else:
            states = list(self.states.values())
        return Workpiece(self.stock, self.frame0, states)

    def exact_workpiece(self, only_built: bool = True):
        from . import exact
        parts = [self.sweeps[k] if k in self.sweeps else self.ideal_exact[k] for k in self.frames if k in self.sweeps or not only_built]
        return exact.blank_solid(self.stock_raw, parts)

    # ------------------------------------------------------------------------------------------------ per setup
    def check_setup(self, k: int, gcode_file: str, req: dict, features: list[dict]) -> dict:
        fr = self.frames[k]
        part_k, stock_k = transform(self.part_raw, fr), transform(self.stock_raw, fr)
        others = [s for kk, s in self.states.items() if kk != k]
        exact_others = [self.sweeps.get(kk) or self.ideal_exact[kk] for kk in self.frames if kk != k]
        sim = simulate_setup(gcode_file, part_k, stock_k, req["operations"], req["tools"], features, tol=self.tol, max_cells=self.max_cells,
                             tooling=self.tooling, samples=self.brep, others=others, frame=fr, stock_state=(self.stock, self.frame0),
                             exact_others=exact_others, stock_raw=self.stock_raw,
                             request_frame=Frame(req["up"], req["translation"])
                             if req.get("up") is not None and req.get("translation") is not None else None)
        self.states[k] = SetupState(fr, dexel=sim["dexel"], ideal=self.states[k].ideal)
        self.sweeps[k] = sim["sweep"]
        self.results[k] = sim
        self.programs[k] = (gcode_file, req["operations"], req["tools"])  # the last check = the delivered program
        stats = {**sim["stats"], "view": fr.up, "later_setups": "ideal" if len(self.sweeps) < len(self.frames) else "programmed"}
        # private keys (popped by the CAM agent): the simulation and the dexel workpiece test, for rest machining
        observations = {"exact_observations": {**sim["exact_observations"], "setup_index": k, "setup_side": fr.up}} if "exact_observations" in sim else {}
        if "cavity_region_observations" in sim:
            observations["cavity_region_observations"] = {**sim["cavity_region_observations"], "setup_index": k}
        return {"stats": stats, "regions": sim["regions"], "errors": sim["errors"], **observations, "_sim": sim,
                "_inside": self.workpiece(only_built=True).inside, "_frame": fr}

    def stock_surface(self, k: int, path, res: float = 0.5) -> dict:
        """Top of the material that is really left before setup k (built setups only), seen from its side."""
        fr = self.frames[k]
        sk = transform(self.stock_raw, fr)
        pk = transform(self.part_raw, fr)
        lo = np.minimum(np.asarray(sk["bbox"][:3]), np.asarray(pk["bbox"][:3]))
        hi = np.maximum(np.asarray(sk["bbox"][3:]), np.asarray(pk["bbox"][3:]))
        g = Grid(lo[0], lo[1], hi[0], hi[1], res)
        wp = self.workpiece(only_built=True)
        zs = np.arange(hi[2], lo[2] - res, -res)
        h = np.full(g.shape, lo[2])
        found = np.zeros(g.shape, bool)
        X, Y = g.X.ravel(), g.Y.ravel()
        for z in zs:  # top-down: the first level with material is the surface
            if found.all():
                break
            P = fr.to_original(np.c_[X, Y, np.full(X.shape, z)])
            m = wp.inside(P).reshape(g.shape) & ~found
            h[m] = z
            found |= m
        np.save(path, h)
        return {"file": str(path), "x0": float(g.x0), "y0": float(g.y0), "res": float(res)}

    # ------------------------------------------------------------------------------------------------ final
    def final(self) -> dict:
        """The surface metric of the whole job: blank AND every programmed setup (exact)."""
        from . import exact
        built = sorted(self.sweeps)
        sd = surface_metric(self.brep, self.exact_workpiece(only_built=True).inside, self.tol)
        self._final_surface = sd
        return {**sd["stats"], "arc_tol_mm": exact.ARC_TOL, "blank": "box" if exact.box_of(self.stock_raw) is not None else "mesh",
                "setups_built": built}

    def save_surface(self, path) -> None:
        """The last final()'s points and verdicts (original frame) for the videos."""
        if self._final_surface is not None:
            save_surface(path, self._final_surface, "original")
