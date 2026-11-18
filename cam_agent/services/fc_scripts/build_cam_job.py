"""Headless FreeCAD: build a CAM Job from the plan (sequence + tool assignments), compute toolpaths, post-process to
G-code and verify the result (bounds, depths, rapids, path length, time estimate).

Request JSON:
{ "path": <STEP/FCStd>, "out_dir": <dir>, "post": "linuxcnc", "stock": {"xy_margin": 2, "z_top": 1, "z_bottom": 0, "solid": <optional blank STEP>},
  "tools": [{"tool_number", "tool_id", "name", "category", "diameter_mm", "flutes", "freecad_shape", "shape_params",
             "spindle_rpm", "feed_mm_min", "plunge_feed_mm_min"}],
  "operations": [{"step", "process", "op_type": "MillFace|Pocket|Drilling|Profile|Deburr", "faces": [...], "tool_number",
                  "doc_mm", "woc_mm", "peck_mm", "label", "tapping": bool, "finish_stock": float}] }
"""
import math
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import load_oriented, main  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cam_agent.services.cnc import export_job, safe_comment_label, custom_program_lines  # noqa: E402

import FreeCAD  # noqa: E402
import Part  # noqa: E402

LOG = []


def patch_toolcontroller_lookup():
    """Headless FreeCAD 1.x: PathUtils.findToolController leaves 'tc' unbound when a job has several tool
    controllers and no GUI to ask the user. Return the first supported controller instead; we assign the
    right one explicitly right after each operation is created."""
    import PathScripts.PathUtils as PathUtils

    def find(obj, proxy, name=None):
        controllers = PathUtils.getToolControllers(obj, proxy)
        if not controllers:
            job = PathUtils.findParentJob(obj)
            controllers = list(job.Tools.Group) if job else []
        if name:
            named = [c for c in controllers if c.Label == name]
            if named:
                return named[0]
        return controllers[0] if controllers else None

    PathUtils.findToolController = find


def log(*a):
    LOG.append(" ".join(str(x) for x in a))


def q(val, unit="mm"):
    return FreeCAD.Units.Quantity("%s %s" % (val, unit))


def load_base(doc, path, up="+Z", translation=None):
    shape, translation = load_oriented(path, up, translation)
    base = doc.addObject("Part::Feature", "Model")
    base.Shape = shape
    doc.recompute()
    return base, translation


def use_stock_solid(doc, job, path, up, translation):
    """Replace the job's bounding-box stock by the user's blank, placed with the model's own transform."""
    import Path.Base.Util as PathUtil
    shape, _ = load_oriented(path, up, translation)
    solid = doc.addObject("Part::Feature", "StockSolid")
    solid.Shape = shape
    old = job.Stock
    job.Stock = solid
    if old is not None:
        try:
            PathUtil.clearExpressionEngine(old)
        except Exception:  # noqa: BLE001
            pass
        doc.removeObject(old.Name)
    doc.recompute()
    return solid


# --------------------------------------------------------------------------- tools
def make_toolbit(doc, t):
    """Try the FreeCAD 1.0 and 1.1 tool-bit APIs in turn; return a ToolBit document object."""
    shape_file = t.get("freecad_shape") or "endmill.fcstd"
    bit = None
    errors = []
    try:  # FreeCAD 1.0.x
        import Path.Tool.Bit as PathToolBit
        bit = PathToolBit.Factory.Create("TB_%s" % t["tool_id"], shape_file)
    except Exception as e:  # noqa: BLE001
        errors.append("Bit.Factory: %s" % e)
    if bit is None:
        try:  # FreeCAD 1.1.x toolbit package
            from Path.Tool.toolbit import ToolBit
            tb = ToolBit.from_shape_id(shape_file, label="T%d %s" % (t["tool_number"], t.get("name") or t["tool_id"]))
            params = dict(t.get("shape_params") or {})
            params["Diameter"] = t.get("diameter_mm") or params.get("Diameter", 6)
            if t.get("flutes"):
                params["Flutes"] = int(t["flutes"])
            for k, v in params.items():
                try:
                    tb.set_property(k, v)
                except Exception as e:  # noqa: BLE001
                    log("toolbit set_property", k, e)
            bit = tb.attach_to_doc(doc)
        except Exception as e:  # noqa: BLE001
            errors.append("toolbit pkg: %s" % e)
    if bit is None:
        raise RuntimeError("cannot create tool bit: " + "; ".join(errors))
    if hasattr(bit, "Label"):
        bit.Label = "T%d %s" % (t["tool_number"], t.get("name") or t["tool_id"])
    params = dict(t.get("shape_params") or {})
    params["Diameter"] = t.get("diameter_mm") or params.get("Diameter", 6)
    if "Flutes" in params or hasattr(bit, "Flutes"):
        params["Flutes"] = int(t.get("flutes") or params.get("Flutes", 2))
    for k, v in params.items():
        if hasattr(bit, k):
            try:
                prop_type = bit.getTypeIdOfProperty(k)
                if "Quantity" in prop_type or "Length" in prop_type or "Angle" in prop_type:
                    setattr(bit, k, q(v, "deg" if "Angle" in prop_type else "mm"))
                else:
                    setattr(bit, k, v)
            except Exception as e:  # noqa: BLE001
                log("toolbit prop", k, e)
    return bit


def make_toolcontroller(doc, job, t, c=None, bit=None):
    """One controller per speed/feed set of a tool (c); several controllers may share one tool bit and tool number."""
    import Path.Tool.Controller as PathToolController
    bit = bit or make_toolbit(doc, t)
    c = c or t
    name = "TC_T%d_%d" % (t["tool_number"], int(c.get("tc", 0)))
    tc = PathToolController.Create(name, tool=bit, toolNumber=int(t["tool_number"]))
    tc.Label = safe_comment_label("TC: T%d %s S%s F%s" % (t["tool_number"], t.get("name") or t["tool_id"], c.get("spindle_rpm"), c.get("feed_mm_min")))
    t = {**t, **{k: c.get(k) for k in ("spindle_rpm", "feed_mm_min", "plunge_feed_mm_min")}}
    tc.SpindleSpeed = float(t.get("spindle_rpm") or 3000)
    tc.SpindleDir = "Forward"
    tc.HorizFeed = q(t.get("feed_mm_min") or 300, "mm/min")
    tc.VertFeed = q(t.get("plunge_feed_mm_min") or 100, "mm/min")
    tc.HorizRapid = q(5000, "mm/min")
    tc.VertRapid = q(3000, "mm/min")
    job.Proxy.addToolController(tc)
    return tc, bit


# ----------------------------------------------------------------------- operations
def create_op(doc, job, base, o, tcs):
    import Path.Op.Drilling as PathDrilling
    import Path.Op.MillFace as PathMillFace
    import Path.Op.Pocket as PathPocket
    import Path.Op.Profile as PathProfile
    kind = o["op_type"]
    name = "Op%02d_%s" % (o["step"], kind)
    faces = [f for f in o.get("faces", []) if f]
    if kind == "MillFace":
        op = PathMillFace.Create(name, parentJob=job) if "parentJob" in PathMillFace.Create.__code__.co_varnames else PathMillFace.Create(name)
        op.Base = [(base, faces)] if faces else []
        # "Stock" boundary ignores ClearEdges and leaves the stock corners un-faced under a round cutter;
        # "Boundbox" + ClearEdges extends the cleared area by the tool radius beyond the face
        op.BoundaryShape = "Boundbox"
        if hasattr(op, "ClearEdges"):
            op.ClearEdges = True
        if hasattr(op, "ClearingPattern"):
            op.ClearingPattern = "ZigZag"
        op.StepOver = int(max(10, min(90, round(100 * (o.get("woc_mm") or 0.7 * o["tool_diameter"]) / o["tool_diameter"]))))
        op.ZigZagAngle = 0
    elif kind == "Pocket":
        op = PathPocket.Create(name, parentJob=job) if "parentJob" in PathPocket.Create.__code__.co_varnames else PathPocket.Create(name)
        op.Base = [(base, faces)]
        op.StepOver = int(max(10, min(90, round(100 * (o.get("woc_mm") or 0.4 * o["tool_diameter"]) / o["tool_diameter"]))))
        if hasattr(op, "ClearingPattern"):
            op.ClearingPattern = "Offset"  # contour-parallel passes: clean walls (plain zig-zag leaves scallops)
        if hasattr(op, "ExtraOffset"):
            op.ExtraOffset = q(o.get("finish_stock") or 0.0)
        if hasattr(op, "UseOutline"):
            op.UseOutline = False
    elif kind == "Adaptive":
        # Adaptive clearing / inside profile: robust on filleted pockets where PocketShape drops the floor pass
        import Path.Op.Adaptive as PathAdaptive
        op = PathAdaptive.Create(name, parentJob=job) if "parentJob" in PathAdaptive.Create.__code__.co_varnames else PathAdaptive.Create(name)
        op.Base = [(base, faces)]
        op.OperationType = "Profiling" if o.get("adaptive_mode") == "profile" else "Clearing"
        op.Side = "Inside"
        op.StepOver = int(max(10, min(60, round(100 * (o.get("woc_mm") or 0.3 * o["tool_diameter"]) / o["tool_diameter"]))))
        op.Tolerance = 0.05
        if hasattr(op, "FinishingProfile"):
            op.FinishingProfile = bool(o.get("finish_stock", 0) == 0)
        if hasattr(op, "StockToLeave"):
            op.StockToLeave = q(o.get("finish_stock") or 0.0)
        if hasattr(op, "HelixAngle"):
            op.HelixAngle = 3.0
    elif kind in ("Adaptive3D", "Surface3D", "Waterline3D"):
        # CAM_FREECAD_3D=1 (experiment): FreeCAD's own 3D operations on the whole model instead of the harness's height-map
        # paths. Adaptive3D - model-aware adaptive clearing (layers from the stock top down to the model bottom or the
        # reach limit, the allowance in XY and Z); Surface3D - OCL drop-cutter, planar zig-zag over the model's box;
        # Waterline3D - OCL waterline contours (steep walls)
        mb = base.Shape.BoundBox
        lim = o.get("reach_limit_mm")
        o["final_depth"] = max(mb.ZMin, mb.ZMax - float(lim)) if lim else mb.ZMin
        d = o["tool_diameter"]
        if kind == "Adaptive3D":
            import Path.Op.Adaptive as PathAdaptive
            op = PathAdaptive.Create(name, parentJob=job)
            op.Base = []
            op.OperationType = "Clearing"
            op.Side = "Inside"
            op.ModelAwareExperiment = True
            op.StepOver = int(max(10, min(60, round(100 * (o.get("woc_mm") or 0.3 * d) / d))))
            op.StockToLeave = q(o.get("finish_stock") or 0.0)
            op.ZStockToLeave = q(o.get("finish_stock") or 0.0)
            op.Tolerance = 0.05
            op.HelixAngle = 3.0
            o["start_depth"] = job.Stock.Shape.BoundBox.ZMax
        elif kind == "Surface3D":
            import Path.Op.Surface as PathSurface
            op = PathSurface.Create(name, parentJob=job)
            op.Base = []
            op.ScanType = "Planar"
            op.LayerMode = "Single-pass"
            op.CutPattern = "ZigZag"
            op.BoundBox = "BaseBoundBox"
            op.StepOver = int(max(5, min(50, round(100 * (o.get("woc_mm") or 0.1 * d) / d))))
            op.SampleInterval = q(float(os.environ.get("CAM_FC3D_SAMPLE", "0.2")))
            op.DepthOffset = q(o.get("finish_stock") or 0.0)
        else:
            import Path.Op.Waterline as PathWaterline
            op = PathWaterline.Create(name, parentJob=job)
            op.Base = []
            op.Algorithm = "OCL Dropcutter"
            op.BoundBox = "BaseBoundBox"
            op.SampleInterval = q(float(os.environ.get("CAM_FC3D_SAMPLE", "0.2")))
            o["doc_mm"] = float(os.environ.get("CAM_FC3D_WATERLINE_STEP", "0.5"))
    elif kind == "Custom":
        # toolpath generated outside FreeCAD (height-map strategies); it still gets this job's tool change, post and FCStd
        with open(o["gcode_file"]) as fh:
            lines = custom_program_lines(fh.read())
        if not lines:
            raise ValueError("Custom program is empty")
        import Path.Op.Custom as PathCustom
        op = PathCustom.Create(name, parentJob=job) if "parentJob" in PathCustom.Create.__code__.co_varnames else PathCustom.Create(name)
        if hasattr(op, "Source"):
            op.Source = "Text"
        op.Gcode = lines
    elif kind == "Drilling":
        op = PathDrilling.Create(name, parentJob=job) if "parentJob" in PathDrilling.Create.__code__.co_varnames else PathDrilling.Create(name)
        op.Base = [(base, faces)]
        if o.get("peck_mm"):
            op.PeckEnabled = True
            op.PeckDepth = q(o["peck_mm"])
        if hasattr(op, "ExtraOffset"):
            # drawing depth of a hole is the full-diameter depth: add the drill point (not for spot drills / taps)
            op.ExtraOffset = "None" if (o.get("a34_tip_endpoint") or o.get("a35_tip_endpoint") or o.get("process") in ("spot_drilling", "tapping", "thread_milling", "reaming", "boring")) else "Drill Tip"
        if hasattr(op, "RetractMode") and o.get("tapping"):
            op.RetractMode = "G99"
    elif kind == "Profile":
        op = PathProfile.Create(name, parentJob=job) if "parentJob" in PathProfile.Create.__code__.co_varnames else PathProfile.Create(name)
        op.Base = []  # whole model outline
        op.Side = "Outside"
        op.Direction = "CW"  # climb milling
        if hasattr(op, "processHoles"):
            op.processHoles = False
            op.processPerimeter = True
        if hasattr(op, "OffsetExtra"):
            op.OffsetExtra = q(o.get("finish_stock") or 0.0)
    elif kind == "Deburr":
        import Path.Op.Deburr as PathDeburr
        op = PathDeburr.Create(name, parentJob=job) if "parentJob" in PathDeburr.Create.__code__.co_varnames else PathDeburr.Create(name)
        top = base.Shape.getElement(faces[0]) if faces else None
        ch_faces = [base.Shape.getElement(n) for n in o.get("chamfer_faces", [])]
        edge_names = []
        if top is not None:
            for i, e in enumerate(base.Shape.Edges, start=1):
                if not any(e.isSame(te) for te in top.Edges):
                    continue
                on_chamfer = any(e.isSame(ce) for cf in ch_faces for ce in cf.Edges)
                if (o.get("deburr_mode") == "modelled_chamfer") == on_chamfer:
                    edge_names.append("Edge%d" % i)
        if o.get("deburr_mode") == "modelled_chamfer":
            # the chamfer already exists in the model: run the tool flank through its upper edge (width ~0)
            # and plunge the tip below its lower edge, so the flank reproduces the modelled chamfer face
            op.Base = [(base, edge_names)]
            op.Width = q(0.02)
            op.ExtraDepth = q(float(o.get("chamfer_size") or 0.5) + 0.2)
        else:
            op.Base = [(base, edge_names or faces)]
            op.Width = q(o.get("chamfer_size") or 0.3)
            op.ExtraDepth = q(0.1)
    else:
        raise ValueError("unsupported op_type " + kind)
    op.Label = o.get("label") or name
    op.ToolController = tcs[o["tc"]]
    if o.get("coolant_mode") and hasattr(op, "CoolantMode"):
        try:
            op.CoolantMode = o["coolant_mode"]  # None | Flood | Mist -> M8 / M7 in the post
        except Exception as e:  # noqa: BLE001
            log("coolant", op.Label, e)
    if o.get("doc_mm") and hasattr(op, "StepDown"):
        # StepDown is bound to the setup-sheet expression (tool diameter) by default: unbind it first
        try:
            op.setExpression("StepDown", None)
        except Exception:  # noqa: BLE001
            pass
        op.StepDown = q(o["doc_mm"])
    for key, prop in (("start_depth", "StartDepth"), ("final_depth", "FinalDepth")):
        if o.get(key) is not None and hasattr(op, prop):
            try:
                op.setExpression(prop, None)
            except Exception:  # noqa: BLE001
                pass
            setattr(op, prop, q(o[key]))
    if o.get("a34_tip_endpoint") and hasattr(op, "RetractHeight"):
        op.setExpression("RetractHeight", None)
        op.RetractHeight = q(o["start_depth"])
    if o.get("tapping"):
        # FreeCAD's Drilling op writes G81/G83: cam_agent.services.postfix rewrites this operation to G84 (or G33.1)
        op.Comment = "TAPPING %s pitch %s: rewritten to a rigid tapping cycle by the harness" % (o.get("thread", ""), o.get("pitch_mm", ""))
    return op


# ---------------------------------------------------------------------- post
def post_process(job, ops, post_name, out_path):
    """Controller-aware supported exporter; fail closed on export/NC errors."""
    return export_job(job, ops, post_name, out_path)


def write_builtin_gcode(ops, out_path):
    lines = ["(Generated by cam_agent builtin post)", "G21 G17 G90 G94 G40 G49 G54", "G0 Z25.0"]
    last_tool = None
    for op in ops:
        tc = op.ToolController
        lines.append("(%s)" % op.Label)
        if tc.ToolNumber != last_tool:
            lines.append("M5")
            lines.append("T%d M6" % tc.ToolNumber)
            lines.append("S%d M3" % int(tc.SpindleSpeed))
            lines.append("M8")
            last_tool = tc.ToolNumber
        for c in op.Path.Commands:
            n = c.Name
            if n.startswith("("):
                lines.append(n)
                continue
            parts = [n] + ["%s%.3f" % (k, v) for k, v in c.Parameters.items()]
            lines.append(" ".join(parts))
    lines += ["M9", "M5", "G0 Z25.0", "M30"]
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --------------------------------------------------------------------- verify
def verify(job, ops, stock_bb, model_bb, tools_by_number):
    report = {"operations": [], "issues": [], "warnings": []}
    total_len, total_time_min = 0.0, 0.0
    for op in ops:
        p = op.Path
        tc = op.ToolController
        d = float(tc.Tool.Diameter.Value) if hasattr(tc.Tool, "Diameter") else 0
        x = y = None
        z = 25.0
        feed_len = rapid_len = 0.0
        min_z = 1e9
        rapid_in_material = 0
        max_x = max_y = -1e9
        min_x = min_y = 1e9
        for c in p.Commands:
            if c.Name not in ("G0", "G00", "G1", "G01", "G2", "G02", "G3", "G03", "G81", "G82", "G83", "G73", "G84", "G85"):
                continue
            prm = c.Parameters
            nx, ny, nz = prm.get("X", x), prm.get("Y", y), prm.get("Z", z)
            canned = c.Name in ("G81", "G82", "G83", "G73", "G84", "G85")
            if x is not None and nx is not None and ny is not None:
                seg = math.sqrt((nx - x) ** 2 + (ny - y) ** 2 + (nz - z) ** 2)
                if c.Name in ("G0", "G00"):
                    rapid_len += seg
                    if min(z, nz) < stock_bb[5] - 1e-6 and (abs(nx - x) > 1e-6 or abs(ny - y) > 1e-6):
                        rapid_in_material += 1
                else:
                    feed_len += seg
                    f = prm.get("F", float(tc.HorizFeed.Value) or 1)  # Path commands store feed in mm/s (internal unit)
                    total_time_min += seg / max(f, 1e-6) / 60.0
            if nz is not None:
                min_z = min(min_z, nz)
            if nx is not None:
                min_x, max_x = min(min_x, nx), max(max_x, nx)
            if ny is not None:
                min_y, max_y = min(min_y, ny), max(max_y, ny)
            if canned:  # the spindle returns to the retract plane after a canned cycle
                nz = prm.get("R", z if z is not None else nz)
            x, y, z = nx, ny, nz
        depths = {k: round(float(getattr(op, k).Value), 3) for k in ("StartDepth", "FinalDepth", "StepDown", "SafeHeight", "ClearanceHeight") if hasattr(op, k)}
        try:
            depths["passes"] = [round(float(h), 3) for h in op.Proxy.depthparams]
        except Exception:  # noqa: BLE001
            pass
        entry = {"op": op.Label, "tool": "T%d" % tc.ToolNumber, "tool_diameter": d, "commands": len(p.Commands), "depths": depths,
                 "feed_length_mm": round(feed_len, 1), "rapid_length_mm": round(rapid_len, 1), "min_z": round(min_z, 3) if min_z < 1e9 else None,
                 "xy_extent": [round(min_x, 2), round(min_y, 2), round(max_x, 2), round(max_y, 2)] if max_x > -1e9 else None}
        if len(p.Commands) < 3:
            report["issues"].append("%s produced no toolpath" % op.Label)
        breakthrough = (0.35 * d + 1.5) if "Drilling" in op.Name else 0.5  # drill tip + breakthrough on through holes
        if min_z < 1e9 and min_z < stock_bb[2] - breakthrough:
            report["issues"].append("%s cuts %.2f mm below the stock bottom (%.2f)" % (op.Label, stock_bb[2] - min_z, stock_bb[2]))
        if min_z < 1e9 and min_z < model_bb[2] - 0.05 and "Profile" not in op.Name and "Drilling" not in op.Name:
            report["warnings"].append("%s reaches below the model bottom (%.2f < %.2f)" % (op.Label, min_z, model_bb[2]))
        if rapid_in_material and "Adaptive" not in op.Name:  # adaptive links move rapidly inside the already cleared area
            report["issues"].append("%s has %d rapid (G0) XY moves below the stock top" % (op.Label, rapid_in_material))
        if max_x > -1e9 and (min_x < stock_bb[0] - d - 5 or max_x > stock_bb[3] + d + 5 or min_y < stock_bb[1] - d - 5 or max_y > stock_bb[4] + d + 5):
            report["warnings"].append("%s toolpath extends far outside the stock" % op.Label)
        report["operations"].append(entry)
        total_len += feed_len + rapid_len
    report["total_path_length_mm"] = round(total_len, 1)
    report["estimated_cutting_time_min"] = round(total_time_min, 1)
    report["stock_bbox"] = stock_bb
    report["model_bbox"] = model_bb
    report["ok"] = not report["issues"]
    return report


# ----------------------------------------------------------------------- main
def retry_empty_path(op):
    """Existing production empty-path fallback; callable for saved-op recovery."""
    n = len(op.Path.Commands) if op.Path else 0
    if n < 3 and hasattr(op, "HelixDiameterLimit"):
        # The default entry helix can fail in a narrow pocket. Keep this
        # historical 0.1D fallback only after the original path is empty.
        try:
            op.HelixDiameterLimit = q(0.1 * float(op.ToolController.Tool.Diameter))
            op.recompute()
            n = len(op.Path.Commands) if op.Path else 0
        except Exception as e:  # noqa: BLE001
            log("helix retry", op.Label, e)
    return n


def run(req):
    import Path.Main.Job as PathJob
    patch_toolcontroller_lookup()
    out_dir = req["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    doc = FreeCAD.newDocument("cam")
    base, translation = load_base(doc, req["path"], req.get("up", "+Z"), req.get("translation"))
    job = PathJob.Create("Job", [base], None)
    job.PostProcessor = req.get("post", "linuxcnc")
    job.PostProcessorOutputFile = os.path.join(out_dir, "program.ngc")
    st = req.get("stock", {})
    if st.get("solid"):
        stock = use_stock_solid(doc, job, st["solid"], req.get("up", "+Z"), translation)
    else:
        stock = job.Stock
        # stock-from-base sizes the box by the model's loose BoundBox: take the excess off each side, so the stock is the
        # exact box plus the margins (brep2cam@savrasov)
        loose, exact = base.Shape.BoundBox, base.Shape.optimalBoundingBox()
        m, zb, zt = st.get("xy_margin", 2), st.get("z_bottom", 0), st.get("z_top", 1)
        for prop, val in (("ExtXneg", m - (exact.XMin - loose.XMin)), ("ExtXpos", m - (loose.XMax - exact.XMax)),
                          ("ExtYneg", m - (exact.YMin - loose.YMin)), ("ExtYpos", m - (loose.YMax - exact.YMax)),
                          ("ExtZneg", zb - (exact.ZMin - loose.ZMin)), ("ExtZpos", zt - (loose.ZMax - exact.ZMax))):
            if hasattr(stock, prop):
                setattr(stock, prop, q(val))
    # default tool controllers created with the job are removed to keep the tool table clean
    for tc in list(job.Tools.Group):
        try:
            doc.removeObject(tc.Name)
        except Exception:  # noqa: BLE001
            pass
    tcs, bits = {}, {}
    tools = {t["tool_number"]: t for t in req["tools"]}
    for c in req.get("controllers") or [{"tc": t["tool_number"], "tool_number": t["tool_number"]} for t in req["tools"]]:
        t = tools.get(c["tool_number"])
        if t is None:
            continue
        try:
            # a tool bit per controller: a ToolController claims its bit as a child, so bits are not shared
            tcs[c["tc"]], bits[c["tc"]] = make_toolcontroller(doc, job, t, c)
        except Exception as e:  # noqa: BLE001
            log("tool controller failed for", t.get("tool_id"), e, traceback.format_exc()[-800:])
    ops, op_errors = [], []
    for o in req["operations"]:
        o.setdefault("tc", o["tool_number"])
        if o["tc"] not in tcs:
            op_errors.append({"step": o["step"], "error": "no tool controller for T%s" % o["tool_number"]})
            continue
        try:
            o["tool_diameter"] = float(req_tool(req, o["tool_number"]).get("diameter_mm") or 6)
            op = create_op(doc, job, base, o, tcs)
            ops.append(op)
            if o["op_type"] == "Surface3D" and o.get("waterline"):  # steep walls: OCL waterline after the drop-cutter pass
                ops.append(create_op(doc, job, base, {**o, "op_type": "Waterline3D", "label": "%s waterline" % o.get("label", "")}, tcs))
        except Exception as e:  # noqa: BLE001
            op_errors.append({"step": o["step"], "error": "%s: %s" % (type(e).__name__, e), "traceback": traceback.format_exc()[-1200:]})
    doc.recompute()
    for op in ops:  # second pass: ops sometimes need the job fully set up
        if op.Name.endswith(("Surface3D", "Waterline3D")) and op.Path and len(op.Path.Commands) >= 3:
            continue  # OCL drop-cutter: minutes per computation, already done
        try:
            op.recompute()
        except Exception as e:  # noqa: BLE001
            log("recompute", op.Label, e)
    doc.recompute()
    valid_ops = []
    for op in ops:
        n = retry_empty_path(op)
        if n < 3:
            op_errors.append({"step": int(op.Name[2:4]), "error": "%s computed an empty path" % op.Label})
        else:
            valid_ops.append(op)
    gcode_path = os.path.join(out_dir, "program.ngc")
    post_used = None
    if valid_ops:
        post_used = post_process(job, valid_ops, req.get("post", "linuxcnc"), gcode_path)
    sbb = stock.Shape.BoundBox
    mbb = base.Shape.optimalBoundingBox()
    stock_bb = [round(sbb.XMin, 3), round(sbb.YMin, 3), round(sbb.ZMin, 3), round(sbb.XMax, 3), round(sbb.YMax, 3), round(sbb.ZMax, 3)]
    model_bb = [round(mbb.XMin, 3), round(mbb.YMin, 3), round(mbb.ZMin, 3), round(mbb.XMax, 3), round(mbb.YMax, 3), round(mbb.ZMax, 3)]
    ver = verify(job, valid_ops, stock_bb, model_bb, tcs)
    ver["op_errors"] = op_errors
    if op_errors:
        ver["ok"] = False
    fcstd = os.path.join(out_dir, "cam_job.FCStd")
    doc.saveAs(fcstd)
    gcode_lines = 0
    if os.path.exists(gcode_path):
        with open(gcode_path) as f:
            gcode_lines = sum(1 for _ in f)
    return {"job_file": fcstd, "gcode_file": gcode_path if os.path.exists(gcode_path) else None, "gcode_lines": gcode_lines,
            "post": post_used, "operations": [op.Label for op in valid_ops], "verification": ver, "log": LOG,
            "freecad_version": ".".join(FreeCAD.Version()[:3]), "stock_solid": bool(st.get("solid"))}


def req_tool(req, number):
    for t in req["tools"]:
        if t["tool_number"] == number:
            return t
    return {}


main(run)
