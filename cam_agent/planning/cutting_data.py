"""Cutting-data model (speeds, feeds, engagement) for a tool / process / material / machine.

  n  = Vc * 1000 / (pi * D)                     spindle speed, clamped to the machine
  vf = fz * z * n   (milling)   vf = f * n (drilling, reaming)   vf = pitch * n (tapping)
  fz = k_group * D  (solid endmills), raised by radial chip thinning D / (2 sqrt(ae (D - ae))) when ae < D/2
  ap, ae from the strategy (adaptive / rough / slot / finish / face) and the ISO 513 group
  Pc = ap * ae * vf * kc / (60e6 * eta),  kc = kc1.1 * hm^-mc (Kienzle)    -> ap (then vf) reduced to the machine power
  Mc = Pc * 30e3 / (pi * n)                                                  -> same for the spindle torque

Values are starting values of the kind tool makers publish; the machine profile (knowledge/data/machine.json) bounds them.
"""
from __future__ import annotations

import math

from ..knowledge import get_store

MILLING = ("endmill", "ball_endmill", "bull_endmill", "face_mill", "chamfer_mill")
HOLE_TOOLS = ("twist_drill", "spot_drill", "reamer", "boring_bar", "tap", "countersink")


def tool_key(tool: dict) -> str:
    return "HSS" if str(tool.get("tool_material", "carbide")).upper().startswith("HSS") else "carbide"


def strategy_for(process: str, op_type: str | None = None, geom: dict | None = None, tool_d: float | None = None) -> str:
    """Engagement strategy of a milling step."""
    geom = geom or {}
    if process == "facing":
        return "face"
    if process.endswith("finishing") or process in ("fillet_milling", "chamfering", "deburring"):
        return "finish"
    if op_type == "Adaptive" or process == "pocket_milling":
        return "adaptive"
    if process == "slot_milling" and tool_d and geom.get("width") and float(geom["width"]) <= 1.2 * tool_d:
        return "slot"
    return "rough"


def coolant_mode(material: dict) -> str:
    """FreeCAD CoolantMode (None | Flood | Mist) from the material's coolant recommendation."""
    c = str(material.get("coolant", "flood")).lower()
    if c.startswith("flood"):
        return "Flood"
    if "mql" in c or "mist" in c:
        return "Mist"
    return "None"  # dry or air blast (FreeCAD has no air-blast mode)


def _mean_chip_thickness(fz: float, ae: float, d: float, kappa_deg: float = 90.0) -> float:
    """Sandvik: hm = 360 sin(kr) ae fz / (pi D arccos(1 - 2 ae / D))   (arccos in degrees)."""
    ae = min(max(ae, 1e-6), d)
    phi = math.degrees(math.acos(max(-1.0, 1.0 - 2.0 * ae / d)))
    return 360.0 * math.sin(math.radians(kappa_deg)) * ae * fz / (math.pi * d * max(phi, 1e-6))


def cutting_data(tool: dict, process: str, material: dict | None, geom: dict | None = None, strategy: str | None = None,
                 op_type: str | None = None) -> dict:
    """Speeds, feeds and engagement for one step. `geom` may carry diameter, depth, width, pitch, thread, top_allowance."""
    store = get_store()
    mach = store.machine
    geom = geom or {}
    material = material or _fallback_material()
    iso = material.get("iso_group", "P")
    grp = store.iso_groups.get(iso, store.iso_groups["P"])
    cat = tool.get("category", "endmill")
    key = tool_key(tool)
    vtab = (material.get("cutting_speed_m_min") or {}).get(key) or {}
    notes: list[str] = []
    d = float(tool.get("diameter_mm") or geom.get("diameter") or 6.0)
    z = int(tool.get("flutes") or 2)
    stick = float(tool.get("overall_length_mm") or (tool.get("shape_params") or {}).get("Length") or 4 * d) - 30.0
    long_reach = cat in MILLING and d > 0 and stick / d > store.cutting["long_reach"]["l_over_d"]
    lr = store.cutting["long_reach"]

    out: dict = {"tool_diameter_mm": round(d, 3), "flutes": z, "coolant": material.get("coolant", "flood"), "coolant_mode": coolant_mode(material)}
    peck = None
    max_rpm = float(mach["max_rpm"])
    if cat == "tap":
        spec = store.thread(geom.get("thread")) or {}
        d = float(spec.get("major_mm") or d)
        pitch = float(geom.get("pitch") or spec.get("pitch") or 1.0)
        vc = float(vtab.get("tap") or 0)
        if vc <= 0:
            return {"error": f"{tool.get('id')}: tapping is not rated for {material.get('name')}"}
        max_rpm = min(max_rpm, float(mach.get("tap_max_rpm", max_rpm)))
        n = min(vc * 1000 / (math.pi * d), max_rpm)
        f_rev = pitch
        vf = pitch * n
        out.update(tool_diameter_mm=round(d, 3), feed_per_rev_mm=pitch, doc_mm=round(float(geom.get("depth") or d), 2), woc_mm=round(d, 2),
                   cycle="G84 rigid tapping", pitch_mm=pitch)
        vc = n * math.pi * d / 1000
        return _finish(out, vc, n, vf, vf, None, None, notes, mach)

    if cat in ("twist_drill", "spot_drill", "reamer", "boring_bar", "countersink"):
        if cat == "twist_drill":
            vc = float(vtab.get("drill") or 0)
            f_rev = min(max(grp["f_drill_k"] * d, 0.02), 0.5) * (0.7 if key == "HSS" else 1.0)
            ratio = float(geom.get("depth") or 0) / max(d, 1e-6)
            limit = 3.0 if key == "HSS" else 5.0
            if ratio > limit:
                peck = round((1.5 if iso in ("N", "K") else 1.0) * d, 1)
                notes.append(f"depth {ratio:.1f}xD > {limit:g}xD: peck {peck} mm (G83)")
            if ratio > float(tool.get("max_depth_x_d") or 99):
                notes.append(f"depth {ratio:.1f}xD exceeds this drill's {tool.get('max_depth_x_d')}xD")
        elif cat == "spot_drill":
            vc = 0.8 * float(vtab.get("drill") or vtab.get("mill_finish") or 0)
            f_rev = min(max(0.5 * grp["f_drill_k"] * d, 0.02), 0.2)
        elif cat == "countersink":
            vc = 0.5 * float(vtab.get("drill") or 0)
            f_rev = min(max(0.4 * grp["f_drill_k"] * d, 0.03), 0.25)
        elif cat == "reamer":
            vc = float(vtab.get("ream") or 0)
            f_rev = round(grp["f_ream"] * (0.5 + d / 20.0), 3)
        else:  # boring bar
            vc = 0.8 * float(vtab.get("mill_finish") or 0)
            f_rev = 0.08 if iso in ("N", "P", "K") else 0.05
        if vc <= 0:
            return {"error": f"{tool.get('id')}: {key} {cat} is not rated for {material.get('name')}"}
        n = min(vc * 1000 / (math.pi * d), max_rpm)
        vf = f_rev * n
        out.update(feed_per_rev_mm=round(f_rev, 4), doc_mm=round(float(geom.get("depth") or d), 2), woc_mm=round(d, 2), peck_mm=peck)
        # drilling power and torque (Sandvik): Pc = f vc D kc / 240000 kW, hm = f/2 sin(kr)
        kappa = float((tool.get("shape_params") or {}).get("TipAngle") or 118.0) / 2.0
        hm = max(f_rev / 2.0 * math.sin(math.radians(kappa)), 0.005)
        kc = float(material.get("kc11_n_mm2", 1700)) * hm ** (-float(material.get("mc", 0.25)))
        vc_eff = n * math.pi * d / 1000
        pc = f_rev * vc_eff * d * kc / 240000.0 if cat == "twist_drill" else None
        plunge = vf
        if cat == "reamer" or cat == "boring_bar":
            out["cycle"] = "G85 feed in / feed out"
        elif cat == "twist_drill":
            out["cycle"] = "G83 peck" if peck else "G81"
        return _finish(out, vc_eff, n, vf, plunge, pc, kc, notes, mach)

    if cat == "thread_mill":
        vc = float(vtab.get("thread_mill") or vtab.get("mill_finish") or 0)
        if vc <= 0:
            return {"error": f"{tool.get('id')}: thread milling not rated for {material.get('name')}"}
        spec = store.thread(geom.get("thread")) or {}
        n = min(vc * 1000 / (math.pi * d), max_rpm)
        fz = grp["fz_thread_mill"]
        vf_periphery = fz * z * n
        major = float(spec.get("major_mm") or (d + 2))
        vf = vf_periphery * max(major - d, 0.1) / major  # programmed at the tool centre for an internal thread
        out.update(feed_per_tooth_mm=fz, doc_mm=round(float(geom.get("depth") or d), 2), woc_mm=round(float(spec.get("pitch") or 1.0) * 0.6, 2),
                   cycle="helical interpolation (G2/G3), climb")
        return _finish(out, n * math.pi * d / 1000, n, vf, 0.3 * vf, None, None, notes, mach)

    # ---------------------------------------------------------------- milling
    strategy = strategy or strategy_for(process, op_type, geom, d)
    eng = store.engagement[strategy].get(iso, store.engagement[strategy]["P"])
    if cat == "face_mill" or strategy == "face":
        vc = float(vtab.get("face") or vtab.get("mill_rough") or 0)
        fz = grp["fz_face"] * (0.7 if strategy == "finish" else 1.0)
        top = geom.get("top_allowance")
        ap = min(float(eng[0]), float(tool.get("max_doc_mm") or 3.0))
        if top:
            ap = min(ap, max(float(top), 0.2))
        ae = eng[1] * d
    else:
        vc = float(vtab.get("mill_finish" if strategy == "finish" else "mill_rough") or 0)
        if cat == "chamfer_mill":
            vc *= 0.6
            d_eff = max(d / 2.0, 1.0)
        else:
            d_eff = d
        fz = min(max(grp["fz_mill_k"] * d_eff, 0.005), 0.25) * (0.6 if key == "HSS" else 1.0)
        if cat == "ball_endmill":
            fz *= 0.8
        lc = float(tool.get("flute_length_mm") or d)
        if strategy == "finish":
            allowance = store.cutting["finish_allowance_mm"]
            ap = min(lc, float(geom.get("depth") or d))
            ae = max(eng[1] * d, allowance["wall"] if d >= allowance["small_tool_d_mm"] else allowance["small_tool_wall"])
            fz *= 0.8
        else:
            ap = min(eng[0] * d, lc)
            ae = min(eng[1] * d, d)
            if strategy == "slot" and geom.get("width"):
                ae = min(float(geom["width"]), d)
        if ae < 0.5 * d:
            thin = d / (2.0 * math.sqrt(ae * (d - ae)))
            thin = min(thin, store.cutting["chip_thinning"]["max_factor"])
            if thin > 1.01:
                notes.append(f"radial chip thinning x{thin:.2f} (ae {ae / d:.0%} of D)")
            fz *= thin
        if long_reach:
            vc *= lr["vc_factor"]
            fz *= lr["fz_factor"]
            ap *= lr["ap_factor"]
            notes.append(f"long reach (stick-out {stick / d:.1f}xD): Vc x{lr['vc_factor']}, fz x{lr['fz_factor']}, ap x{lr['ap_factor']}")
    if vc <= 0:
        return {"error": f"{tool.get('id')}: {key} milling not rated for {material.get('name')}"}
    if key == "carbide" and str(tool.get("coating", "")).lower() in ("uncoated", "none", "") and iso != "N":
        vc *= 0.8
    n = min(vc * 1000 / (math.pi * d), max_rpm)
    vf = fz * z * n
    # Kienzle power / torque check
    hm = _mean_chip_thickness(fz, ae, d, 45.0 if cat == "face_mill" else 90.0)
    kc = float(material.get("kc11_n_mm2", 1700)) * max(hm, 0.005) ** (-float(material.get("mc", 0.25)))
    pc = ap * ae * min(vf, float(mach["max_feed_mm_min"])) * kc / 60e6
    avail = float(mach["spindle_power_kw"]) * float(mach.get("efficiency", 0.8))
    if pc > avail and strategy != "finish":
        scale = avail / pc
        ap *= scale
        pc *= scale
        notes.append(f"spindle power: ap reduced to {ap:.2f} mm ({avail:.1f} kW available)")
    plunge = vf * (0.5 if strategy == "finish" else 0.3)
    out.update(strategy=strategy, feed_per_tooth_mm=round(fz, 4), doc_mm=round(ap, 2), woc_mm=round(ae, 2))
    return _finish(out, n * math.pi * d / 1000, n, vf, plunge, pc, kc, notes, mach)


def _finish(out: dict, vc: float, n: float, vf: float, plunge: float, pc: float | None, kc: float | None, notes: list, mach: dict) -> dict:
    max_f = float(mach["max_feed_mm_min"])
    if vf > max_f:
        notes.append(f"feed clamped to the machine maximum {max_f:.0f} mm/min")
        plunge = plunge * max_f / vf
        vf = max_f
    if n >= float(mach["max_rpm"]) - 1:
        notes.append(f"spindle speed clamped to {mach['max_rpm']} rpm (Vc {vc:.0f} m/min)")
    torque = pc * 30e3 / (math.pi * max(n, 1.0)) if pc else None
    if torque and torque > float(mach["spindle_torque_nm"]):
        notes.append(f"spindle torque {torque:.0f} Nm exceeds the machine ({mach['spindle_torque_nm']} Nm): reduce the depth or use a smaller tool")
    out.update(cutting_speed_m_min=round(vc, 1), spindle_rpm=int(round(n / 10) * 10), feed_mm_min=int(max(5, round(vf / 5) * 5)),
               plunge_feed_mm_min=int(max(5, round(plunge / 5) * 5)), power_kw=round(pc, 2) if pc else None,
               torque_nm=round(torque, 1) if torque else None, kc_n_mm2=round(kc) if kc else None, notes=notes)
    out.setdefault("peck_mm", None)
    return out


def _fallback_material() -> dict:
    """Unknown material: steel C45 data, so that the plan still has conservative numbers."""
    store = get_store()
    return dict(store.material("C45") or store.materials[0], name="unknown material (C45 data assumed)")
