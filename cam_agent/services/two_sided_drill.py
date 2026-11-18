"""Default-off A34: recognised through bore, two drilling-only suffix setups.

The baseline plan/CAM runs first. No ideal future A34 view can affect it.
Both actual blank-entry boxes and both delivered half paths must be certified
before either half is admitted to the final workpiece/session.
"""
from __future__ import annotations
import copy
import hashlib
import itertools
import math
from pathlib import Path
import re
import time
import numpy as np

from .gcode import parse, cycle_time_min
from .verify import ops_meta_for
from .voxel import Frame, OPPOSITE

OVERLAP = 0.1
MARGIN = 0.003  # covers 0.001mm native post rounding and finite checks


def prefix_program(path, native_operations=None):
    text = Path(path).read_text()
    clean = "\n".join(re.sub(r"\(.*?\)"," ",line.split(";")[0]) for line in text.splitlines())
    # NC words may be compact (G21G90, G1G92). A word boundary would miss
    # every second G word after a numeric value.
    codes = [float(x) for x in re.findall(r"G\s*([-+]?(?:\d+\.?\d*|\.\d+))", clean, re.I)]
    # Arcs, coordinate shifts, compensation, other work offsets and units are
    # rejected, never approximated into a sufficient-volume certificate.
    allowed = {0,1,17,21,40,49,54,80,81,82,83,84,85,86,87,88,89,90,94,98,99}
    if set(codes)-allowed or not {21,90}.issubset(codes):
        raise ValueError("A34 unsupported prefix modal/arc/coordinate mode")
    parsed = parse(str(path))
    if parsed["warnings"]:
        raise ValueError("A34 prefix parse warning")
    # A bounded T/M6 guard, not another motion parser. Reject changes inside an
    # operation: the shared parser otherwise retroactively changes op.tool for
    # earlier moves. Exports without T/M6 require the actual kernel object's
    # operation/ToolController report, never request-only tool inference.
    active=pending=None
    opened=None
    timeline={}
    explicit=False
    for raw in text.splitlines():
        begin=re.search(r"\(\s*begin operation\s*:\s*(.*)\)\s*$",raw.strip(),re.I)
        finish=re.search(r"\(\s*finish operation\s*:",raw,re.I)
        if begin:
            if re.sub(r"\(.*\)","",raw.split(";")[0]).strip():
                raise ValueError("A34 code mixed with operation marker")
            opened=begin.group(1).strip()
            if opened in timeline:
                raise ValueError("A34 duplicate operation label")
            timeline[opened]=active
            continue
        if finish:
            if re.sub(r"\(.*\)","",raw.split(";")[0]).strip():
                raise ValueError("A34 code mixed with operation marker")
            opened=None
            continue
        line=re.sub(r"\(.*?\)"," ",raw.split(";")[0]).upper()
        words=re.findall(r"([A-Z])\s*([-+]?(?:\d+\.?\d*|\.\d+))",line)
        if any(float(value)!=int(float(value)) for word,value in words if word=="M"):
            raise ValueError("A34 noninteger M code")
        ts=[float(value) for word,value in words if word=="T"]
        changes=[float(value) for word,value in words if word=="M" and float(value)==6]
        if ts or changes:
            explicit=True
            if opened is not None:
                raise ValueError("A34 T/M6 inside operation is unsupported")
            if len(ts)>1 or len(changes)>1 or any(t<1 or t!=int(t) for t in ts):
                raise ValueError("A34 ambiguous tool timeline")
            if ts:
                pending=int(ts[0])
            if changes:
                if pending is None:
                    raise ValueError("A34 M6 without selected tool")
                active=pending
    parsed["_a34_tools"]=timeline
    parsed["_a34_explicit_tool_timeline"]=explicit
    bindings={}
    for item in native_operations or []:
        if item.get("op") in bindings:
            raise ValueError("A34 duplicate native tool binding")
        bindings[item.get("op")]=item
    parsed["_a34_native_tools"]=bindings
    return parsed


def bound_tool(parsed, op, tool):
    timeline=parsed.get("_a34_tools")
    if timeline is None or op["name"] not in timeline or not tool.get("tool_id"):
        raise ValueError("A34 missing actual operation/tool binding")
    if parsed.get("_a34_explicit_tool_timeline"):
        actual=timeline[op["name"]]
        if actual is None or actual!=tool.get("tool_number"):
            raise ValueError("A34 actual tool timeline mismatches request")
    native=parsed.get("_a34_native_tools",{}).get(op["name"])
    if not parsed.get("_a34_explicit_tool_timeline") and native is None:
        raise ValueError("A34 no actual tool identity: native binding required")
    if native is not None:
        if native.get("tool")!=f"T{tool.get('tool_number')}" or abs(float(native.get("tool_diameter") or 0)-float(tool.get("diameter_mm") or 0))>1e-7:
            raise ValueError("A34 actual native ToolController mismatches request")
    return dict(source="explicit T/M6 timeline" if parsed.get("_a34_explicit_tool_timeline") else "delivered best kernel ToolController report",
                actual_timeline_tool=timeline[op["name"]],native_operation=native)


def clearing_witness(corners, parsed, operations, tools, frame, catalog):
    points = frame.to_setup(np.asarray(corners, float))
    for item, op in zip(ops_meta_for(parsed, operations, tools), parsed["ops"]):
        tool = item.get("tool") or {}
        binding=bound_tool(parsed,op,tool)
        lib = catalog.get(tool.get("tool_id"), {})
        if tool.get("category") != "endmill" or lib.get("category") != "endmill":
            continue
        flute, radius = float(lib.get("flute_length_mm") or 0), float(tool.get("diameter_mm") or 0)/2
        if flute <= 0 or radius <= 0:
            continue
        for index, move in enumerate(op["moves"]):
            a,b = np.asarray(move[:3]),np.asarray(move[3:6])
            if not move[6] or abs(a[2]-b[2]) > 1e-9:
                continue
            if points[:,2].min() < a[2]-1e-9 or points[:,2].max() > a[2]+flute-MARGIN:
                continue
            d = b[:2]-a[:2]; square = float(d@d)
            t = np.clip(((points[:,:2]-a[:2])@d)/square,0,1) if square else np.zeros(len(points))
            dist = np.linalg.norm(points[:,:2]-(a[:2]+t[:,None]*d),axis=1)
            if dist.max() <= radius-MARGIN:
                return dict(op=op["name"],move=index,segment=list(move),tool_id=tool["tool_id"],
                            radius_mm=radius,flute_mm=flute,max_corner_radius_mm=float(dist.max()),tool_binding=binding)
    return None


def prepare_pairs(recognition, setups, stock, store, iso, maximum, freecad, cad_path, workdir):
    """Only existing recognised complete matching face sets; no feature search."""
    from ..agents.tool_selection import ToolSelectionAgent
    ts = ToolSelectionAgent(None, mode="deterministic")
    catalog = {t["id"]:t for t in store.tool_library["tools"]}
    pairs, seen = [], set()
    if len(setups)+2 > maximum or not iso:
        return []
    for setup in setups:
        up = setup["up"]; other = OPPOSITE[up]
        r0,r1 = recognition[up], recognition[other]
        f0,f1 = Frame(up,r0["translation"]), Frame(other,r1["translation"])
        for h in r0["features"]:
            key = tuple(sorted(h.get("faces",[])))
            if h.get("type") != "hole" or not h.get("vertical") or not h.get("through") or key in seen:
                continue
            seen.add(key)
            if h.get("subtype") not in (None,"hole") or h.get("thread") or h.get("tolerance_grade"):
                continue
            annotations=[f for entry in setups for f in entry.get("features",[])
                         if tuple(sorted(f.get("faces",[])))==key]
            if any(f.get("thread") or f.get("tolerance_grade") or f.get("subtype") not in (None,"hole") for f in annotations):
                continue
            if not np.isfinite([h.get("diameter",0),h.get("depth",0),*h.get("top",[]),*h.get("bottom_point",[])]).all() or min(h.get("diameter",0),h.get("depth",0))<=0:
                continue
            matches = [g for g in r1["features"] if g.get("type")=="hole" and tuple(sorted(g.get("faces",[])))==key]
            if len(matches)!=1 or not matches[0].get("vertical") or not matches[0].get("through"):
                continue
            g=matches[0]
            e0=f0.to_original(np.array([h["top"],h["bottom_point"]]))
            e1=f1.to_original(np.array([g["top"],g["bottom_point"]]))
            if not np.allclose(e0,e1[::-1],atol=1e-6,rtol=0) or abs(h["diameter"]-g["diameter"])>1e-6:
                continue
            geom=ts.geometry("drilling",[h])
            if ts.candidates("drilling",iso,geom):
                continue
            half=h["depth"]/2+OVERLAP/2
            candidates=ts.candidates("drilling",iso,{**geom,"depth":half})
            candidates=[t for t in candidates if t["category"]=="twist_drill" and t.get("shape_params",{}).get("TipAngle")]
            supported=[]
            for t in candidates:
                tip=h["diameter"]/(2*math.tan(math.radians(float(t["shape_params"]["TipAngle"])/2)))
                limit=float(t["max_depth_x_d"])*h["diameter"]
                if half+tip+MARGIN <= limit:
                    supported.append((t,tip,limit))
            if not supported:
                continue
            chosen=ts.primary("drilling",[t for t,tip,limit in supported],geom)
            t,tip,limit=next(x for x in supported if x[0]["id"]==chosen["id"])
            parts=[]
            for u,r,hh,frame in ((up,r0,h,f0),(other,r1,g,f1)):
                bb=frame.to_setup(np.asarray(list(itertools.product(*zip(stock["bbox"][:3],stock["bbox"][3:])))))
                z0,z1=float(bb[:,2].min()),float(bb[:,2].max())
                cert=freecad.run("certify_drill_pair.py",dict(path=str(cad_path),up=u,translation=r["translation"],hole=hh,stock_z=[z0,z1]),workdir/f"a34-{len(pairs)}-{u}")
                if not cert.get("accepted"):
                    break
                # Use all available catalog reach, including the tip; stock
                # above StartDepth must have a real preceding clearing witness.
                final=round(float(hh["top_z"])-half-tip,3)
                start=math.floor(min(z1,final+limit-MARGIN)*1000)/1000
                if start < float(hh["top_z"]):
                    break
                radius=hh["diameter"]/2
                lo=[hh["top"][0]-radius,hh["top"][1]-radius,start]
                hi=[hh["top"][0]+radius,hh["top"][1]+radius,z1]
                corners=frame.to_original(np.asarray(list(itertools.product(*zip(lo,hi))))).tolist()
                partial=copy.deepcopy(hh)
                partial.update(depth=start-final,top=[hh["top"][0],hh["top"][1],start],top_z=start,
                               bottom_point=[hh["top"][0],hh["top"][1],final])
                contract=dict(tool_id=t["id"],catalog_limit_mm=limit,tip_mm=tip,start_depth=start,final_depth=final,
                              bore_top_z=float(hh["top_z"]),bore_depth_mm=float(hh["depth"]),
                              full_diameter_depth=half,overlap_mm=OVERLAP,entry_corners_original=corners,
                              geometric_certificate=cert,original_bore_endpoints=e0.tolist(),face_ids=list(key))
                partial["two_sided_drill"]=contract
                parts.append(dict(up=u,translation=r["translation"],part=r["part"],feature=partial))
            if len(parts)==2:
                pairs.append(parts)
                # One intrinsic pair per available two re-clamps, no budget expansion.
                if len(setups)+2*len(pairs)+2 > maximum:
                    return pairs
    return pairs


def validate_half(path, request, feature, native_operations=None):
    """Delivered postrounded native tip path; geometry metadata alone is insufficient."""
    c=feature["two_sided_drill"]; parsed=prefix_program(path,native_operations)
    metadata=ops_meta_for(parsed,request["operations"],request["tools"])
    if len(parsed["ops"])!=1 or len(metadata)!=1 or metadata[0].get("process")!="drilling":
        raise ValueError("A34 suffix must contain exactly one drilling operation")
    binding=bound_tool(parsed,parsed["ops"][0],metadata[0].get("tool") or {})
    if metadata[0]["tool"].get("tool_id")!=c["tool_id"]:
        raise ValueError("A34 native tool differs from certified catalog tool")
    if abs(float(metadata[0]["tool"].get("diameter_mm") or 0)-float(feature["diameter"]))>1e-7:
        raise ValueError("A34 native diameter differs from certified bore")
    moves=parsed["ops"][0]["moves"]
    cuts=[m for m in moves if m[6]]
    if not cuts:
        raise ValueError("A34 empty native half")
    if any(abs(m[i]-feature["top"][i])>1e-6 or abs(m[i+3]-feature["top"][i])>1e-6 for m in cuts for i in (0,1)):
        raise ValueError("A34 native axis mismatch")
    low=min(min(m[2],m[5]) for m in cuts)
    high=max(max(m[2],m[5]) for m in cuts)
    if high<c["start_depth"]-1e-9 or high>c["start_depth"]+0.0011 or abs(low-c["final_depth"])>0.0011 or high-low>c["catalog_limit_mm"]+1e-9:
        raise ValueError("A34 native tip depth exceeds contract/catalog")
    return dict(min_tip_z=low,max_cut_start_z=high,actual_tip_depth_mm=high-low,
                cylindrical_reach_from_bore_entry_mm=c["bore_top_z"]-low-c["tip_mm"],
                tool_binding=binding,
                gcode_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def deliver_suffix(orch, session, plan, cad_path, fused, stock_info, stock_raw, verifier):
    """Called only AFTER all original setups. Failed pair remains diagnostic only."""
    from ..agents.cam_generation import CAMGenerationAgent
    from ..agents.tool_selection import ToolSelectionAgent
    from .postfix import fix_program, safety_report
    from .verify import SetupState
    from . import exact
    recognition=plan.get("a34_recognition")
    if not recognition or verifier is None:
        return
    iso=(fused.get("material_data") or {}).get("iso_group")
    wd=Path(session["session_dir"])
    catalog={t["id"]:t for t in orch.store.tool_library["tools"]}
    diagnostic=[]
    session["setup_plan"]["two_sided_drill"] = diagnostic
    try:
        pairs=prepare_pairs(recognition,session["setups"],stock_raw,orch.store,iso,
                            plan["a34_max_setups"],orch.freecad,cad_path,wd/"fc")
    except Exception as exc:
        diagnostic.append(dict(status="rejected",reason=f"geometry certificate failed: {type(exc).__name__}: {exc}"))
        return
    for pair in pairs:
        started=time.monotonic()
        record={"status":"rejected", "parts":pair, "clearing":[]}
        diagnostic.append(record)
        staged=[]
        try:
            # Both entries are proved from the actual fully delivered base prefix.
            for part in pair:
                proof=None
                for entry in session["setups"]:
                    cam=entry.get("cam") or {}; req=cam.get("request") or {}
                    path=cam.get("gcode_file")
                    if not path or not req:
                        continue
                    try:
                        parsed=prefix_program(path,(cam.get("verification") or {}).get("operations"))
                    except ValueError:
                        continue
                    found=clearing_witness(part["feature"]["two_sided_drill"]["entry_corners_original"],
                        parsed,req["operations"],req["tools"],Frame(entry["up"],entry["translation"]),catalog)
                    if found:
                        proof={"setup":entry["index"],"gcode_sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),**found}
                        break
                if proof is None:
                    raise ValueError("A34 actual delivered entry clearing not certified for "+part["up"])
                record["clearing"].append(proof)
            for offset,part in enumerate(pair,1):
                k=len(session["setups"])+offset
                feature=copy.deepcopy(part["feature"])
                feature.update(id=f"S{k}A34",setup=k,subtype="hole",material=fused["material"]["id"])
                fk={**fused,"features":[feature],"part":part["part"],"process_annotations":[],"user_sequence_override":[]}
                ps={"sequence":[{"step":1,"process":"drilling","feature_ids":[feature["id"]]}],
                    "source":"generated"}
                ts=ToolSelectionAgent(orch.llm,mode="deterministic").run(fk,ps).output
                from ..planning import sequencing as SEQ
                ps["validation"]=SEQ.validate(ps["sequence"],fk["features"],fused["material"]["group"],
                                               fk,orch.s.sequencing,ts["assignments"],final_setup=False)
                if ts["assignments"][0].get("tool_id")!=feature["two_sided_drill"]["tool_id"]:
                    raise ValueError("A34 actual TS/catalog tool mismatch")
                out=wd/f"setup{k}_{part['up'].replace('+','p').replace('-','m')}_a34"
                out.mkdir(parents=True,exist_ok=True)
                agent=CAMGenerationAgent(orch.llm,orch.freecad,orch.s.post_processor,mode="deterministic",settings=orch.s)
                req=agent.build_request(str(cad_path),out,fk,ps,ts,stock_info,part["up"],part["translation"])
                result=orch.freecad.run("build_cam_job.py",req,out/"fc")
                if result.get("verification",{}).get("op_errors") or not result.get("gcode_file"):
                    raise ValueError("A34 native operation failed")
                result["postfix"]=fix_program(result["gcode_file"],req["operations"],orch.s.tap_cycle,orch.s.tap_m29)
                native_operations=(result.get("verification") or {}).get("operations")
                parsed=prefix_program(result["gcode_file"],native_operations)
                if safety_report(parsed,req["operations"]):
                    raise ValueError("A34 native safety report failed")
                native=validate_half(result["gcode_file"],req,feature,native_operations)
                result.setdefault("verification",{})["cycle_time"]=cycle_time_min(
                    parsed,orch.store.machine["rapid_mm_min"],orch.store.machine["tool_change_s"])
                result.update(request=req,available=True,skipped=req["skipped"],two_sided_native=native,
                              rounds=[{"round":0,"op_errors":[],"issues":[],"warnings":[]}])
                staged.append({"index":k,"up":part["up"],"translation":part["translation"],"features":[feature],
                               "ps":ps,"ts":ts,"cam":result,"a34_suffix":True})
            # Check delivered full-diameter coverage from both native tip extrema.
            reaches=[]
            for part,entry in zip(pair,staged):
                contract=part["feature"]["two_sided_drill"]
                native=entry["cam"]["two_sided_native"]
                reaches.append(contract["bore_top_z"]-native["min_tip_z"]-contract["tip_mm"])
            length=float(np.linalg.norm(np.subtract(pair[0]["feature"]["two_sided_drill"]["original_bore_endpoints"][0],
                                                    pair[0]["feature"]["two_sided_drill"]["original_bore_endpoints"][1])))
            if min(reaches)<=0 or sum(reaches)-length < OVERLAP-0.0021:
                raise ValueError("A34 delivered full-diameter union/overlap failed")
            # Atomic admission: no first half is included if the other failed.
            for entry in staged:
                k=entry["index"]; fr=Frame(entry["up"],entry["translation"])
                verifier.frames[k]=fr
                verifier.states[k]=SetupState(fr,ideal=verifier._ideal(fr))
                verifier.ideal_exact[k]=exact.IdealView(fr,verifier.part_raw)
            for entry in staged:
                cam=entry["cam"]
                dv=verifier.check_setup(entry["index"],cam["gcode_file"],cam["request"],entry["features"])
                for private in ("_sim","_inside","_frame"):
                    dv.pop(private,None)
                ver=cam.setdefault("verification",{})
                ver["deviation"]=dv
                ver.setdefault("issues",[]).extend(f"deviation: {e['kind']} on {e.get('feature','unknown')}" for e in dv.get("errors",[]))
                ver["ok"]=not ver["issues"] and not ver.get("op_errors")
                cam["rounds"][0].update(issues=ver["issues"],deviation=dv.get("stats",{}),ok=ver["ok"])
            session["setups"].extend(staged)
            record.update(status="delivered",native_cylindrical_reaches_mm=reaches,
                          delivered_overlap_mm=sum(reaches)-length,indices=[e["index"] for e in staged])
        except Exception as exc:
            for entry in staged:
                for collection in (verifier.frames,verifier.states,verifier.ideal_exact,verifier.sweeps,verifier.results,verifier.programs):
                    collection.pop(entry["index"],None)
            record["reason"]=f"{type(exc).__name__}: {exc}"
        record["wall_s"]=time.monotonic()-started
    session["orientation"]["setups"]=[e["up"] for e in session["setups"]]
