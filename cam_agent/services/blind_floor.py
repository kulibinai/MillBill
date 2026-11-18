"""Conservative A35 existing helical-counterbore tool eligibility."""
import math


def floor_tool(tools, hole, iso):
    if not hole.get('accepted'):
        return None
    eligible=[]
    for t in tools:
        sp=t.get('shape_params') or {}; d=float(t.get('diameter_mm') or 0)
        # Full cutting edge below raw-stock top avoids shank-wall contact entirely.
        if (t.get('category')!='endmill' or iso not in t.get('iso',[]) or
                t.get('corner_radius_mm',0) or not 0<d<hole['diameter']-.1 or
                float(t.get('flute_length_mm') or 0)<hole['depth']+.01 or
                float(t.get('overall_length_mm') or 0)-25<hole['depth']+.01):
            continue
        # The existing helix's first swept circle must cover the centre (no core).
        orbit=min((hole['diameter']-d)/2,.3*d)
        if orbit>=d/2-.01:
            continue
        eligible.append(t)
    return max(eligible,key=lambda t:float(t['diameter_mm']),default=None)


def tip_height(tool, diameter):
    angle=float((tool.get('shape_params') or {}).get('TipAngle') or 0)
    if tool.get('category')!='twist_drill' or not 0<angle<180 or diameter<=0:
        return None
    return diameter/2/math.tan(math.radians(angle/2))
