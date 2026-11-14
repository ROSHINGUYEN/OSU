"""
Mathematical Solver for osu! Standard Slider Curves.
=====================================================
Computes authentic smooth trajectory waypoints for:
1. Perfect Circle Arc (P): Circumscribed circle through 3 control points.
2. Multi-segment Bézier (B): De Casteljau evaluation partitioned by duplicated vertices.
3. Linear (L) & Catmull-Rom (C): Direct polygonal or spline interpolation.
Ensures cursor stays locked to the center of the slider ball, preventing slider breaks.
"""

import math
from typing import List, Optional, Tuple

from src.gameplay.osu.beatmap.models import CurveType, Slider


def de_casteljau(points: List[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """De Casteljau's algorithm for evaluating Bézier curves at parameter t in [0, 1]."""
    pts = list(points)
    n = len(pts)
    for r in range(1, n):
        for i in range(n - r):
            pts[i] = (
                (1.0 - t) * pts[i][0] + t * pts[i + 1][0],
                (1.0 - t) * pts[i][1] + t * pts[i + 1][1],
            )
    return pts[0]


def sample_bezier_segments(
    points: List[Tuple[float, float]],
    num_points_per_seg: int = 21,
) -> List[Tuple[float, float]]:
    """Evaluates multi-segment Bézier curves partitioned by duplicate consecutive points."""
    segments: List[List[Tuple[float, float]]] = []
    curr: List[Tuple[float, float]] = []

    for p in points:
        if curr and p == curr[-1]:
            if len(curr) >= 2:
                segments.append(curr)
            curr = [p]
        else:
            curr.append(p)
    if len(curr) >= 2:
        segments.append(curr)

    if not segments:
        return points

    sampled: List[Tuple[float, float]] = []
    for seg in segments:
        steps = max(4, num_points_per_seg)
        for i in range(steps):
            t = i / float(steps - 1)
            pt = de_casteljau(seg, t)
            if not sampled or pt != sampled[-1]:
                sampled.append((round(pt[0], 2), round(pt[1], 2)))
    return sampled


def sample_perfect_circle(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    pixel_length: float = 0.0,
    num_points: int = 25,
) -> List[Tuple[float, float]]:
    """Computes circumscribed circle arc through 3 control points matching osu! mechanics."""
    d = 2 * (p0[0] * (p1[1] - p2[1]) + p1[0] * (p2[1] - p0[1]) + p2[0] * (p0[1] - p1[1]))
    if abs(d) < 1e-4:
        # Collinear fallback: straight line connecting points
        pts = [p0, p1, p2]
        if pixel_length > 0.0:
            return clip_points_to_length(pts, pixel_length)
        return pts

    ux = (
        (p0[0] ** 2 + p0[1] ** 2) * (p1[1] - p2[1])
        + (p1[0] ** 2 + p1[1] ** 2) * (p2[1] - p0[1])
        + (p2[0] ** 2 + p2[1] ** 2) * (p0[1] - p1[1])
    ) / d
    uy = (
        (p0[0] ** 2 + p0[1] ** 2) * (p2[0] - p1[0])
        + (p1[0] ** 2 + p1[1] ** 2) * (p0[0] - p2[0])
        + (p2[0] ** 2 + p2[1] ** 2) * (p1[0] - p0[0])
    ) / d
    r = math.hypot(p0[0] - ux, p0[1] - uy)
    if r < 1e-4 or r > 20000.0:
        pts = [p0, p1, p2]
        if pixel_length > 0.0:
            return clip_points_to_length(pts, pixel_length)
        return pts

    a0 = math.atan2(p0[1] - uy, p0[0] - ux)
    a1 = math.atan2(p1[1] - uy, p1[0] - ux)
    a2 = math.atan2(p2[1] - uy, p2[0] - ux)

    # Angular displacement along unit circle S^1 in positive (CCW) direction
    d1 = (a1 - a0) % (2.0 * math.pi)
    d2 = (a2 - a0) % (2.0 * math.pi)

    # Determine travel orientation: CCW if traversing CCW visits p1 before p2
    if d1 < d2:
        direction = 1.0
        da = d2
    else:
        direction = -1.0
        da = d2 - (2.0 * math.pi)

    if d2 == 0.0:
        direction = 1.0 if d1 > 0 else -1.0
        da = direction * 2.0 * math.pi

    # Clamp by pixel_length if provided
    if pixel_length > 0.0 and r > 0.0:
        angle_span = direction * (pixel_length / r)
    else:
        angle_span = da

    steps = max(6, num_points)
    pts: List[Tuple[float, float]] = []
    for i in range(steps):
        frac = i / float(steps - 1)
        ang = a0 + frac * angle_span
        pts.append((round(ux + r * math.cos(ang), 2), round(uy + r * math.sin(ang), 2)))
    return pts


def sample_catmull_rom(
    points: List[Tuple[float, float]],
    pixel_length: float = 0.0,
    num_points_per_seg: int = 16,
) -> List[Tuple[float, float]]:
    """Evaluates authentic uniform Catmull-Rom spline through control points matching osu! legacy rules."""
    if len(points) < 2:
        return points
    if len(points) == 2:
        pts = [points[0], points[1]]
        return clip_points_to_length(pts, pixel_length) if pixel_length > 0.0 else pts

    n = len(points)
    sampled: List[Tuple[float, float]] = []

    for i in range(n - 1):
        p0 = points[max(0, i - 1)]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[min(n - 1, i + 2)]

        chord = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        steps = max(6, min(40, int(chord / 5.0))) if chord > 0 else num_points_per_seg

        for s in range(steps):
            t = s / float(steps)
            t2 = t * t
            t3 = t2 * t
            x = 0.5 * (
                (2.0 * p1[0])
                + (-p0[0] + p2[0]) * t
                + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
                + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
            )
            y = 0.5 * (
                (2.0 * p1[1])
                + (-p0[1] + p2[1]) * t
                + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
                + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
            )
            pt = (round(x, 2), round(y, 2))
            if not sampled or pt != sampled[-1]:
                sampled.append(pt)

    final_pt = (round(points[-1][0], 2), round(points[-1][1], 2))
    if not sampled or final_pt != sampled[-1]:
        sampled.append(final_pt)

    if pixel_length > 0.0:
        return clip_points_to_length(sampled, pixel_length)
    return sampled


def clip_points_to_length(pts: List[Tuple[float, float]], target_len: float) -> List[Tuple[float, float]]:
    """Clip or extend a polyline to exact pixel_length matching osu! engine rules."""
    if not pts or target_len <= 0.0:
        return pts
    clipped: List[Tuple[float, float]] = [pts[0]]
    curr_len = 0.0
    for i in range(len(pts) - 1):
        p1 = pts[i]
        p2 = pts[i + 1]
        seg_dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        if seg_dist < 1e-6:
            continue
        if curr_len + seg_dist >= target_len:
            remain = target_len - curr_len
            t = remain / seg_dist
            clipped.append((
                round(p1[0] + (p2[0] - p1[0]) * t, 3),
                round(p1[1] + (p2[1] - p1[1]) * t, 3),
            ))
            return clipped
        curr_len += seg_dist
        clipped.append(p2)

    # Extension rule matching ppy/osu SliderPath.cs:
    # If total curve length is shorter than target_len, extend along final direction
    if curr_len < target_len and len(clipped) >= 2:
        p_last = clipped[-1]
        p_prev = clipped[-2]
        dx = p_last[0] - p_prev[0]
        dy = p_last[1] - p_prev[1]
        seg_dist = math.hypot(dx, dy)
        if seg_dist > 1e-6:
            remain = target_len - curr_len
            clipped.append((
                round(p_last[0] + (dx / seg_dist) * remain, 3),
                round(p_last[1] + (dy / seg_dist) * remain, 3),
            ))
    return clipped


def calculate_slider_path(hit: Slider) -> List[Tuple[float, float]]:
    """
    Calculates the true continuous path points for any osu! slider.
    Evaluates curves so the cursor follows the authentic slider ball path.
    """
    all_ctrl = [(hit.x, hit.y)]
    if hasattr(hit, "curve_points") and hit.curve_points:
        all_ctrl.extend(hit.curve_points)

    if len(all_ctrl) < 2:
        return all_ctrl

    curve_type = getattr(hit, "curve_type", CurveType.BEZIER)
    c_val = curve_type.value if hasattr(curve_type, "value") else str(curve_type)
    pixel_len = getattr(hit, "pixel_length", 0.0)

    # 1. Perfect Circle Arc (P) with exactly 3 control points
    if c_val == "P" and len(all_ctrl) == 3:
        return sample_perfect_circle(all_ctrl[0], all_ctrl[1], all_ctrl[2], pixel_len)

    # 2. Catmull-Rom Spline (C)
    if c_val == "C":
        return sample_catmull_rom(all_ctrl, pixel_len)

    # 3. Bézier Curve (B)
    if c_val == "B" or len(all_ctrl) > 3:
        raw_pts = sample_bezier_segments(all_ctrl)
        if pixel_len > 0.0:
            return clip_points_to_length(raw_pts, pixel_len)
        return raw_pts

    # 4. Linear (L) or fallback
    if pixel_len > 0.0:
        return clip_points_to_length(all_ctrl, pixel_len)
    return all_ctrl
