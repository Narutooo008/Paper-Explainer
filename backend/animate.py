"""Renders animated 1280x720 explainer scenes (24 fps) with Pillow."""
import math
import os
from pathlib import Path
from typing import Iterator

from PIL import Image, ImageDraw

import config
from config import quality_mode
from slides import BG, H, INK, MARK, MUTED, TRACK, W, font, shorten, wrap

FPS = 24
PAD = 0.7


def quality_spec(quality: str | None = None) -> tuple[int, int, int]:
    """Return (width, height, fps) for 'fast' (960x540 @ 15fps) or 'normal' (1280x720 @ 24fps)."""
    q = (quality or config.quality_mode()).strip().lower()
    if q == "fast":
        return 960, 540, 15
    return W, H, FPS


COBALT = (43, 80, 255)
STAGE_BG = (20, 35, 64)
STAGE_BORDER = (38, 58, 98)
NODE_BG = (26, 45, 82)
NODE_ACTIVE_BG = (48, 58, 44)
COL_LEFT_BG = (25, 40, 70)
COL_RIGHT_BG = (24, 46, 90)
GRID_LINE = (32, 50, 86)


def clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def ease_out(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return 1.0 - (1.0 - t) ** 3


def lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    t = clamp(t, 0.0, 1.0)
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


def _visual_step_count(visual: dict) -> int:
    vtype = (visual or {}).get("type", "none")
    if vtype == "flow":
        return len(visual.get("nodes", []))
    if vtype == "compare":
        return max(len(visual.get("left_items", [])), len(visual.get("right_items", [])))
    if vtype == "bar_chart":
        return len(visual.get("bars", []))
    if vtype == "cycle":
        return len(visual.get("steps", []))
    if vtype == "figure":
        return 1
    if vtype == "outro":
        return len(visual.get("takeaways", [])) or 3
    return 0


def _draw_centered_lines(
    d: ImageDraw.ImageDraw,
    lines: list,
    fnt,
    cx: float,
    cy: float,
    line_h: int,
    fill: tuple,
) -> None:
    total_h = len(lines) * line_h
    y = cy - total_h / 2
    for line in lines:
        w = d.textlength(line, font=fnt)
        d.text((cx - w / 2, y), line, font=fnt, fill=fill)
        y += line_h


def _draw_arrow(
    d: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    progress: float,
    color: tuple,
    width: int = 6,
    head_len: float = 20.0,
) -> None:
    p = ease_out(progress)
    if p <= 0.02:
        return
    ex = x1 + (x2 - x1) * p
    ey = y1 + (y2 - y1) * p
    d.line([(x1, y1), (ex, ey)], fill=color, width=width)

    angle = math.atan2(y2 - y1, x2 - x1)
    a1 = angle + math.pi * 0.82
    a2 = angle - math.pi * 0.82
    p1 = (ex + head_len * math.cos(a1), ey + head_len * math.sin(a1))
    p2 = (ex + head_len * math.cos(a2), ey + head_len * math.sin(a2))
    d.polygon([(ex, ey), p1, p2], fill=color)


def _box_anchor(
    ux: float, uy: float, wx: float, wy: float, hw: float, hh: float, pad: float = 8.0
) -> tuple:
    dx, dy = wx - ux, wy - uy
    if abs(dx) >= abs(dy) * 0.9:
        sx = ux + (hw + pad if dx > 0 else -(hw + pad))
        sy = uy
        tx = wx - (hw + pad if dx > 0 else -(hw + pad))
        ty = wy
    else:
        sx = ux
        sy = uy + (hh + pad if dy > 0 else -(hh + pad))
        tx = wx
        ty = wy - (hh + pad if dy > 0 else -(hh + pad))
    return sx, sy, tx, ty


def _render_flow_ss(
    d: ImageDraw.ImageDraw,
    sw: int,
    sh: int,
    visual: dict,
    step_prog: list,
    active_idx: int,
    ss: int,
) -> None:
    nodes = visual.get("nodes", [])
    edges = visual.get("edges", [])
    n = len(nodes)
    if n < 2:
        return

    pad_x, pad_y = 22 * ss, 36 * ss
    usable_w = sw - 2 * pad_x

    # Auto-layout nodes so they never overlap or leave the stage
    centers = []
    if n <= 3:
        node_w = min(156 * ss, (usable_w - (n - 1) * 64 * ss) // n)
        node_h = 80 * ss
        gap = (usable_w - n * node_w) / max(1, n - 1) if n > 1 else 0
        for i in range(n):
            cx = pad_x + node_w / 2 + i * (node_w + gap)
            centers.append((cx, sh / 2))
    else:
        top_n = (n + 1) // 2
        bot_n = n - top_n
        node_w = min(146 * ss, (usable_w - (top_n - 1) * 62 * ss) // top_n)
        node_h = 74 * ss
        top_y = pad_y + node_h / 2 + 26 * ss
        bot_y = sh - pad_y - node_h / 2 - 22 * ss

        top_gap = (usable_w - top_n * node_w) / max(1, top_n - 1) if top_n > 1 else 0
        for i in range(top_n):
            cx = pad_x + node_w / 2 + i * (node_w + top_gap)
            centers.append((cx, top_y))

        # Second row placed right-to-left for a clean serpentine flow
        bot_xs = []
        if bot_n == 1:
            bot_xs = [sw / 2]
        else:
            bot_gap = (usable_w - bot_n * node_w) / max(1, bot_n - 1)
            for j in range(bot_n):
                bot_xs.append(sw - pad_x - node_w / 2 - j * (node_w + bot_gap))
        for cx in bot_xs:
            centers.append((cx, bot_y))

    hw, hh = node_w / 2, node_h / 2
    lbl_font = font(18 * ss, bold=True)
    edge_font = font(13 * ss, bold=False)

    # Draw edges first
    for e in edges:
        u, w = e["from"], e["to"]
        if not (0 <= u < n and 0 <= w < n):
            continue
        target_step = max(u, w)
        ep = step_prog[target_step] if target_step < len(step_prog) else 0.0
        if ep <= 0.02:
            continue
        ux, uy = centers[u]
        wx, wy = centers[w]
        sx, sy, tx, ty = _box_anchor(ux, uy, wx, wy, hw, hh, pad=5 * ss)
        is_active = target_step == active_idx
        ecolor = lerp_color(STAGE_BG, MARK if is_active else COBALT, ep)
        _draw_arrow(d, sx, sy, tx, ty, ep, ecolor, width=4 * ss, head_len=13 * ss)

        elbl = e.get("label", "")
        if elbl and ep > 0.6:
            is_horiz = abs(tx - sx) >= abs(ty - sy)
            mx = (sx + tx) / 2
            my = (sy - 20 * ss) if is_horiz else (sy + ty) / 2
            tw = d.textlength(elbl, font=edge_font)
            d.rounded_rectangle(
                [mx - tw / 2 - 7 * ss, my - 10 * ss, mx + tw / 2 + 7 * ss, my + 10 * ss],
                radius=5 * ss,
                fill=STAGE_BG,
                outline=ecolor,
                width=max(1, ss),
            )
            d.text((mx - tw / 2, my - 8 * ss), elbl, font=edge_font, fill=MUTED)

    # Draw nodes with pop-in scaling
    badge_font = font(13 * ss, bold=True)
    for i, (cx, cy) in enumerate(centers):
        p = ease_out(step_prog[i]) if i < len(step_prog) else 0.0
        if p <= 0.02:
            continue
        scale = 0.76 + 0.24 * p
        cw, ch = hw * scale, hh * scale
        is_active = i == active_idx
        bg_col = lerp_color(STAGE_BG, NODE_ACTIVE_BG if is_active else NODE_BG, p)
        border_col = lerp_color(STAGE_BG, MARK if is_active else COBALT, p)
        text_col = lerp_color(STAGE_BG, MARK if is_active else INK, p)

        d.rounded_rectangle(
            [cx - cw, cy - ch, cx + cw, cy + ch],
            radius=12 * ss,
            fill=bg_col,
            outline=border_col,
            width=(3 if is_active else 2) * ss,
        )

        # Small step number pill in top-left corner of node
        br = 12 * ss
        bx, by = cx - cw + 14 * ss, cy - ch + 14 * ss
        d.ellipse([bx - br, by - br, bx + br, by + br], fill=border_col)
        num_s = str(i + 1)
        nw = d.textlength(num_s, font=badge_font)
        d.text(
            (bx - nw / 2, by - 8 * ss),
            num_s,
            font=badge_font,
            fill=BG if is_active else INK,
        )

        lines = wrap(d, nodes[i], lbl_font, int(node_w - 28 * ss))[:2]
        _draw_centered_lines(d, lines, lbl_font, cx, cy + 4 * ss, 23 * ss, text_col)


def _render_compare_ss(
    d: ImageDraw.ImageDraw,
    sw: int,
    sh: int,
    visual: dict,
    step_prog: list,
    active_idx: int,
    ss: int,
) -> None:
    left_title = visual.get("left_title", "Existing Approach")
    right_title = visual.get("right_title", "Paper's Approach")
    left_items = visual.get("left_items", [])
    right_items = visual.get("right_items", [])
    rows = max(len(left_items), len(right_items))
    if rows == 0:
        return

    pad_x, pad_y = 26 * ss, 26 * ss
    gap_x = 22 * ss
    col_w = (sw - 2 * pad_x - gap_x) // 2
    lx1, lx2 = pad_x, pad_x + col_w
    rx1, rx2 = lx2 + gap_x, lx2 + gap_x + col_w

    head_h = 46 * ss
    head_font = font(19 * ss, bold=True)

    # Column headers
    d.rounded_rectangle(
        [lx1, pad_y, lx2, pad_y + head_h],
        radius=8 * ss,
        fill=TRACK,
        outline=STAGE_BORDER,
        width=2 * ss,
    )
    _draw_centered_lines(
        d, [shorten(d, left_title, head_font, col_w - 20 * ss)], head_font, (lx1 + lx2) / 2, pad_y + head_h / 2 - 2 * ss, 22 * ss, MUTED
    )

    d.rounded_rectangle(
        [rx1, pad_y, rx2, pad_y + head_h],
        radius=8 * ss,
        fill=COBALT,
        outline=MARK,
        width=2 * ss,
    )
    _draw_centered_lines(
        d, [shorten(d, right_title, head_font, col_w - 20 * ss)], head_font, (rx1 + rx2) / 2, pad_y + head_h / 2 - 2 * ss, 22 * ss, INK
    )

    top_y = pad_y + head_h + 18 * ss
    avail_h = sh - top_y - pad_y
    row_gap = 12 * ss
    row_h = min(74 * ss, (avail_h - (rows - 1) * row_gap) // max(1, rows))
    item_font = font(17 * ss, bold=False)

    for r in range(rows):
        p = ease_out(step_prog[r]) if r < len(step_prog) else 0.0
        if p <= 0.02:
            continue
        ry1 = top_y + r * (row_h + row_gap)
        ry2 = ry1 + row_h
        is_active = r == active_idx

        # Left item slides in from left
        if r < len(left_items):
            dx = -int(24 * ss * (1.0 - p))
            bg_l = lerp_color(STAGE_BG, COL_LEFT_BG, p)
            bd_l = lerp_color(STAGE_BG, MARK if is_active else STAGE_BORDER, p)
            tx_l = lerp_color(STAGE_BG, INK, p)
            d.rounded_rectangle(
                [lx1 + dx, ry1, lx2 + dx, ry2],
                radius=8 * ss,
                fill=bg_l,
                outline=bd_l,
                width=(2 if is_active else 1) * ss,
            )
            lines = wrap(d, left_items[r], item_font, col_w - 24 * ss)[:2]
            _draw_centered_lines(
                d, lines, item_font, (lx1 + lx2) / 2 + dx, (ry1 + ry2) / 2 - 2 * ss, 21 * ss, tx_l
            )

        # Right item slides in from right
        if r < len(right_items):
            dx = int(24 * ss * (1.0 - p))
            bg_r = lerp_color(STAGE_BG, NODE_ACTIVE_BG if is_active else COL_RIGHT_BG, p)
            bd_r = lerp_color(STAGE_BG, MARK if is_active else COBALT, p)
            tx_r = lerp_color(STAGE_BG, MARK if is_active else INK, p)
            d.rounded_rectangle(
                [rx1 + dx, ry1, rx2 + dx, ry2],
                radius=8 * ss,
                fill=bg_r,
                outline=bd_r,
                width=(3 if is_active else 2) * ss,
            )
            lines = wrap(d, right_items[r], item_font, col_w - 24 * ss)[:2]
            _draw_centered_lines(
                d, lines, item_font, (rx1 + rx2) / 2 + dx, (ry1 + ry2) / 2 - 2 * ss, 21 * ss, tx_r
            )


def _fmt_num(val: float, unit: str) -> str:
    s = f"{int(val)}" if abs(val - round(val)) < 1e-4 else f"{val:.1f}"
    if not unit:
        return s
    return f"{s}{unit}" if unit in ("%", "x", "×") else f"{s} {unit}"


def _render_bar_chart_ss(
    d: ImageDraw.ImageDraw,
    sw: int,
    sh: int,
    visual: dict,
    step_prog: list,
    active_idx: int,
    ss: int,
) -> None:
    bars = visual.get("bars", [])
    unit = visual.get("unit", "")
    n = len(bars)
    if n == 0:
        return

    pad_x = 42 * ss
    base_y = sh - 86 * ss
    top_y = 66 * ss
    max_bar_h = base_y - top_y

    # Subtle horizontal grid lines + baseline
    for frac in (0.25, 0.5, 0.75, 1.0):
        gy = int(base_y - max_bar_h * frac)
        d.line([(pad_x, gy), (sw - pad_x, gy)], fill=GRID_LINE, width=ss)
    d.line([(pad_x, base_y), (sw - pad_x, base_y)], fill=MUTED, width=2 * ss)

    max_val = max(max(abs(float(b["value"])) for b in bars), 1e-6)
    usable_w = sw - 2 * pad_x
    slot_w = usable_w / n
    bar_w = min(74 * ss, int(slot_w * 0.56))

    val_font = font(18 * ss, bold=True)
    lbl_font = font(15 * ss, bold=False)

    for i, b in enumerate(bars):
        p = ease_out(step_prog[i]) if i < len(step_prog) else 0.0
        if p <= 0.02:
            continue
        val = max(0.0, float(b["value"]))
        bh = max(6 * ss, int((val / max_val) * max_bar_h * p))
        cx = pad_x + slot_w * (i + 0.5)
        x1, x2 = int(cx - bar_w / 2), int(cx + bar_w / 2)
        y1, y2 = base_y - bh, base_y

        is_active = i == active_idx
        fill_col = lerp_color(STAGE_BG, MARK if is_active else COBALT, p)
        outline_col = lerp_color(STAGE_BG, INK if is_active else MARK, p)

        d.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=6 * ss,
            fill=fill_col,
            outline=outline_col if is_active else fill_col,
            width=2 * ss,
        )

        # Value label above bar
        v_str = _fmt_num(float(b["value"]), unit)
        vw = d.textlength(v_str, font=val_font)
        v_col = lerp_color(STAGE_BG, MARK if is_active else INK, p)
        d.text((cx - vw / 2, y1 - 28 * ss), v_str, font=val_font, fill=v_col)

        # Category label below axis
        l_col = lerp_color(STAGE_BG, MARK if is_active else MUTED, p)
        lines = wrap(d, b["label"], lbl_font, int(slot_w - 8 * ss))[:2]
        _draw_centered_lines(d, lines, lbl_font, cx, base_y + 34 * ss, 19 * ss, l_col)


def _render_cycle_ss(
    d: ImageDraw.ImageDraw,
    sw: int,
    sh: int,
    visual: dict,
    step_prog: list,
    active_idx: int,
    ss: int,
) -> None:
    steps = visual.get("steps", [])
    n = len(steps)
    if n < 3:
        return

    cx, cy = sw / 2, sh / 2
    rx, ry = 192 * ss, 130 * ss
    node_w, node_h = 152 * ss, 62 * ss
    hw, hh = node_w / 2, node_h / 2

    # Subtle orbit ring
    d.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], outline=GRID_LINE, width=2 * ss)

    centers = []
    for i in range(n):
        ang = -math.pi / 2 + (2 * math.pi * i) / n
        centers.append((cx + rx * math.cos(ang), cy + ry * math.sin(ang)))

    # Draw cycle arrows between consecutive steps
    for i in range(n):
        p = step_prog[i] if i < len(step_prog) else 0.0
        if p <= 0.05:
            continue
        j = (i + 1) % n
        ux, uy = centers[i]
        wx, wy = centers[j]
        dx, dy = wx - ux, wy - uy
        dist = max(1.0, math.hypot(dx, dy))
        trim_s = 68 * ss
        trim_e = 68 * ss
        if dist > trim_s + trim_e + 16 * ss:
            sx = ux + (dx / dist) * trim_s
            sy = uy + (dy / dist) * trim_s
            tx = wx - (dx / dist) * trim_e
            ty = wy - (dy / dist) * trim_e
        else:
            sx, sy, tx, ty = _box_anchor(ux, uy, wx, wy, hw, hh, pad=6 * ss)

        is_active = i == active_idx
        acol = lerp_color(STAGE_BG, MARK if is_active else COBALT, p)
        _draw_arrow(d, sx, sy, tx, ty, p, acol, width=4 * ss, head_len=14 * ss)

    lbl_font = font(17 * ss, bold=True)
    for i, (nx, ny) in enumerate(centers):
        p = ease_out(step_prog[i]) if i < len(step_prog) else 0.0
        if p <= 0.02:
            continue
        scale = 0.76 + 0.24 * p
        cw, ch = hw * scale, hh * scale
        is_active = i == active_idx
        bg_col = lerp_color(STAGE_BG, NODE_ACTIVE_BG if is_active else NODE_BG, p)
        bd_col = lerp_color(STAGE_BG, MARK if is_active else COBALT, p)
        tx_col = lerp_color(STAGE_BG, MARK if is_active else INK, p)

        d.rounded_rectangle(
            [nx - cw, ny - ch, nx + cw, ny + ch],
            radius=12 * ss,
            fill=bg_col,
            outline=bd_col,
            width=(3 if is_active else 2) * ss,
        )
        lines = wrap(d, steps[i], lbl_font, int(node_w - 20 * ss))[:2]
        _draw_centered_lines(d, lines, lbl_font, nx, ny - 2 * ss, 20 * ss, tx_col)


_fig_cache = {}


def _render_figure_ss(
    hi: Image.Image,
    d: ImageDraw.ImageDraw,
    sw: int,
    sh: int,
    visual: dict,
    step_prog: list,
    active_idx: int,
    ss: int,
) -> None:
    fig_path_str = visual.get("path", "")
    if not fig_path_str:
        return
    p = ease_out(step_prog[0]) if step_prog else 1.0
    if p <= 0.02:
        return

    pad_x, pad_y = 26 * ss, 24 * ss
    cap_h = 42 * ss
    avail_w = sw - 2 * pad_x
    avail_h = sh - 2 * pad_y - cap_h

    if fig_path_str not in _fig_cache:
        try:
            with Image.open(fig_path_str) as raw:
                _fig_cache[fig_path_str] = raw.convert("RGB")
        except Exception:
            _fig_cache[fig_path_str] = None
    src_img = _fig_cache.get(fig_path_str)
    if src_img is None:
        return

    scale = (0.86 + 0.14 * p) * min(
        (avail_w - 24 * ss) / max(1, src_img.width),
        (avail_h - 24 * ss) / max(1, src_img.height),
    )
    tw = max(1, int(src_img.width * scale))
    th = max(1, int(src_img.height * scale))
    resized = src_img.resize((tw, th), resample=Image.Resampling.LANCZOS)

    cx = sw // 2
    cy = pad_y + avail_h // 2
    mat_pad = 10 * ss
    mx1, my1 = cx - tw // 2 - mat_pad, cy - th // 2 - mat_pad
    mx2, my2 = cx + tw // 2 + mat_pad, cy + th // 2 + mat_pad

    is_active = active_idx == 0
    bd_col = lerp_color(STAGE_BG, MARK if is_active else COBALT, p)
    d.rounded_rectangle(
        [mx1, my1, mx2, my2],
        radius=10 * ss,
        fill=(248, 250, 253),
        outline=bd_col,
        width=3 * ss,
    )
    hi.paste(resized, (cx - tw // 2, cy - th // 2))

    cap_font = font(16 * ss, bold=True)
    cap = shorten(d, visual.get("caption", "Figure from the paper"), cap_font, avail_w)
    cw = d.textlength(cap, font=cap_font)
    cap_col = lerp_color(STAGE_BG, MARK if is_active else INK, p)
    d.text((cx - cw / 2, sh - pad_y - 26 * ss), cap, font=cap_font, fill=cap_col)


def compute_diagram_boxes(visual: dict, sw: int = 640, sh: int = 445) -> list:
    """Return bounding boxes [(x1, y1, x2, y2), ...] of diagram items in stage coordinates."""
    vtype = (visual or {}).get("type", "none")
    boxes = []
    if vtype == "flow":
        n = len(visual.get("nodes", []))
        if n < 2:
            return []
        pad_x, pad_y = 22, 36
        usable_w = sw - 2 * pad_x
        if n <= 3:
            node_w = min(156, (usable_w - (n - 1) * 64) // n)
            node_h = 80
            gap = (usable_w - n * node_w) / max(1, n - 1) if n > 1 else 0
            for i in range(n):
                cx = pad_x + node_w / 2 + i * (node_w + gap)
                cy = sh / 2
                boxes.append((cx - node_w / 2, cy - node_h / 2, cx + node_w / 2, cy + node_h / 2))
        else:
            top_n = (n + 1) // 2
            bot_n = n - top_n
            node_w = min(146, (usable_w - (top_n - 1) * 62) // top_n)
            node_h = 74
            top_y = pad_y + node_h / 2 + 26
            bot_y = sh - pad_y - node_h / 2 - 22
            top_gap = (usable_w - top_n * node_w) / max(1, top_n - 1) if top_n > 1 else 0
            for i in range(top_n):
                cx = pad_x + node_w / 2 + i * (node_w + top_gap)
                boxes.append((cx - node_w / 2, top_y - node_h / 2, cx + node_w / 2, top_y + node_h / 2))
            if bot_n == 1:
                bot_xs = [sw / 2]
            else:
                bot_gap = (usable_w - bot_n * node_w) / max(1, bot_n - 1)
                bot_xs = [sw - pad_x - node_w / 2 - j * (node_w + bot_gap) for j in range(bot_n)]
            for cx in bot_xs:
                boxes.append((cx - node_w / 2, bot_y - node_h / 2, cx + node_w / 2, bot_y + node_h / 2))
    elif vtype == "cycle":
        n = len(visual.get("steps", []))
        if n < 3:
            return []
        cx, cy = sw / 2, sh / 2
        rx, ry = 192, 130
        node_w, node_h = 152, 62
        for i in range(n):
            ang = -math.pi / 2 + (2 * math.pi * i) / n
            nx = cx + rx * math.cos(ang)
            ny = cy + ry * math.sin(ang)
            boxes.append((nx - node_w / 2, ny - node_h / 2, nx + node_w / 2, ny + node_h / 2))
    elif vtype == "compare":
        rows = max(len(visual.get("left_items", [])), len(visual.get("right_items", [])))
        pad_x, pad_y, gap_x, head_h = 26, 26, 22, 46
        col_w = (sw - 2 * pad_x - gap_x) // 2
        top_y = pad_y + head_h + 18
        avail_h = sh - top_y - pad_y
        row_gap = 12
        row_h = min(74, (avail_h - (rows - 1) * row_gap) // max(1, rows))
        for r in range(rows):
            ry1 = top_y + r * (row_h + row_gap)
            boxes.append((pad_x, ry1, pad_x + col_w, ry1 + row_h))
            boxes.append((pad_x + col_w + gap_x, ry1, pad_x + 2 * col_w + gap_x, ry1 + row_h))
    elif vtype == "bar_chart":
        bars = visual.get("bars", [])
        n = len(bars)
        if n:
            pad_x, base_y, top_y = 42, sh - 86, 66
            slot_w = (sw - 2 * pad_x) / n
            bar_w = min(74, int(slot_w * 0.56))
            for i in range(n):
                cx = pad_x + slot_w * (i + 0.5)
                boxes.append((cx - bar_w / 2, top_y, cx + bar_w / 2, base_y))
    return boxes


def _render_diagram_box(
    sw: int,
    sh: int,
    visual: dict,
    step_prog: list,
    active_idx: int,
    ss: int = 2,
) -> Image.Image:
    """Render the right-side diagram panel with supersampling for smooth lines."""
    hi = Image.new("RGB", (sw * ss, sh * ss), STAGE_BG)
    d = ImageDraw.Draw(hi)
    d.rounded_rectangle(
        [ss, ss, sw * ss - ss * 2, sh * ss - ss * 2],
        radius=12 * ss,
        fill=STAGE_BG,
        outline=STAGE_BORDER,
        width=2 * ss,
    )

    vtype = (visual or {}).get("type", "none")
    if vtype == "flow":
        _render_flow_ss(d, sw * ss, sh * ss, visual, step_prog, active_idx, ss)
    elif vtype == "compare":
        _render_compare_ss(d, sw * ss, sh * ss, visual, step_prog, active_idx, ss)
    elif vtype == "bar_chart":
        _render_bar_chart_ss(d, sw * ss, sh * ss, visual, step_prog, active_idx, ss)
    elif vtype == "cycle":
        _render_cycle_ss(d, sw * ss, sh * ss, visual, step_prog, active_idx, ss)
    elif vtype == "figure":
        _render_figure_ss(hi, d, sw * ss, sh * ss, visual, step_prog, active_idx, ss)

    if ss == 1:
        return hi
    return hi.resize((sw, sh), resample=Image.Resampling.BILINEAR)


def _prepare_scene_layout(scene: dict, index: int, total: int, paper_title: str) -> dict:
    """Pre-compute text layout and static base image once per scene."""
    base = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(base)
    d.rectangle([0, 0, 14, H], fill=MARK)

    margin = 80
    max_w = W - margin * 2

    # Footer static text + progress track
    small = font(22)
    d.text((margin, H - 62), f"{index + 1} / {total}", font=small, fill=MUTED)
    label = shorten(d, paper_title, small, 700)
    d.text((W - margin - d.textlength(label, font=small), H - 62), label, font=small, fill=MUTED)
    d.rectangle([0, H - 10, W, H], fill=TRACK)

    # Title sizing (fits in max 2 lines)
    tsize = 52
    while True:
        title_font = font(tsize, bold=True)
        title_lines = wrap(d, scene["title"], title_font, max_w)
        if len(title_lines) <= 2 or tsize <= 36:
            break
        tsize -= 4

    title_y = 56
    after_title_y = title_y + len(title_lines) * int(tsize * 1.18) + 18
    content_top = max(190, after_title_y + 34)
    bottom = H - 100

    visual = scene.get("visual") or {"type": "none"}
    vtype = visual.get("type", "none")
    has_diagram = vtype in ("flow", "compare", "bar_chart", "cycle", "figure")

    bullet_max_w = 435 if has_diagram else max_w
    bsize = 29 if has_diagram else 36
    min_bsize = 20 if has_diagram else 24

    while True:
        bullet_font = font(bsize)
        line_h = int(bsize * 1.32)
        gap = int(bsize * 0.55)
        bullet_layout, yy = [], content_top
        for bullet in scene["bullets"]:
            bl = wrap(d, bullet, bullet_font, bullet_max_w - 40)
            bullet_layout.append((yy, bl))
            yy += len(bl) * line_h + gap
        if yy - gap <= bottom or bsize <= min_bsize:
            break
        bsize -= 2

    return {
        "base": base,
        "margin": margin,
        "title_font": title_font,
        "title_size": tsize,
        "title_lines": title_lines,
        "title_y": title_y,
        "underline_y": after_title_y,
        "bullet_font": bullet_font,
        "bullet_size": bsize,
        "bullet_line_h": line_h,
        "bullet_layout": bullet_layout,
        "has_diagram": has_diagram,
        "visual": visual,
        "vtype": vtype,
        "scene": scene,
        "stage_box": (560, 185, 640, 445),
    }


def _render_intro_frame(
    layout: dict, t: float, duration: float, index: int, total: int
) -> Image.Image:
    img = layout["base"].copy()
    d = ImageDraw.Draw(img)
    scene = layout["scene"]
    visual = layout["visual"]
    summary = visual.get("summary") or (scene["bullets"][0] if scene.get("bullets") else "")

    p1 = ease_out(t / 0.55)
    p2 = ease_out((t - 0.35) / 0.55)

    # Top pill badge
    badge_font = font(18, bold=True)
    badge_text = "PAPER EXPLAINER"
    bw = d.textlength(badge_text, font=badge_font) + 32
    bx1 = (W - bw) / 2
    by1 = 135 - int(18 * (1.0 - p1))
    d.rounded_rectangle(
        [bx1, by1, bx1 + bw, by1 + 38],
        radius=19,
        fill=lerp_color(BG, COBALT, p1),
        outline=lerp_color(BG, MARK, p1),
        width=2,
    )
    d.text((bx1 + 16, by1 + 7), badge_text, font=badge_font, fill=lerp_color(BG, INK, p1))

    # Centered animated paper title
    tsize = 58
    while True:
        tfont = font(tsize, bold=True)
        tlines = wrap(d, scene["title"], tfont, 980)
        if len(tlines) <= 2 or tsize <= 42:
            break
        tsize -= 4

    ty = 215 + int(20 * (1.0 - p1))
    tcol = lerp_color(BG, INK, p1)
    for line in tlines:
        lw = d.textlength(line, font=tfont)
        d.text(((W - lw) / 2, ty), line, font=tfont, fill=tcol)
        ty += int(tsize * 1.2)

    # Expanding center yellow accent bar
    bar_w = int(180 * p1)
    if bar_w > 0:
        d.rectangle([(W - bar_w) // 2, ty + 18, (W + bar_w) // 2, ty + 25], fill=MARK)

    # Summary card fading in below
    if summary and p2 > 0.02:
        sfont = font(26)
        slines = wrap(d, summary, sfont, 880)[:3]
        card_h = len(slines) * 38 + 44
        cy1 = ty + 52 + int(16 * (1.0 - p2))
        cx1, cx2 = (W - 940) // 2, (W + 940) // 2
        d.rounded_rectangle(
            [cx1, cy1, cx2, cy1 + card_h],
            radius=12,
            fill=lerp_color(BG, STAGE_BG, p2),
            outline=lerp_color(BG, STAGE_BORDER, p2),
            width=2,
        )
        sy = cy1 + 22
        scol = lerp_color(BG, MUTED, p2)
        for sline in slines:
            sw_len = d.textlength(sline, font=sfont)
            d.text(((W - sw_len) / 2, sy), sline, font=sfont, fill=scol)
            sy += 38

    scene_frac = clamp(t / max(0.1, duration), 0.0, 1.0)
    overall_frac = (index + scene_frac) / max(1, total)
    d.rectangle([0, H - 10, int(W * overall_frac), H], fill=MARK)
    return img


def _render_outro_frame(
    layout: dict, t: float, duration: float, index: int, total: int
) -> Image.Image:
    img = layout["base"].copy()
    d = ImageDraw.Draw(img)
    margin = layout["margin"]
    visual = layout["visual"]
    takeaways = visual.get("takeaways") or layout["scene"].get("bullets", [])[:3]

    # Header
    tp = ease_out(t / 0.45)
    tfont = font(48, bold=True)
    d.text((margin - int(32 * (1.0 - tp)), 58), "3 Key Takeaways", font=tfont, fill=lerp_color(BG, INK, tp))
    d.rectangle([margin, 126, margin + int(110 * tp), 132], fill=MARK)

    speech_dur = max(1.0, duration - PAD)
    n = len(takeaways)
    span = max(0.6, speech_dur - 0.55)
    active_idx = -1
    progs = []
    for i in range(n):
        t0 = 0.35 + span * (i / max(1, n))
        p = clamp((t - t0) / 0.42, 0.0, 1.0)
        progs.append(p)
        if p > 0.05 and t <= speech_dur + 0.15:
            active_idx = i

    card_x1, card_x2 = margin, W - margin
    card_h = 122
    gap_y = 22
    top_start = 172
    kfont = font(28, bold=False)
    num_font = font(30, bold=True)

    for i, item in enumerate(takeaways[:3]):
        p = ease_out(progs[i])
        if p <= 0.02:
            continue
        is_active = i == active_idx
        dx = int(42 * (1.0 - p))
        y1 = top_start + i * (card_h + gap_y)
        y2 = y1 + card_h

        bg_c = lerp_color(BG, NODE_ACTIVE_BG if is_active else STAGE_BG, p)
        bd_c = lerp_color(BG, MARK if is_active else COBALT, p)
        tx_c = lerp_color(BG, MARK if is_active else INK, p)

        d.rounded_rectangle(
            [card_x1 + dx, y1, card_x2 + dx, y2],
            radius=12,
            fill=bg_c,
            outline=bd_c,
            width=3 if is_active else 2,
        )

        # Number badge
        bx = card_x1 + dx + 48
        by = (y1 + y2) // 2
        r = 26
        d.ellipse([bx - r, by - r, bx + r, by + r], fill=bd_c)
        ns = str(i + 1)
        nw = d.textlength(ns, font=num_font)
        d.text((bx - nw / 2, by - 18), ns, font=num_font, fill=BG if is_active else INK)

        lines = wrap(d, item, kfont, (card_x2 - card_x1) - 130)[:2]
        line_h = 36
        ty = by - (len(lines) * line_h) // 2 - 2
        for line in lines:
            d.text((card_x1 + dx + 98, ty), line, font=kfont, fill=tx_c)
            ty += line_h

    scene_frac = clamp(t / max(0.1, duration), 0.0, 1.0)
    overall_frac = (index + scene_frac) / max(1, total)
    d.rectangle([0, H - 10, int(W * overall_frac), H], fill=MARK)
    return img


def render_scene_frame(
    layout: dict,
    t: float,
    duration: float,
    index: int,
    total: int,
    diagram_cache: dict,
) -> Image.Image:
    """Render one 1280x720 RGB frame at timestamp `t` seconds."""
    vtype = layout.get("vtype", "none")
    if vtype == "intro":
        return _render_intro_frame(layout, t, duration, index, total)
    if vtype == "outro":
        return _render_outro_frame(layout, t, duration, index, total)

    img = layout["base"].copy()
    d = ImageDraw.Draw(img)
    margin = layout["margin"]

    # 1. Animated Title (slides in over first 0.45s)
    tp = ease_out(t / 0.45)
    title_dx = -int(36 * (1.0 - tp))
    title_col = lerp_color(BG, INK, tp)
    ty = layout["title_y"]
    t_step = int(layout["title_size"] * 1.18)
    for line in layout["title_lines"]:
        d.text((margin + title_dx, ty), line, font=layout["title_font"], fill=title_col)
        ty += t_step

    bar_w = int(90 * tp)
    if bar_w > 0:
        d.rectangle(
            [margin, layout["underline_y"], margin + bar_w, layout["underline_y"] + 6],
            fill=MARK,
        )

    # Timing window synced to narration length
    speech_dur = max(1.0, duration - PAD)
    bullet_layout = layout["bullet_layout"]
    nb = len(bullet_layout)

    # 2. Animated Bullets (fade + slide in one by one; active bullet highlighted in yellow)
    b_start_base = 0.35
    b_span = max(0.6, speech_dur - 0.55)
    active_bullet = -1
    b_progs = []
    for i in range(nb):
        t0 = b_start_base + b_span * (i / max(1, nb))
        p = clamp((t - t0) / 0.38, 0.0, 1.0)
        b_progs.append(p)
        if p > 0.05 and t <= speech_dur + 0.15:
            active_bullet = i

    bsize = layout["bullet_size"]
    bfont = layout["bullet_font"]
    line_h = layout["bullet_line_h"]
    for i, (top, bl) in enumerate(bullet_layout):
        p = ease_out(b_progs[i])
        if p <= 0.02:
            continue
        is_active = i == active_bullet
        dy = int(14 * (1.0 - p))
        r = max(5, bsize // 6) + (2 if is_active else 0)
        cy = top + dy + int(bsize * 0.68)
        dot_col = lerp_color(BG, MARK, p)
        txt_col = lerp_color(BG, MARK if is_active else INK, p)
        d.ellipse([margin, cy - r, margin + 2 * r, cy + r], fill=dot_col)
        for j, text in enumerate(bl):
            d.text((margin + 38, top + dy + j * line_h), text, font=bfont, fill=txt_col)

    # 3. Animated Diagram Stage (if scene has a visual diagram)
    if layout["has_diagram"]:
        visual = layout["visual"]
        nv = _visual_step_count(visual)
        v_start_base = 0.40
        v_span = max(0.6, speech_dur - 0.60)
        active_vis = -1
        v_progs = []
        for i in range(nv):
            t0 = v_start_base + v_span * (i / max(1, nv))
            p = clamp((t - t0) / 0.48, 0.0, 1.0)
            v_progs.append(p)
            if p > 0.05 and t <= speech_dur + 0.15:
                active_vis = i

        # Quantize progress to 50 steps so settled frames hit cache in 0ms
        ss = layout.get("ss", 2)
        cache_key = (active_vis, ss, tuple(int(round(p * 50)) for p in v_progs))
        if cache_key not in diagram_cache:
            sx, sy, sw, sh = layout["stage_box"]
            diagram_cache[cache_key] = _render_diagram_box(sw, sh, visual, v_progs, active_vis, ss=ss)
        sx, sy, _, _ = layout["stage_box"]
        img.paste(diagram_cache[cache_key], (sx, sy))

    # 4. Smooth footer progress bar across the whole video
    scene_frac = clamp(t / max(0.1, duration), 0.0, 1.0)
    overall_frac = (index + scene_frac) / max(1, total)
    d.rectangle([0, H - 10, int(W * overall_frac), H], fill=MARK)

    return img


def render_scene_frames(
    scene: dict,
    index: int,
    total: int,
    paper_title: str,
    duration: float,
    PreviewPath: Path | None = None,
    quality: str | None = None,
) -> Iterator[bytes]:
    """Yield raw RGB24 bytes for each frame of `scene` at the configured QUALITY."""
    out_w, out_h, fps = quality_spec(quality)
    layout = _prepare_scene_layout(scene, index, total, paper_title)
    layout["ss"] = 1 if out_w < W else 2
    total_frames = max(1, int(round(duration * fps)))
    diagram_cache = {}
    preview_saved = False
    speech_dur = max(1.0, duration - PAD)

    last_state_key = None
    last_bytes = None

    for f_idx in range(total_frames):
        t = f_idx / fps
        # Quantize state key to avoid re-rendering identical settled frames
        if t > speech_dur + 0.25:
            bar_px = int(W * ((index + clamp(t / max(0.1, duration), 0.0, 1.0)) / max(1, total)))
            state_key = ("settled", bar_px // 4)
        else:
            state_key = ("anim", f_idx)

        if state_key == last_state_key and last_bytes is not None:
            yield last_bytes
            continue

        frame = render_scene_frame(layout, t, duration, index, total, diagram_cache)
        if PreviewPath is not None and not preview_saved and t >= speech_dur:
            PreviewPath.parent.mkdir(parents=True, exist_ok=True)
            frame.save(PreviewPath, "PNG")
            preview_saved = True

        if (out_w, out_h) != (W, H):
            frame = frame.resize((out_w, out_h), resample=Image.Resampling.BILINEAR)

        raw = frame.tobytes()
        last_state_key = state_key
        last_bytes = raw
        yield raw

    if PreviewPath is not None and not preview_saved:
        PreviewPath.parent.mkdir(parents=True, exist_ok=True)
        frame = render_scene_frame(layout, speech_dur, duration, index, total, diagram_cache)
        frame.save(PreviewPath, "PNG")

