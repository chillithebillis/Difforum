"""
Curve rasterizer for Difforum - render a schedule's per-frame values as an image.

Server-side (numpy) so it shows the *actual* computed curve, expressions and
audio reactivity included - more faithful than a client-side JS preview that
would have to re-implement the expression engine. The node wraps the output as
a ComfyUI IMAGE so it plugs into Preview Image / Save Image.
"""

from __future__ import annotations

import numpy as np

_BG = (0.10, 0.11, 0.13)
_GRID = (0.20, 0.22, 0.26)
_AXIS = (0.35, 0.38, 0.42)
_LINE = (0.30, 0.80, 1.00)


def _vline(c, x, color):
    c[:, x] = color


def _hline(c, y, color):
    c[y, :] = color


def render_curve(
    values, width: int = 512, height: int = 256, thickness: int = 2
) -> np.ndarray:
    """Rasterize `values` (1D sequence) into an [H,W,3] float image in 0..1."""
    vals = np.asarray(list(values), dtype=np.float64)
    h, w = int(height), int(width)
    canvas = np.empty((h, w, 3), dtype=np.float32)
    canvas[:] = _BG
    if vals.size == 0:
        return canvas

    pad = 6
    pw, ph = max(2, w - 2 * pad), max(2, h - 2 * pad)

    lo, hi = float(vals.min()), float(vals.max())
    span = (hi - lo) or 1.0

    # grid (quarters) + border
    for q in (0.25, 0.5, 0.75):
        _hline(canvas, pad + int((1 - q) * (ph - 1)), _GRID)
    for q in (0.25, 0.5, 0.75):
        _vline(canvas, pad + int(q * (pw - 1)), _GRID)
    canvas[pad, pad:pad + pw] = _AXIS
    canvas[pad + ph - 1, pad:pad + pw] = _AXIS
    canvas[pad:pad + ph, pad] = _AXIS
    canvas[pad:pad + ph, pad + pw - 1] = _AXIS

    n = vals.size

    def y_of(v):
        t = (v - lo) / span
        return pad + int(round((1.0 - t) * (ph - 1)))

    prev_y = None
    for px in range(pw):
        u = px / (pw - 1) if pw > 1 else 0.0
        fpos = u * (n - 1)
        i0 = int(np.floor(fpos))
        i1 = min(i0 + 1, n - 1)
        frac = fpos - i0
        v = vals[i0] * (1 - frac) + vals[i1] * frac
        y = y_of(v)
        x = pad + px
        ys = [y] if prev_y is None else list(range(min(prev_y, y), max(prev_y, y) + 1))
        for yy in ys:
            for t in range(-(thickness // 2), thickness // 2 + 1):
                yt = min(max(yy + t, 0), h - 1)
                canvas[yt, x] = _LINE
        prev_y = y

    return canvas


_START = (0.30, 0.95, 0.45)
_END = (1.00, 0.55, 0.20)


def _dot(c, y, x, color, r=3):
    h, w = c.shape[:2]
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                yy, xx = min(max(y + dy, 0), h - 1), min(max(x + dx, 0), w - 1)
                c[yy, xx] = color


def render_camera_path(poses, zoom, mode: str = "3d",
                       width: int = 512, height: int = 512) -> np.ndarray:
    """Top-down view of the camera trajectory (see the direction before you
    render). Path fades dim -> bright over time; green dot = start, orange =
    end. A strip at the bottom shows the zoom curve. [H,W,3] float 0..1."""
    h, w = int(height), int(width)
    canvas = np.empty((h, w, 3), dtype=np.float32)
    canvas[:] = _BG

    pts = np.asarray([[p[0, 3], p[1, 3] if mode == "2d" else p[2, 3]] for p in poses],
                     dtype=np.float64)
    n = len(pts)
    if n == 0:
        return canvas

    strip = max(24, h // 6)                       # zoom strip at the bottom
    pad = 14
    ph, pw = h - strip - 2 * pad, w - 2 * pad

    span = np.maximum(pts.max(axis=0) - pts.min(axis=0), 1e-6)
    scale = min(pw / span[0], ph / span[1]) * 0.9
    center = (pts.min(axis=0) + pts.max(axis=0)) / 2.0

    def to_px(p):
        x = pad + pw / 2 + (p[0] - center[0]) * scale
        y = pad + ph / 2 + (p[1] - center[1]) * scale
        return int(round(y)), int(round(x))

    # grid cross through the origin of the plot area
    _hline(canvas[: h - strip], pad + ph // 2, _GRID)
    _vline(canvas[: h - strip], pad + pw // 2, _GRID)

    for i in range(1, n):
        y0, x0 = to_px(pts[i - 1])
        y1, x1 = to_px(pts[i])
        steps = max(abs(y1 - y0), abs(x1 - x0), 1)
        bright = 0.35 + 0.65 * (i / max(1, n - 1))
        color = (0.30 * bright, 0.80 * bright, 1.00 * bright)
        for s in range(steps + 1):
            t = s / steps
            yy = min(max(int(round(y0 + (y1 - y0) * t)), 0), h - strip - 1)
            xx = min(max(int(round(x0 + (x1 - x0) * t)), 0), w - 1)
            canvas[yy, xx] = color
            if xx + 1 < w:
                canvas[yy, xx + 1] = color
    _dot(canvas, *to_px(pts[0]), _START)
    _dot(canvas, *to_px(pts[-1]), _END)

    # zoom strip
    zvals = np.asarray(list(zoom), dtype=np.float64)
    if zvals.size:
        lo, hi = float(zvals.min()), float(zvals.max())
        zspan = (hi - lo) or 1.0
        base = h - strip
        canvas[base] = _AXIS
        for px in range(w):
            i = int(px / max(1, w - 1) * (zvals.size - 1))
            t = (zvals[i] - lo) / zspan
            y = base + 2 + int((1.0 - t) * (strip - 5))
            canvas[min(y, h - 1), px] = _LINE
    return canvas
