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
