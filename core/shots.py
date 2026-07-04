"""
Camera direction as a shot list. Instead of one preset for the whole clip (or
raw math strings), you direct the camera like an edit:

    0: dolly_in 1.0 1.0
    48: orbit_right 1.2 0.8
    96: spiral

Each line is `frame: preset [speed] [intensity]`. Every segment runs its preset
from its start frame until the next keyframe (the last one runs to the end),
and the per-frame increments are concatenated so motion accumulates seamlessly
across cuts. Output feeds build_camera directly.
"""

from __future__ import annotations

import re

from .camera_presets import CAMERA_PRESETS, _AXES, preset_schedules
from .schedule import build_schedule

_LINE = re.compile(
    r"^\s*(\d+)\s*:\s*([a-zA-Z_]+)"      # frame: preset
    r"(?:\s+(-?\d+(?:\.\d+)?))?"          # optional speed
    r"(?:\s+(-?\d+(?:\.\d+)?))?\s*$"      # optional intensity
)


def parse_shots(text: str) -> list[tuple[int, str, float, float]]:
    """Parse the shot list into sorted (frame, preset, speed, intensity)."""
    shots = []
    for raw in str(text).splitlines():
        line = raw.split("#", 1)[0].strip()   # allow comments
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            raise ValueError(f"bad shot line: {raw!r} (want 'frame: preset [speed] [intensity]')")
        frame, preset = int(m.group(1)), m.group(2).lower()
        if preset not in CAMERA_PRESETS:
            raise ValueError(f"unknown camera preset {preset!r} in {raw!r}; pick from {CAMERA_PRESETS}")
        speed = float(m.group(3)) if m.group(3) else 1.0
        intensity = float(m.group(4)) if m.group(4) else 1.0
        shots.append((frame, preset, speed, intensity))
    if not shots:
        raise ValueError("empty shot list")
    shots.sort(key=lambda s: s[0])
    if shots[0][0] != 0:
        shots.insert(0, (0, "still", 1.0, 1.0))
    return shots


def shots_to_axis_values(text: str, max_frames: int, fps: int = 24,
                         extra_vars=None) -> tuple[dict[str, list[float]], str]:
    """Expand the shot list into per-frame value lists for each camera axis.

    Each segment's preset expressions are evaluated over the segment length with
    a local clock (t restarts per shot), then concatenated. Returns
    (axis -> values, human summary)."""
    shots = parse_shots(text)
    n = int(max_frames)
    values: dict[str, list[float]] = {ax: [] for ax in _AXES}
    lines = []
    for i, (start, preset, speed, intensity) in enumerate(shots):
        if start >= n:
            break
        end = shots[i + 1][0] if i + 1 < len(shots) else n
        seg = max(1, min(end, n) - start)
        exprs = preset_schedules(preset, speed=speed, intensity=intensity)
        for ax in _AXES:
            sched = build_schedule(exprs[ax], max_frames=seg, fps=fps, extra_vars=extra_vars)
            values[ax].extend(sched.as_list())
        lines.append(f"{start}-{min(end, n) - 1}: {preset} (speed {speed:g}, intensity {intensity:g})")
    # pad/trim to exactly n frames (repeat last value if the list ran short)
    for ax in _AXES:
        vals = values[ax]
        if len(vals) < n:
            pad = vals[-1] if vals else (1.0 if ax == "zoom" else 0.0)
            vals.extend([pad] * (n - len(vals)))
        values[ax] = vals[:n]
    return values, "\n".join(lines)
