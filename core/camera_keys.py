"""
Keyframed camera: moves that blend into each other instead of cutting.

The shot list in `shots.py` concatenates segments - each shot runs at a constant
rate and the next one replaces it outright. That is a cut, and it is the right
answer when a cut is what you want. It is the wrong answer for a crane move that
should ease out of a push-in and settle into an orbit: the velocity jumps at the
boundary and the warp shows it as a shear.

Here a keyframe is a *camera state*, not a segment:

    frame   when it applies
    move    which preset shapes the motion
    speed   rate multiplier
    lens    field of view in degrees at this keyframe
    ease    how the motion arrives from the previous keyframe

Between two keyframes both presets are evaluated and cross-faded with the
easing curve, so the camera accelerates out of one move and into the next with
no discontinuity. `ease="step"` restores the hard cut.

Because the lens is a channel of its own, a dolly-zoom is just a keyframe pair
that pushes in while the lens widens - no special preset needed.
"""

from __future__ import annotations

import re

from .camera_presets import _AXES, CAMERA_PRESETS, preset_schedules
from .schedule import _ease, build_schedule

EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out", "step")

# `frame: move [speed] [intensity] [lens] [ease]`
_KEY_RE = re.compile(
    r"^\s*(\d+)\s*:\s*([a-zA-Z_]+)"
    r"(?:\s+(-?\d+(?:\.\d+)?))?"          # speed
    r"(?:\s+(-?\d+(?:\.\d+)?))?"          # intensity
    r"(?:\s+(-?\d+(?:\.\d+)?))?"          # lens (fov degrees)
    r"(?:\s+([a-z_]+))?\s*$"              # ease
)

DEFAULT_LENS = 40.0


def parse_camera_keys(text: str) -> list[dict]:
    """Parse the keyframe list. Tolerates the plain shot-list syntax too."""
    keys: list[dict] = []
    for raw in str(text or "").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        m = _KEY_RE.match(line)
        if not m:
            raise ValueError(
                f"bad camera key: {raw!r} "
                "(want 'frame: move [speed] [intensity] [lens] [ease]')"
            )
        move = m.group(2).lower()
        if move not in CAMERA_PRESETS:
            raise ValueError(
                f"unknown camera move {move!r} in {raw!r}; pick from {CAMERA_PRESETS}"
            )
        ease = (m.group(6) or "ease_in_out").lower()
        if ease not in EASINGS:
            raise ValueError(f"unknown easing {ease!r} in {raw!r}; pick from {EASINGS}")
        keys.append({
            "frame": int(m.group(1)),
            "move": move,
            "speed": float(m.group(3)) if m.group(3) else 1.0,
            "intensity": float(m.group(4)) if m.group(4) else 1.0,
            "lens": float(m.group(5)) if m.group(5) else DEFAULT_LENS,
            "ease": ease,
        })

    if not keys:
        raise ValueError("empty camera keyframe list")

    keys.sort(key=lambda k: k["frame"])
    if keys[0]["frame"] != 0:
        first = dict(keys[0])
        first.update(frame=0, move="still", speed=1.0, intensity=1.0)
        keys.insert(0, first)
    return keys


def _dense(preset: str, speed: float, intensity: float, n: int, fps: float,
           extra_vars=None) -> dict[str, list[float]]:
    """Evaluate one preset over n frames -> per-frame value for every axis."""
    exprs = preset_schedules(preset, speed=speed, intensity=intensity)
    return {
        ax: build_schedule(exprs[ax], max_frames=n, fps=fps,
                           extra_vars=extra_vars).as_list()
        for ax in _AXES
    }


def keys_to_axis_values(
    text: str,
    max_frames: int,
    fps: float = 24.0,
    extra_vars=None,
    blend: float = 1.0,
) -> tuple[dict[str, list[float]], list[float], str]:
    """Expand keyframes into per-frame axis values plus a per-frame lens track.

    `blend` scales how much of each interval is spent transitioning: 1.0 eases
    across the whole span between keyframes, 0.25 holds the move and transitions
    only in the last quarter, 0.0 is a hard cut everywhere.

    Returns (axis values, fov per frame, human summary).
    """
    return keys_list_to_axis_values(parse_camera_keys(text), max_frames, fps,
                                    extra_vars=extra_vars, blend=blend)


def keys_list_to_axis_values(
    keys: list[dict],
    max_frames: int,
    fps: float = 24.0,
    extra_vars=None,
    blend: float = 1.0,
) -> tuple[dict[str, list[float]], list[float], str]:
    """Same as `keys_to_axis_values`, from already-parsed key dicts
    (frame, move, speed, intensity, lens, ease) - the timeline UI path."""
    keys = sorted(keys, key=lambda k: k["frame"])
    if not keys:
        keys = [{"frame": 0, "move": "still", "speed": 1.0, "intensity": 1.0,
                 "lens": DEFAULT_LENS, "ease": "ease_in_out"}]
    if keys[0]["frame"] != 0:
        first = dict(keys[0])
        first.update(frame=0, move="still", speed=1.0, intensity=1.0)
        keys.insert(0, first)
    n = max(1, int(max_frames))

    values: dict[str, list[float]] = {ax: [0.0] * n for ax in _AXES}
    lens: list[float] = [DEFAULT_LENS] * n
    lines: list[str] = []

    # Each preset is evaluated over the whole clip so its internal clock (the
    # `t` in oscillating presets like spiral or sway) stays continuous while it
    # fades in and out - restarting it per segment is what makes blends jitter.
    dense = {}
    for i, k in enumerate(keys):
        sig = (k["move"], k["speed"], k["intensity"])
        if sig not in dense:
            dense[sig] = _dense(k["move"], k["speed"], k["intensity"], n, fps, extra_vars)

    b = max(0.0, min(1.0, float(blend)))

    for i, k in enumerate(keys):
        start = max(0, min(k["frame"], n))
        end = min(keys[i + 1]["frame"], n) if i + 1 < len(keys) else n
        if start >= n:
            break
        end = max(end, start)

        cur = dense[(k["move"], k["speed"], k["intensity"])]
        nxt_k = keys[i + 1] if i + 1 < len(keys) else None
        nxt = dense[(nxt_k["move"], nxt_k["speed"], nxt_k["intensity"])] if nxt_k else None

        span = max(1, end - start)
        # transition occupies the tail `b` of the interval, easing into the next key
        trans_start = end - int(round(span * b)) if nxt is not None else end

        for f in range(start, end):
            if nxt is None or f < trans_start or f >= end:
                w = 0.0
            else:
                u = (f - trans_start) / max(1, end - trans_start)
                w = _ease(u, nxt_k["ease"])

            for ax in _AXES:
                a = cur[ax][f]
                if w > 0.0:
                    a = a * (1.0 - w) + nxt[ax][f] * w
                values[ax][f] = a

            lo = k["lens"]
            hi = nxt_k["lens"] if nxt_k else lo
            lens[f] = lo * (1.0 - w) + hi * w

        lines.append(
            f"{start}-{max(start, end - 1)}: {k['move']} "
            f"(speed {k['speed']:g}, intensity {k['intensity']:g}, "
            f"lens {k['lens']:g}mm-eq fov, {k['ease']})"
        )

    # zoom is multiplicative: a blended 0.0 would freeze the frame
    for f in range(n):
        if values["zoom"][f] == 0.0:
            values["zoom"][f] = 1.0

    return values, lens, "\n".join(lines)


def lens_note(fov: float) -> str:
    """Plain-language read of a field of view, for node info output."""
    if fov <= 18:
        return "ultra tele - compressed, flat, distant"
    if fov <= 30:
        return "telephoto - compressed planes, subject isolation"
    if fov <= 45:
        return "normal - close to how the eye reads depth"
    if fov <= 70:
        return "wide - open space, mild perspective stretch"
    if fov <= 100:
        return "ultra wide - strong perspective, fast apparent motion"
    return "fisheye - extreme curvature, everything rushes past"
