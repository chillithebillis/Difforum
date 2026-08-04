"""
Seamless loops without a crossfade.

A blended tail is a patch: it hides the seam by dissolving two frames that were
never going to meet. For a projection that runs for hours the eye finds that
dissolve every cycle. The fix is to make the clip genuinely periodic instead.

Two things have to close:

1. THE CAMERA. The accumulated transform over the clip must return to identity,
   and the *velocity* has to match across the wrap or the motion pops. Both fall
   out of one operation: take each per-frame axis curve, drop the DC bin (that
   is the net drift) and the high harmonics (those are what encode the jump at
   the wrap), then transform back. What is left is periodic by construction,
   smooth at the seam, and still carries the low-frequency shape of the move
   that was directed. Zoom is multiplicative, so it gets the same treatment in
   log space - mean of the log going to zero means the product of the zooms is
   exactly 1.

2. THE FEEDBACK STATE. Even on a closed path the *image* has drifted, because
   each frame is re-diffused from the last. Given a periodic camera the loop
   settles onto a periodic attractor after a lap or two, so the answer is to
   render several laps and keep only the final one. No blending anywhere.

Uses numpy's FFT, in keeping with the audio engine - no new dependencies.
"""

from __future__ import annotations

import math

import numpy as np

# Axes that accumulate additively (a per-frame delta).
ADDITIVE_AXES = (
    "translation_x", "translation_y", "translation_z",
    "rotation_3d_x", "rotation_3d_y", "rotation_3d_z",
)
# Axes that accumulate multiplicatively (a per-frame scale factor).
MULTIPLICATIVE_AXES = ("zoom",)

LOOP_MODES = ("harmonic", "zero_mean", "off")


def harmonic_close(values, keep_harmonics: int = 4, drop_dc: bool = True):
    """Project a curve onto the first `keep_harmonics` periodic components.

    The result is exactly periodic over its own length: value[-1] flows into
    value[0] with matching slope. With `drop_dc` the mean is removed too, so the
    curve integrates to zero over the clip and the camera returns to its origin.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = arr.size
    if n < 4:
        return arr.tolist()

    spec = np.fft.rfft(arr)
    keep = max(1, int(keep_harmonics))

    if drop_dc:
        spec[0] = 0.0
    if keep + 1 < spec.size:
        spec[keep + 1:] = 0.0

    return np.fft.irfft(spec, n=n).tolist()


def zero_mean_close(values, drop_dc: bool = True):
    """Cheapest close: just remove the net drift, keep every wiggle.

    The path returns to its origin but the velocity across the wrap is whatever
    the original shot list had, so a fast move can still pop. Use it when the
    shot list is already smooth and the harmonic projection softens it too much.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0 or not drop_dc:
        return arr.tolist()
    return (arr - arr.mean()).tolist()


def close_axis_values(
    values: dict[str, list[float]],
    mode: str = "harmonic",
    keep_harmonics: int = 4,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Make every camera axis periodic over the clip.

    Returns (closed values, residual report). The residual is the leftover net
    motion per axis - it should be ~0 for the additive axes and ~1 for zoom.
    """
    if mode not in LOOP_MODES:
        raise ValueError(f"unknown loop mode {mode!r}, pick from {LOOP_MODES}")
    if mode == "off":
        return values, _residuals(values)

    out: dict[str, list[float]] = {}
    for axis, series in values.items():
        if not series:
            out[axis] = series
            continue

        if axis in MULTIPLICATIVE_AXES:
            # work in log space so the product (not the sum) closes to 1
            logs = np.log(np.clip(np.asarray(series, dtype=np.float64), 1e-6, None))
            closed = (
                harmonic_close(logs, keep_harmonics)
                if mode == "harmonic"
                else zero_mean_close(logs)
            )
            out[axis] = np.exp(np.asarray(closed)).tolist()
        elif axis in ADDITIVE_AXES:
            out[axis] = (
                harmonic_close(series, keep_harmonics)
                if mode == "harmonic"
                else zero_mean_close(series)
            )
        else:
            out[axis] = series

    return out, _residuals(out)


def _residuals(values: dict[str, list[float]]) -> dict[str, float]:
    res: dict[str, float] = {}
    for axis, series in values.items():
        if not series:
            continue
        arr = np.asarray(series, dtype=np.float64)
        if axis in MULTIPLICATIVE_AXES:
            res[axis] = float(np.exp(np.log(np.clip(arr, 1e-6, None)).sum()))
        elif axis in ADDITIVE_AXES:
            res[axis] = float(arr.sum())
    return res


def tile_laps(values: dict[str, list[float]], laps: int) -> dict[str, list[float]]:
    """Repeat every axis `laps` times, back to back.

    Because each lap is already periodic the joins are silent; the extra laps
    exist purely to let the feedback image settle onto its cycle.
    """
    laps = max(1, int(laps))
    if laps == 1:
        return values
    return {axis: list(series) * laps for axis, series in values.items()}


def seam_report(frames, clip_len: int) -> str:
    """Compare the final lap's first and last frame - the loop's actual error.

    Also compares against the *previous* lap so you can see whether the feedback
    has settled: if lap-to-lap drift is far below the seam error, more laps will
    not help and the residual is coming from the camera instead.
    """
    import torch

    n = len(frames)
    if n < 2:
        return "seam: not enough frames"

    def diff(a, b) -> float:
        return float((a.float() - b.float()).abs().mean())

    lines = []
    head, tail = frames[0], frames[-1]
    seam = diff(tail, head)
    lines.append(f"  seam (last -> first)   {seam:.5f}")

    if clip_len > 0 and n >= clip_len * 2:
        lap_drift = diff(frames[-1], frames[-1 - clip_len])
        lines.append(f"  lap-to-lap drift       {lap_drift:.5f}")
        if lap_drift < seam * 0.5:
            lines.append("  -> settled; residual is the camera path, not the feedback")
        else:
            lines.append("  -> not settled yet; try one more lap")

    scale = "excellent" if seam < 0.02 else "good" if seam < 0.05 else \
            "visible" if seam < 0.10 else "poor"
    lines.append(f"  verdict: {scale}")
    return "\n".join(lines)


def cycles_for(max_frames: int, seconds_per_cycle: float, fps: float) -> int:
    """How many whole camera cycles fit the clip - handy when authoring moves.

    Returns at least 1; a periodic move needs a whole number of cycles or the
    wrap will not line up.
    """
    if seconds_per_cycle <= 0 or fps <= 0:
        return 1
    return max(1, int(round(max_frames / (seconds_per_cycle * fps))))


def suggest_harmonics(max_frames: int, fps: float, min_cycle_seconds: float = 2.0) -> int:
    """Highest harmonic whose period is still longer than `min_cycle_seconds`.

    Keeping the projection below that bound is what stops the closed path from
    turning into a jitter: anything faster is exactly the content that was
    encoding the seam.
    """
    if fps <= 0 or min_cycle_seconds <= 0:
        return 4
    return max(1, min(16, int(max_frames / (min_cycle_seconds * fps))))
