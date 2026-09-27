"""
Difforum loop nodes - seamless cycles without a crossfade.

Difforum · Seamless Camera   builds a camera whose path is periodic, so the
                             clip's last frame flows into its first.
Difforum · Loop Take         keeps the settled lap and scores the seam.
"""

from __future__ import annotations



from ..core.camera import CAMERA_MODES, build_camera  # noqa: E402
from ..core.loop import (  # noqa: E402
    LOOP_MODES,
    close_axis_values,
    seam_report,
    suggest_harmonics,
    tile_laps,
)
from ..core.shots import shots_to_axis_values  # noqa: E402

CATEGORY = "Difforum/loop"


class DifforumSeamlessCamera:
    """Camera Shots, but the path closes on itself.

    Drop-in replacement for `Camera Shots (director)` when the clip has to loop.
    The shot list is written exactly the same way; what changes is that the
    per-frame motion is projected onto components that are periodic over
    `cycle_frames`, so the accumulated transform returns to identity and the
    velocity matches across the wrap.

    Set the Anim Setup's `max_frames` to a whole multiple of `cycle_frames`:
    the extra laps let the feedback image settle onto its cycle, and
    `Loop Take` keeps the last one.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "shots": ("STRING", {
                    "multiline": True,
                    "default": "0: orbit_right 1.0 0.8\n60: spiral 1.2 1.0",
                }),
                "cycle_frames": ("INT", {"default": 120, "min": 8, "max": 100000}),
                "loop_mode": (list(LOOP_MODES), {"default": "harmonic"}),
                "harmonics": ("INT", {"default": 3, "min": 1, "max": 16}),
                "mode": (list(CAMERA_MODES), {"default": "2d"}),
                "fov": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 170.0, "step": 1.0}),
            },
            "optional": {"audio": ("DIFFORUM_AUDIO",)},
        }

    RETURN_TYPES = ("DIFFORUM_CAMERA", "INT", "STRING")
    RETURN_NAMES = ("camera", "cycle_frames", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, shots, cycle_frames, loop_mode, harmonics, mode, fov,
            audio=None):
        extra = audio.get("curves") if isinstance(audio, dict) else None
        total = int(params["max_frames"])
        fps = float(params["fps"])
        cyc = max(8, min(int(cycle_frames), total))

        # One cycle's worth of motion, then close it.
        values, summary = shots_to_axis_values(
            shots, max_frames=cyc, fps=fps, extra_vars=extra
        )
        closed, residual = close_axis_values(
            values, mode=loop_mode, keep_harmonics=int(harmonics)
        )

        laps = max(1, total // cyc)
        tiled = tile_laps(closed, laps)

        # pad to the requested length if max_frames is not a clean multiple
        for axis, series in tiled.items():
            if len(series) < total:
                series.extend([series[-1]] * (total - len(series)))
            tiled[axis] = series[:total]

        cam = build_camera(tiled, max_frames=total, mode=mode, fov=fov)

        drift = max(
            (abs(v - (1.0 if ax == "zoom" else 0.0)) for ax, v in residual.items()),
            default=0.0,
        )
        rec = suggest_harmonics(cyc, fps, min_cycle_seconds=2.0)
        notes = []
        if total % cyc:
            notes.append(
                f"  ! max_frames {total} is not a multiple of cycle {cyc}; "
                f"the tail was padded - set max_frames to {laps * cyc} or {(laps + 1) * cyc}"
            )
        if laps < 2 and loop_mode != "off":
            notes.append(
                "  ! only 1 lap: the feedback image has no room to settle. "
                f"Set max_frames to {cyc * 3} and use Loop Take."
            )
        if int(harmonics) > rec:
            notes.append(
                f"  ! harmonics {harmonics} puts motion under 2s per cycle "
                f"(<= {rec} keeps it calm at {fps:g}fps)"
            )

        info = "\n".join([
            f"seamless camera ({mode}, {loop_mode}, harmonics={harmonics})",
            f"  cycle {cyc}f ({cyc / fps:.1f}s) x {laps} laps = {total}f "
            f"({total / fps:.1f}s rendered)",
            f"  path closure error: {drift:.2e}  (0 = returns exactly to origin)",
            "",
            summary,
            *notes,
        ])
        return (cam, cyc, info)


class DifforumLoopTake:
    """Keep the settled lap of a multi-lap render, and score the seam.

    The earlier laps exist so the feedback image can converge onto its cycle;
    they are discarded here. Nothing is blended - the frames handed back are
    untouched, so the loop is only as good as the camera and the settling, which
    is exactly what the seam score reports.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "cycle_frames": ("INT", {"default": 120, "min": 2, "max": 100000}),
                "take": (["last", "first"], {"default": "last"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "seam_info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, cycle_frames, take):
        total = int(frames.shape[0])
        cyc = max(2, min(int(cycle_frames), total))

        if take == "last":
            out = frames[total - cyc:]
        else:
            out = frames[:cyc]

        laps = total // cyc
        report = seam_report([out[i] for i in range(out.shape[0])], cyc)
        info = "\n".join([
            f"loop take: {cyc} of {total} frames ({laps} lap(s) rendered, kept the {take})",
            report,
            "",
            "  no crossfade applied - raise laps or lower harmonics if the seam shows",
        ])
        return (out, info)


NODE_CLASS_MAPPINGS = {
    "DifforumSeamlessCamera": DifforumSeamlessCamera,
    "DifforumLoopTake": DifforumLoopTake,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumSeamlessCamera": "Difforum · Seamless Camera (loop)",
    "DifforumLoopTake": "Difforum · Loop Take (settled lap)",
}
