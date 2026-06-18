"""
Difforum schedule nodes (Phase 1) - GPU-free.

These expose the core schedule engine to ComfyUI. They deal only in plain
Python/float data so they load and run without torch being involved, which
keeps the animation "brain" testable and fast.

Custom socket types:
    DIFFORUM_PARAMS    -> dict of global animation params
    DIFFORUM_SCHEDULE  -> core.schedule.Schedule (a dense per-frame curve)
"""

from __future__ import annotations

import sys
from pathlib import Path

# allow `from core import ...` whether loaded as a package or standalone
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core import EASINGS, build_schedule  # noqa: E402

CATEGORY = "Difforum/schedule"


class DifforumAnimSetup:
    """Global animation parameters shared by every Difforum node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "max_frames": ("INT", {"default": 120, "min": 1, "max": 100000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("DIFFORUM_PARAMS",)
    RETURN_NAMES = ("params",)
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, width, height, fps, max_frames, seed):
        params = {
            "width": int(width),
            "height": int(height),
            "fps": float(fps),
            "max_frames": int(max_frames),
            "seed": int(seed),
        }
        return (params,)


class DifforumSchedule:
    """Parse a Deforum-style keyframe string into a per-frame curve."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "schedule": ("STRING", {
                    "multiline": True,
                    "default": "0:(0), 60:(0.5*sin(2*pi*t/30)), 120:(1.0)",
                }),
                "easing": (list(EASINGS), {"default": "linear"}),
            },
            "optional": {
                "audio": ("DIFFORUM_AUDIO",),
            },
        }

    RETURN_TYPES = ("DIFFORUM_SCHEDULE", "FLOAT")
    RETURN_NAMES = ("schedule", "values")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, schedule, easing, audio=None):
        extra = audio.get("curves") if isinstance(audio, dict) else None
        sched = build_schedule(
            schedule,
            max_frames=params["max_frames"],
            fps=params["fps"],
            easing=easing,
            extra_vars=extra,
        )
        return (sched, sched.as_list())


class DifforumSampleSchedule:
    """Read the value of a schedule at a single frame index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("DIFFORUM_SCHEDULE",),
                "frame": ("INT", {"default": 0, "min": 0, "max": 100000}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("value",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, schedule, frame):
        return (float(schedule.at(int(frame))),)


class DifforumScheduleInfo:
    """Human-readable summary + ASCII sparkline of a schedule (debug aid)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"schedule": ("DIFFORUM_SCHEDULE",)}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    _BARS = " .:-=+*#%@"

    def run(self, schedule):
        vals = schedule.as_list()
        if not vals:
            return ("(empty schedule)",)
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        step = max(1, len(vals) // 60)
        spark = "".join(
            self._BARS[min(len(self._BARS) - 1, int((v - lo) / span * (len(self._BARS) - 1)))]
            for v in vals[::step]
        )
        txt = (
            f"frames={len(vals)} fps={schedule.fps} easing={schedule.easing}\n"
            f"min={lo:.4g} max={hi:.4g} first={vals[0]:.4g} last={vals[-1]:.4g}\n"
            f"{spark}\n"
            f"src: {schedule.source}"
        )
        return {"ui": {"text": [txt]}, "result": (txt,)}


class DifforumSchedulePlot:
    """Render a schedule's per-frame curve as an IMAGE (plug into Preview Image)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "schedule": ("DIFFORUM_SCHEDULE",),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 256, "min": 64, "max": 4096}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("plot",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, schedule, width, height):
        import torch

        from core.plot import render_curve

        arr = render_curve(schedule.as_list(), width=int(width), height=int(height))
        return (torch.from_numpy(arr).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "DifforumAnimSetup": DifforumAnimSetup,
    "DifforumSchedule": DifforumSchedule,
    "DifforumSampleSchedule": DifforumSampleSchedule,
    "DifforumScheduleInfo": DifforumScheduleInfo,
    "DifforumSchedulePlot": DifforumSchedulePlot,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumAnimSetup": "Difforum · Anim Setup",
    "DifforumSchedule": "Difforum · Schedule",
    "DifforumSampleSchedule": "Difforum · Sample Schedule",
    "DifforumScheduleInfo": "Difforum · Schedule Info",
    "DifforumSchedulePlot": "Difforum · Schedule Plot",
}
