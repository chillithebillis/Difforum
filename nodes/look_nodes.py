"""
Difforum VJ look nodes: a one-shot preset grade (Look), plus granular Colour
Grade and Glow. IMAGE -> IMAGE, work on a single frame or a whole footage batch.
Drop them after a Load Video / feedback render to give footage the polished VJ
look. The Look node takes an optional audio schedule to pulse the intensity to
the beat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.look import (  # noqa: E402
    LOOK_PRESETS,
    apply_look,
    color_grade,
    glow,
)

CATEGORY = "Difforum/look"


class DifforumLook:
    """One-shot VJ look: pick a preset, dial intensity (audio-reactive optional)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (list(LOOK_PRESETS), {"default": "neon"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
            "optional": {
                "intensity_schedule": ("DIFFORUM_SCHEDULE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, preset, intensity, intensity_schedule=None, seed=0):
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if intensity_schedule is None:
            return (apply_look(image, preset=preset, intensity=float(intensity), seed=int(seed)),)
        # per-frame intensity from the schedule (e.g. an audio curve), scaled by
        # the master intensity widget -> pulse the look to the beat
        frames = []
        for i in range(image.shape[0]):
            k = float(intensity) * float(intensity_schedule.at(i))
            frames.append(apply_look(image[i:i + 1], preset=preset, intensity=max(0.0, k), seed=int(seed) + i))
        return (torch.cat(frames, dim=0),)


class DifforumColorGrade:
    """Cinematic colour grade: exposure, contrast, saturation, white balance, LGG."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "lift": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.05}),
                "gain": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, exposure, contrast, saturation, temperature, tint,
            lift=0.0, gamma=1.0, gain=1.0):
        out = color_grade(image, exposure=float(exposure), contrast=float(contrast),
                          saturation=float(saturation), temperature=float(temperature),
                          tint=float(tint), lift=float(lift), gamma=float(gamma), gain=float(gain))
        return (out,)


class DifforumGlow:
    """Neon bloom: blur the bright areas and screen-blend them back."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("INT", {"default": 8, "min": 1, "max": 64}),
                "intensity": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, threshold, radius, intensity):
        return (glow(image, threshold=float(threshold), radius=int(radius), intensity=float(intensity)),)


NODE_CLASS_MAPPINGS = {
    "DifforumLook": DifforumLook,
    "DifforumColorGrade": DifforumColorGrade,
    "DifforumGlow": DifforumGlow,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumLook": "Difforum · VJ Look (presets)",
    "DifforumColorGrade": "Difforum · Colour Grade",
    "DifforumGlow": "Difforum · Glow",
}
