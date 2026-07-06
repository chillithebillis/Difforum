"""
Difforum glitch nodes: procedural DSP databending and optical-flow datamosh.
Batch-native IMAGE to IMAGE, deterministic (seed), no per-instance state.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.glitch import KERNELS, apply_glitch  # noqa: E402

CATEGORY = "Difforum/effects"


class DifforumGlitch:
    """Convolution kernels + DSP delay line + bitcrush + VHS damage + RGB split."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "kernel": (list(KERNELS), {"default": "none"}),
                "kernel_mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "dsp_delay": ("INT", {"default": 0, "min": 0, "max": 500000, "step": 64}),
                "dsp_feedback": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05}),
                "bitcrush": ("INT", {"default": 0, "min": 0, "max": 7}),
                "vhs_jitter": ("INT", {"default": 0, "min": 0, "max": 64}),
                "vhs_band": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "chroma_blur": ("INT", {"default": 0, "min": 0, "max": 31}),
                "rgb_split": ("INT", {"default": 0, "min": 0, "max": 64}),
            },
            "optional": {"seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF})},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, kernel, kernel_mix, dsp_delay, dsp_feedback, bitcrush,
            vhs_jitter, vhs_band, chroma_blur, rgb_split, seed=0):
        out = apply_glitch(frames, kernel=kernel, kernel_mix=float(kernel_mix),
                           delay=int(dsp_delay), feedback=float(dsp_feedback),
                           bits=int(bitcrush), jitter=int(vhs_jitter),
                           band=float(vhs_band), chroma=int(chroma_blur),
                           rgb_split=int(rgb_split), seed=int(seed))
        return (out,)


class DifforumDatamosh:
    """Optical-flow datamosh: motion keeps flowing, the refresh melts away."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05}),
                "mode": (["grid", "melt", "edge"], {"default": "grid"}),
                "block_size": ("INT", {"default": 16, "min": 4, "max": 64}),
                "flow_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.25}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, intensity, mode, block_size, flow_scale):
        from core.flow import datamosh
        return (datamosh(frames, intensity=float(intensity), mode=mode,
                         block_size=int(block_size), flow_scale=float(flow_scale)),)


NODE_CLASS_MAPPINGS = {
    "DifforumGlitch": DifforumGlitch,
    "DifforumDatamosh": DifforumDatamosh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumGlitch": "Difforum · Glitch (DSP databending)",
    "DifforumDatamosh": "Difforum · Datamosh (optical flow)",
}
