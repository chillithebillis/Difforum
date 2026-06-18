"""
Difforum visual effects nodes: symmetry / kaleidoscope and echo trails.

Both are plain IMAGE -> IMAGE and work on a single frame or a whole batch, so
drop them after the Feedback Sampler (or before Save / Video Combine) for
mesmerizing, satisfying output. Symmetry also feeds back nicely (see the
Feedback Sampler's `symmetry` option for a compounding kaleidoscope).
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.effects import echo_trails  # noqa: E402
from core.symmetry import SYMMETRY_MODES, apply_symmetry  # noqa: E402

CATEGORY = "Difforum/effects"


class DifforumSymmetry:
    """Mirror / kaleidoscope an image or frame batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (list(SYMMETRY_MODES), {"default": "mirror_h"}),
                "segments": ("INT", {"default": 6, "min": 2, "max": 64}),
                "flip": ("BOOLEAN", {"default": False}),
                "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, mode, segments, flip, mix, center_x=0.5, center_y=0.5, angle=0.0):
        out = apply_symmetry(image, mode=mode, segments=int(segments), flip=bool(flip),
                            mix=float(mix), center_x=float(center_x),
                            center_y=float(center_y), angle=float(angle))
        return (out,)


class DifforumEchoTrails:
    """Long-exposure motion trails across a frame batch (smooth, hypnotic)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "decay": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.99, "step": 0.01}),
                "mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, decay, mix):
        return (echo_trails(frames, decay=float(decay), mix=float(mix)),)


NODE_CLASS_MAPPINGS = {
    "DifforumSymmetry": DifforumSymmetry,
    "DifforumEchoTrails": DifforumEchoTrails,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumSymmetry": "Difforum · Symmetry / Kaleidoscope",
    "DifforumEchoTrails": "Difforum · Echo Trails",
}
