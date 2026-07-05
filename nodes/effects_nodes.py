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

from core.detail import NOISE_MODES, detail_guard  # noqa: E402
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


class DifforumDetailGuard:
    """Fight feedback mush: unsharp mask + contrast + noise injection."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sharpen": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
                "noise_mode": (list(NOISE_MODES), {"default": "gaussian"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, sharpen, noise, contrast, noise_mode, seed=0):
        out = detail_guard(image, sharpen_amount=float(sharpen), noise_amount=float(noise),
                           contrast=float(contrast), seed=int(seed), noise_mode=noise_mode)
        return (out,)


class DifforumFlowStabilize:
    """Anti-flicker: blend history along optical flow, gated by photometric
    confidence, so texture stops boiling but motion never ghosts."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05}),
                "flow_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.25}),
                "error_gate": ("FLOAT", {"default": 0.15, "min": 0.02, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, frames, strength, flow_scale, error_gate):
        from core.flow import stabilize
        return (stabilize(frames, strength=float(strength),
                          flow_scale=float(flow_scale), error_gate=float(error_gate)),)


NODE_CLASS_MAPPINGS = {
    "DifforumSymmetry": DifforumSymmetry,
    "DifforumEchoTrails": DifforumEchoTrails,
    "DifforumDetailGuard": DifforumDetailGuard,
    "DifforumFlowStabilize": DifforumFlowStabilize,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumSymmetry": "Difforum · Symmetry / Kaleidoscope",
    "DifforumEchoTrails": "Difforum · Echo Trails",
    "DifforumDetailGuard": "Difforum · Detail Guard (anti-mush)",
    "DifforumFlowStabilize": "Difforum · Flow Stabilize (anti-flicker)",
}
