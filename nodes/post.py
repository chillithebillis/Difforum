"""Post: loops, symmetry, trails, anti-flicker, detail."""

from __future__ import annotations

from ..core.detail import NOISE_MODES, detail_guard
from ..core.effects import echo_trails, loop_blend, pingpong
from ..core.loop import seam_report
from ..core.symmetry import SYMMETRY_MODES, apply_symmetry
from ._common import CAT_POST

LOOP_METHODS = ("keep settled lap", "flow crossfade", "ping-pong")


class DifforumLoop:
    """Make a clip loop.

    * keep settled lap - for renders made with a looping camera (Director /
      Camera loop_mode) over several laps: keeps the last lap, no blending.
    * flow crossfade - morphs the tail into the head along optical flow.
    * ping-pong - forward then backward (motion reverses).
    Reports how visible the seam is.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "method": (list(LOOP_METHODS), {"default": "flow crossfade"}),
            "cycle_frames": ("INT", {"default": 0, "min": 0, "max": 100000,
                             "tooltip": "keep settled lap: frames per lap (the Camera node outputs it)."}),
            "blend_frames": ("INT", {"default": 12, "min": 1, "max": 240,
                             "tooltip": "flow crossfade: overlap length."}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "seam_info")
    FUNCTION = "run"
    CATEGORY = CAT_POST

    def run(self, frames, method, cycle_frames, blend_frames):
        total = int(frames.shape[0])
        if method == "keep settled lap":
            cyc = int(cycle_frames) if int(cycle_frames) > 0 else total
            cyc = max(2, min(cyc, total))
            out = frames[total - cyc:]
        elif method == "ping-pong":
            out = pingpong(frames)
        else:
            out = loop_blend(frames, blend=min(int(blend_frames), max(1, total // 2)), flow=True)
        report = seam_report([out[i] for i in range(out.shape[0])], int(out.shape[0]))
        return (out, f"{method}: {total} -> {out.shape[0]} frames\n{report}")


class DifforumSymmetry:
    """Mirror or kaleidoscope a frame or batch (the Feedback Sampler can also
    do it inside the loop, where it compounds into a living pattern)."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "mode": (list(SYMMETRY_MODES), {"default": "kaleidoscope"}),
            "segments": ("INT", {"default": 6, "min": 2, "max": 64}),
            "mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
        }, "optional": {
            "flip": ("BOOLEAN", {"default": False}),
            "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CAT_POST

    def run(self, image, mode, segments, mix, flip=False, center_x=0.5, center_y=0.5, angle=0.0):
        return (apply_symmetry(image, mode=mode, segments=int(segments), flip=bool(flip),
                               mix=float(mix), center_x=float(center_x),
                               center_y=float(center_y), angle=float(angle)),)


class DifforumEchoTrails:
    """Long-exposure trails across a frame batch."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "decay": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 0.99, "step": 0.01}),
            "mix": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CAT_POST

    def run(self, frames, decay, mix):
        return (echo_trails(frames, decay=float(decay), mix=float(mix)),)


class DifforumFlowStabilize:
    """Anti-flicker: blends history along optical flow, only where it matches,
    so texture stops boiling but motion never ghosts."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "frames": ("IMAGE",),
            "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.95, "step": 0.05}),
            "flow_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.25}),
            "error_gate": ("FLOAT", {"default": 0.15, "min": 0.02, "max": 0.5, "step": 0.01}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CAT_POST

    def run(self, frames, strength, flow_scale, error_gate):
        from ..core.flow import stabilize
        return (stabilize(frames, strength=float(strength), flow_scale=float(flow_scale),
                          error_gate=float(error_gate)),)


class DifforumDetailGuard:
    """Unsharp + contrast + grain, for frames that went soft."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "sharpen": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
            "contrast": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 2.0, "step": 0.01}),
            "grain": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.005}),
            "grain_mode": (list(NOISE_MODES), {"default": "gaussian"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = CAT_POST

    def run(self, image, sharpen, contrast, grain, grain_mode):
        return (detail_guard(image, sharpen_amount=float(sharpen), noise_amount=float(grain),
                             contrast=float(contrast), noise_mode=grain_mode),)


NODE_CLASS_MAPPINGS = {
    "Difforum_Loop": DifforumLoop,
    "Difforum_Symmetry": DifforumSymmetry,
    "Difforum_EchoTrails": DifforumEchoTrails,
    "Difforum_FlowStabilize": DifforumFlowStabilize,
    "Difforum_DetailGuard": DifforumDetailGuard,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_Loop": "Difforum · Loop",
    "Difforum_Symmetry": "Difforum · Symmetry",
    "Difforum_EchoTrails": "Difforum · Echo Trails",
    "Difforum_FlowStabilize": "Difforum · Flow Stabilize",
    "Difforum_DetailGuard": "Difforum · Detail Guard",
}
