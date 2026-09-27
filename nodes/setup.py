"""Difforum · Setup - one place for duration, framing and the target model grid."""

from __future__ import annotations

from ._common import CAT_SETUP, PARAMS

ASPECTS = {
    "16:9 landscape": (16, 9),
    "9:16 vertical": (9, 16),
    "1:1 square": (1, 1),
    "4:5 social portrait": (4, 5),
    "4:3 classic": (4, 3),
    "3:4 portrait": (3, 4),
    "21:9 ultrawide": (21, 9),
    "2.39:1 anamorphic": (239, 100),
    "2:1 univisium": (2, 1),
    "custom": (0, 0),
}

# target -> (frame rule, spatial multiple, native fps or None)
#   frame rule (a, b): valid lengths are a*k + b
TARGETS = {
    "feedback (SD/SDXL/Flux)": ((1, 0), 8, None),
    "LTX-2 / 2.5": ((8, 1), 32, None),
    "MiniMax H3": ((17, 5), 32, 24.0),
    "Wan 2.x": ((4, 1), 16, None),
}


def snap_frames(n: int, rule: tuple[int, int]) -> int:
    """Nearest valid length a*k+b (never below b, never 0)."""
    a, b = rule
    if a <= 1:
        return max(1, int(n))
    k = max(0, round((int(n) - b) / a))
    return max(b if b > 0 else a, a * k + b)


def snap(v: float, multiple: int) -> int:
    return max(multiple, int(round(v / multiple)) * multiple)


def resolve_size(aspect: tuple[float, float], long_edge: int, multiple: int) -> tuple[int, int]:
    aw, ah = aspect
    if aw <= 0 or ah <= 0:
        return snap(long_edge, multiple), snap(long_edge, multiple)
    if aw >= ah:
        w, h = long_edge, long_edge * ah / aw
    else:
        w, h = long_edge * aw / ah, long_edge
    return snap(w, multiple), snap(h, multiple)


class DifforumSetup:
    """Duration, framing and the grid of the model you will render with.

    Say how long the clip is (seconds or frames) and how it is framed (aspect +
    long edge); the resolution is derived, snapped to what the target model's
    VAE accepts, and the frame count is snapped to the model's length grid
    (LTX 8k+1, MiniMax H3 17k+5 at 24 fps, Wan 4k+1). Every other Difforum node
    reads these params, so the whole graph agrees on one timeline.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "target": (list(TARGETS), {"default": "feedback (SD/SDXL/Flux)",
                           "tooltip": "Snaps size and length to what this model family accepts."}),
                "duration_mode": (["seconds", "frames"], {"default": "seconds"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100000.0, "step": 0.1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "aspect": (list(ASPECTS), {"default": "16:9 landscape"}),
                "long_edge": ("INT", {"default": 768, "min": 128, "max": 8192, "step": 32,
                              "tooltip": "Pixels on the longer side before snapping."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "custom_aspect_w": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "custom_aspect_h": ("FLOAT", {"default": 9.0, "min": 0.1, "max": 100.0, "step": 0.1}),
                "max_megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.05,
                                   "tooltip": "0 = no cap. Otherwise shrink to fit (e.g. 0.35 for a 16 GB Mac)."}),
            },
        }

    RETURN_TYPES = (PARAMS, "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("params", "width", "height", "frames", "fps", "info")
    FUNCTION = "build"
    CATEGORY = CAT_SETUP

    def build(self, target, duration_mode, duration, fps, aspect, long_edge, seed,
              custom_aspect_w=16.0, custom_aspect_h=9.0, max_megapixels=0.0):
        rule, multiple, native_fps = TARGETS.get(target, TARGETS["feedback (SD/SDXL/Flux)"])
        notes = []
        fps = max(1.0, float(fps))
        if native_fps and abs(fps - native_fps) > 1e-6:
            notes.append(f"  {target} renders at {native_fps:g} fps - fps set to {native_fps:g}")
            fps = native_fps

        raw = float(duration) * fps if duration_mode == "seconds" else float(duration)
        frames = snap_frames(int(round(raw)), rule)
        if frames != int(round(raw)):
            notes.append(f"  length snapped {int(round(raw))} -> {frames} frames "
                         f"({rule[0]}k+{rule[1]} grid)")

        ratio = ASPECTS.get(aspect, (16, 9))
        if ratio == (0, 0):
            ratio = (max(0.1, float(custom_aspect_w)), max(0.1, float(custom_aspect_h)))
        w, h = resolve_size(ratio, int(long_edge), multiple)

        if max_megapixels and w * h > max_megapixels * 1e6:
            s = (max_megapixels * 1e6 / (w * h)) ** 0.5
            w2, h2 = snap(w * s, multiple), snap(h * s, multiple)
            notes.append(f"  capped {w}x{h} -> {w2}x{h2} ({max_megapixels:g} MP)")
            w, h = w2, h2

        params = {"width": int(w), "height": int(h), "fps": float(fps),
                  "max_frames": int(frames), "seed": int(seed), "target": target}
        info = "\n".join([
            f"setup  {w}x{h}  {fps:g} fps  {frames} frames = {frames / fps:.2f}s",
            f"  target {target}  (size multiple {multiple})",
            *notes,
        ])
        return (params, int(w), int(h), int(frames), float(fps), info)


NODE_CLASS_MAPPINGS = {"Difforum_Setup": DifforumSetup}
NODE_DISPLAY_NAME_MAPPINGS = {"Difforum_Setup": "Difforum · Setup"}
