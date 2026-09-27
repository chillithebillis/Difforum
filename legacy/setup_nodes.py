"""
Difforum · Anim Setup+ - duration in frames or seconds, framing by aspect ratio.

Same DIFFORUM_PARAMS as the original Anim Setup, so it drops into any graph in
its place. What changes is that the two things you actually decide - how long
the clip runs and how it is framed - are stated the way you think about them,
and the resolution is derived instead of typed.
"""

from __future__ import annotations

CATEGORY = "Difforum"

# name -> (w, h) ratio
ASPECTS = {
    "16:9  landscape":      (16, 9),
    "9:16  vertical":       (9, 16),
    "4:3   classic":        (4, 3),
    "3:4   portrait":       (3, 4),
    "1:1   square":         (1, 1),
    "21:9  ultrawide":      (21, 9),
    "2.39:1 anamorphic":    (239, 100),
    "2:1   univisium":      (2, 1),
    "custom":               (0, 0),
}

# name -> long edge in pixels
RESOLUTIONS = {
    "384  thumbnail": 384,
    "512  draft": 512,
    "640  fast": 640,
    "768  standard": 768,
    "896  high": 896,
    "1024 max": 1024,
    "1280 oversize": 1280,
    "custom": 0,
}

# Rough ceilings in total pixels, past which a feedback render on that machine
# tends to hit the allocator before it finishes the clip. Unified memory is
# shared with the OS, so the Mac entries are well under the nominal figure.
BUDGETS = {
    "mac 16GB":     560 * 560,
    "mac 24GB":     820 * 820,
    "mac 32GB":    1000 * 1000,
    "mac 64GB+":   1400 * 1400,
    "gpu 8GB":      620 * 620,
    "gpu 12GB":     820 * 820,
    "gpu 16GB":     960 * 960,
    "gpu 24GB+":   1400 * 1400,
    "no limit":     0,
}


def _snap(v: int, multiple: int = 8) -> int:
    return max(multiple, int(round(v / multiple)) * multiple)


def resolve_size(aspect: str, long_edge: int, multiple: int = 8) -> tuple[int, int]:
    aw, ah = ASPECTS.get(aspect, (16, 9))
    if aw <= 0 or ah <= 0:
        return _snap(long_edge, multiple), _snap(long_edge, multiple)
    if aw >= ah:
        w = long_edge
        h = long_edge * ah / aw
    else:
        h = long_edge
        w = long_edge * aw / ah
    return _snap(int(round(w)), multiple), _snap(int(round(h)), multiple)


class DifforumAnimSetupPlus:
    """Global animation parameters, stated as duration and framing.

    `duration_mode` switches the number below it between frames and seconds -
    at 24 fps, 5 seconds and 120 frames are the same clip, so pick whichever
    matches how the piece is being planned.

    The resolution comes from an aspect ratio and a long edge, then gets clamped
    to whatever the chosen hardware can actually finish a feedback render at.
    Both dimensions are snapped to a multiple of 8 (set 16 for any Wan path, or
    its VAE will fail at decode).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "duration_mode": (["frames", "seconds"], {"default": "seconds"}),
                "duration": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 100000.0,
                                       "step": 0.1}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "aspect": (list(ASPECTS), {"default": "16:9  landscape"}),
                "resolution": (list(RESOLUTIONS), {"default": "768  standard"}),
                "fit_to": (list(BUDGETS), {"default": "mac 24GB"}),
                "snap": ([8, 16], {"default": 8}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "custom_long_edge": ("INT", {"default": 768, "min": 64, "max": 8192,
                                             "step": 8}),
                "custom_aspect_w": ("FLOAT", {"default": 16.0, "min": 0.1, "max": 100.0,
                                              "step": 0.1}),
                "custom_aspect_h": ("FLOAT", {"default": 9.0, "min": 0.1, "max": 100.0,
                                              "step": 0.1}),
            },
        }

    RETURN_TYPES = ("DIFFORUM_PARAMS", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("params", "width", "height", "max_frames", "info")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, duration_mode, duration, fps, aspect, resolution, fit_to,
              snap, seed, custom_long_edge=768, custom_aspect_w=16.0,
              custom_aspect_h=9.0):
        fps = max(1.0, float(fps))
        mult = int(snap)

        if duration_mode == "seconds":
            max_frames = max(1, int(round(float(duration) * fps)))
        else:
            max_frames = max(1, int(round(float(duration))))

        long_edge = RESOLUTIONS.get(resolution, 768) or int(custom_long_edge)

        if aspect == "custom":
            aw, ah = max(0.1, float(custom_aspect_w)), max(0.1, float(custom_aspect_h))
            if aw >= ah:
                w, h = long_edge, int(round(long_edge * ah / aw))
            else:
                w, h = int(round(long_edge * aw / ah)), long_edge
            w, h = _snap(w, mult), _snap(h, mult)
            ratio_txt = f"{aw:g}:{ah:g}"
        else:
            w, h = resolve_size(aspect, long_edge, mult)
            ratio_txt = aspect.split()[0]

        notes = []
        budget = BUDGETS.get(fit_to, 0)
        if budget and w * h > budget:
            scale = (budget / (w * h)) ** 0.5
            w2, h2 = _snap(int(w * scale), mult), _snap(int(h * scale), mult)
            notes.append(
                f"  clamped {w}x{h} -> {w2}x{h2} to fit '{fit_to}' "
                f"({w * h / 1e6:.2f}MP over the {budget / 1e6:.2f}MP budget)"
            )
            w, h = w2, h2

        seconds = max_frames / fps
        if seconds > 30:
            notes.append(
                f"  {seconds:.0f}s is a long feedback render - consider chunking "
                "with the Feedback Sampler's start_frame / end_frame"
            )
        if mult == 8:
            notes.append("  snap=8 is right for SD/SDXL/Flux; use 16 for Wan 2.2")

        params = {
            "width": int(w),
            "height": int(h),
            "fps": float(fps),
            "max_frames": int(max_frames),
            "seed": int(seed),
        }

        info = "\n".join([
            f"anim setup+  {w}x{h}  {ratio_txt}  {fps:g}fps",
            f"  {max_frames} frames = {seconds:.2f}s   ({duration_mode} mode)",
            f"  {w * h / 1e6:.2f} MP/frame   budget '{fit_to}'",
            *notes,
        ])
        return (params, int(w), int(h), int(max_frames), info)


NODE_CLASS_MAPPINGS = {
    "DifforumAnimSetupPlus": DifforumAnimSetupPlus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumAnimSetupPlus": "Difforum · Anim Setup+ (time + aspect)",
}
