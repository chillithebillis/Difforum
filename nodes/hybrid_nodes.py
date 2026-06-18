"""
Difforum hybrid-pipeline nodes: camera engine + model/VRAM profile.

DifforumCamera     - Deforum-style camera schedules -> DIFFORUM_CAMERA track
DifforumModelProfile - resolve render settings for the GPU (12..32GB+), GGUF-aware

These are the control-plane of the hybrid mode. A later DifforumHybridRender
node consumes a DIFFORUM_CAMERA + DIFFORUM_PROFILE to warp guide frames and drive
Wan 2.2 (VACE/FLF2V). See DESIGN.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core import build_schedule  # noqa: E402
from core.camera import CAMERA_MODES, build_camera  # noqa: E402
from core.camera_presets import CAMERA_PRESETS, preset_schedules  # noqa: E402
from core.profiles import FAMILIES, QUALITIES, resolve_profile, summarize  # noqa: E402

CATEGORY = "Difforum/hybrid"

_CAM_FIELDS = (
    "translation_x", "translation_y", "translation_z",
    "rotation_3d_x", "rotation_3d_y", "rotation_3d_z",
    "zoom",
)
_CAM_DEFAULTS = {
    "translation_x": "0:(0)",
    "translation_y": "0:(0)",
    "translation_z": "0:(1.5)",          # gentle forward dolly
    "rotation_3d_x": "0:(0)",
    "rotation_3d_y": "0:(0.5*sin(2*pi*t/120))",  # slow sway
    "rotation_3d_z": "0:(0)",
    "zoom": "0:(1.0)",
}


class DifforumCamera:
    """Deforum-style camera motion from per-field schedule strings."""

    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "params": ("DIFFORUM_PARAMS",),
            "mode": (list(CAMERA_MODES), {"default": "3d"}),
            "fov": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 170.0, "step": 1.0}),
        }
        for f in _CAM_FIELDS:
            req[f] = ("STRING", {"multiline": False, "default": _CAM_DEFAULTS[f]})
        return {"required": req, "optional": {"audio": ("DIFFORUM_AUDIO",)}}

    RETURN_TYPES = ("DIFFORUM_CAMERA", "STRING")
    RETURN_NAMES = ("camera", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, mode, fov, audio=None, **fields):
        extra = audio.get("curves") if isinstance(audio, dict) else None
        n = params["max_frames"]
        schedules = {}
        for f in _CAM_FIELDS:
            sched = build_schedule(
                fields[f], max_frames=n, fps=params["fps"], extra_vars=extra
            )
            schedules[f] = sched.as_list()
        cam = build_camera(schedules, max_frames=n, mode=mode, fov=fov)
        # quick textual summary of total motion
        last = cam.poses[-1]
        info = (
            f"camera mode={mode} frames={n} fov={fov}\n"
            f"net translation = ({last[0,3]:.2f}, {last[1,3]:.2f}, {last[2,3]:.2f})\n"
            f"zoom {cam.zoom[0]:.3f} -> {cam.zoom[-1]:.3f}"
        )
        return (cam, info)


class DifforumCameraMove:
    """Intuitive camera control: pick a move preset + speed + intensity."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "preset": (list(CAMERA_PRESETS), {"default": "spiral"}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "mode": (list(CAMERA_MODES), {"default": "2d"}),
                "fov": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 170.0, "step": 1.0}),
            },
            "optional": {"audio": ("DIFFORUM_AUDIO",)},
        }

    RETURN_TYPES = ("DIFFORUM_CAMERA", "STRING")
    RETURN_NAMES = ("camera", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, preset, speed, intensity, mode, fov, audio=None):
        extra = audio.get("curves") if isinstance(audio, dict) else None
        n = params["max_frames"]
        strs = preset_schedules(preset, speed=speed, intensity=intensity)
        schedules = {
            ax: build_schedule(expr, max_frames=n, fps=params["fps"], extra_vars=extra).as_list()
            for ax, expr in strs.items()
        }
        cam = build_camera(schedules, max_frames=n, mode=mode, fov=fov)
        info = f"preset={preset} speed={speed} intensity={intensity} mode={mode}\n" + \
               "\n".join(f"  {k}: {v}" for k, v in strs.items() if v not in ("0:(0)", "0:(1.0)"))
        return (cam, info)


class DifforumModelProfile:
    """Resolve render settings for the GPU (model variant, quant, res, steps)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "family": (list(FAMILIES), {"default": "wan22"}),
                "quality": (list(QUALITIES), {"default": "balanced"}),
                "auto_detect_vram": ("BOOLEAN", {"default": True}),
                "vram_gb": ("INT", {"default": 12, "min": 4, "max": 192}),
                "attention": (["sdpa", "sage"], {"default": "sdpa"}),
            }
        }

    RETURN_TYPES = ("DIFFORUM_PROFILE", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("profile", "summary", "width", "height", "segment_frames")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def _detect_vram(self, fallback: int) -> int:
        try:
            import torch
            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory
                return max(4, int(total / (1024 ** 3)))
        except Exception:  # noqa: BLE001
            pass
        return fallback

    def run(self, family, quality, auto_detect_vram, vram_gb, attention):
        vram = self._detect_vram(vram_gb) if auto_detect_vram else int(vram_gb)
        prof = resolve_profile(vram, family=family, quality=quality, attention=attention)
        text = f"VRAM={vram}GB\n" + summarize(prof)
        bundle = prof.as_dict()
        bundle["vram_gb"] = vram
        return {
            "ui": {"text": [text]},
            "result": (bundle, text, prof.width, prof.height, prof.segment_frames),
        }


NODE_CLASS_MAPPINGS = {
    "DifforumCamera": DifforumCamera,
    "DifforumCameraMove": DifforumCameraMove,
    "DifforumModelProfile": DifforumModelProfile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumCamera": "Difforum · Camera (advanced)",
    "DifforumCameraMove": "Difforum · Camera Move (presets)",
    "DifforumModelProfile": "Difforum · Model Profile",
}
