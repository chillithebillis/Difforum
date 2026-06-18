"""
Difforum warp node - applies the camera motion to an image, Deforum-style.

Keeps the classic Deforum look as an explicit choice:
  - mode "2d": pure affine (translation + angle + zoom). The original look.
  - mode "3d": depth-based perspective parallax (needs a DEPTH/IMAGE input).
  - follow_camera_mode: take 2d/3d from the DIFFORUM_CAMERA track instead.

Outputs the warped image plus the occlusion MASK (holes the warp revealed) -
feed that mask to a sampler to re-diffuse only the new areas (the Deforum
feedback mechanism). This node is the shared front-end for both Classic+ and
Hybrid modes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.warp import warp_2d, warp_3d  # noqa: E402

CATEGORY = "Difforum/warp"


class DifforumWarp:
    """Warp an image by one frame of camera motion (2D affine or 3D depth)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "camera": ("DIFFORUM_CAMERA",),
                "frame": ("INT", {"default": 1, "min": 0, "max": 100000}),
                "warp_mode": (["follow_camera", "force_2d", "force_3d"], {"default": "follow_camera"}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "near": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0}),
                "far": ("FLOAT", {"default": 100.0, "min": 0.02, "max": 10000.0}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "translation_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("warped", "occlusion_mask")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, image, camera, frame, warp_mode,
            depth=None, near=1.0, far=100.0, invert_depth=False, translation_scale=1.0):
        n = len(camera.deltas)
        f = max(0, min(int(frame), n - 1))
        delta = torch.as_tensor(camera.deltas[f], dtype=torch.float32)
        zoom = float(camera.zoom[f])
        fov = float(camera.fov[f])

        mode = camera.mode
        if warp_mode == "force_2d":
            mode = "2d"
        elif warp_mode == "force_3d":
            mode = "3d"

        if mode == "3d" and depth is not None:
            d = depth
            if d.dim() == 4 and d.shape[-1] == 3:
                d = d.mean(dim=-1)  # rgb depth -> single channel
            warped, mask = warp_3d(
                image, d, delta, fov_deg=fov,
                near=float(near), far=float(far), invert_depth=bool(invert_depth),
                translation_scale=float(translation_scale),
            )
        else:
            # 2D affine from the delta: in-plane translation, z-rotation, zoom
            tx = float(delta[0, 3])
            ty = float(delta[1, 3])
            angle = self._z_angle_deg(delta)
            warped, mask = warp_2d(image, tx, ty, angle, zoom)

        # MASK type is [.,H,W]; our mask is [.,H,W,1] -> squeeze last dim
        mask = mask.squeeze(-1)
        return (warped, mask)

    @staticmethod
    def _z_angle_deg(delta) -> float:
        import math
        return math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0])))


NODE_CLASS_MAPPINGS = {
    "DifforumWarp": DifforumWarp,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumWarp": "Difforum · Warp (2D/3D)",
}
