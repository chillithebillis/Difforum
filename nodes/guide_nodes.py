"""
Difforum Guide Builder - the robust hybrid bridge to Wan 2.2 (VACE/FLF2V).

Instead of coupling to WanVideoWrapper's internal API (fragile across
versions), this produces a *standard* sequence of guide frames by warping an
anchor image along the accumulated camera path. The resulting IMAGE batch (and
occlusion MASK batch) plugs straight into the VACE control/reference inputs of
whatever Wan node graph the user already has.

So the camera stays 100% Difforum (math + audio reactive), while Wan does the
temporally-coherent fill. Pure torch + core.warp; no heavy imports at load.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.warp import warp_2d, warp_3d  # noqa: E402

CATEGORY = "Difforum/hybrid"


def _resize_bhwc(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    b, h, w, c = image.shape
    if h == height and w == width:
        return image
    chw = image.permute(0, 3, 1, 2)
    chw = torch.nn.functional.interpolate(
        chw, size=(height, width), mode="bilinear", align_corners=False
    )
    return chw.permute(0, 2, 3, 1)


class DifforumGuideBuilder:
    """Warp an anchor image along the camera path into a Wan-ready guide batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchor_image": ("IMAGE",),
                "camera": ("DIFFORUM_CAMERA",),
                "params": ("DIFFORUM_PARAMS",),
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

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("guide_frames", "occlusion_masks", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, anchor_image, camera, params, warp_mode,
            depth=None, near=1.0, far=100.0, invert_depth=False, translation_scale=1.0):
        w, h = params["width"], params["height"]
        n = params["max_frames"]
        anchor = _resize_bhwc(anchor_image, w, h)[:1]

        mode = camera.mode
        if warp_mode == "force_2d":
            mode = "2d"
        elif warp_mode == "force_3d":
            mode = "3d"

        depth_c = None
        if depth is not None:
            d = _resize_bhwc(depth, w, h)
            depth_c = d.mean(dim=-1) if d.shape[-1] == 3 else d[..., 0]

        use_3d = mode == "3d" and depth_c is not None
        guides, masks = [], []
        cum_zoom = 1.0
        for f in range(n):
            pose = torch.as_tensor(camera.poses[f], dtype=torch.float32)
            cum_zoom *= float(camera.zoom[f]) if f > 0 else 1.0
            if use_3d:
                g, m = warp_3d(
                    anchor, depth_c, pose, fov_deg=float(camera.fov[f]),
                    near=float(near), far=float(far), invert_depth=bool(invert_depth),
                    translation_scale=float(translation_scale),
                )
            else:
                tx, ty = float(pose[0, 3]), float(pose[1, 3])
                angle = math.degrees(math.atan2(float(pose[1, 0]), float(pose[0, 0])))
                g, m = warp_2d(anchor, tx, ty, angle, cum_zoom)
            guides.append(g)
            masks.append(m.squeeze(-1))

        guide_batch = torch.cat(guides, dim=0)
        mask_batch = torch.cat(masks, dim=0)
        info = (
            f"built {n} guide frames ({w}x{h}) mode={'3d' if use_3d else '2d'}\n"
            "wire guide_frames into Wan VACE control input, anchor_image into the\n"
            "VACE reference image, then sample with your Wan 2.2 graph."
        )
        return (guide_batch, mask_batch, info)


NODE_CLASS_MAPPINGS = {
    "DifforumGuideBuilder": DifforumGuideBuilder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumGuideBuilder": "Difforum · Guide Builder (Wan VACE)",
}
