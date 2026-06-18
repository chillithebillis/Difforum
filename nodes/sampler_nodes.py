"""
Difforum Feedback Sampler - the Classic+ mode (modern Deforum feedback loop).

For each frame: warp the previous frame by the camera (2D affine or 3D depth),
img2img re-diffuse it at the scheduled denoise strength, then colour-match to an
anchor to stop drift. This is the recognizable Deforum morphing look, rebuilt
with modern depth and stable colour.

Uses ComfyUI's stock sampling (common_ksampler) + VAE encode/decode, imported
lazily so the package still loads outside a full ComfyUI runtime.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.color import COLOR_MODES, match_color  # noqa: E402
from core.symmetry import SYMMETRY_MODES, apply_symmetry  # noqa: E402
from core.warp import warp_2d, warp_3d  # noqa: E402

CATEGORY = "Difforum/render"


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


def _z_angle_deg(delta) -> float:
    return math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0])))


class DifforumFeedbackSampler:
    """Generate a Deforum-style animation via the warp->re-diffuse feedback loop."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "params": ("DIFFORUM_PARAMS",),
                "camera": ("DIFFORUM_CAMERA",),
                "init_image": ("IMAGE",),
                "strength_schedule": ("DIFFORUM_SCHEDULE",),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "color_coherence": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_mode": (list(COLOR_MODES), {"default": "lab"}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "cfg_schedule": ("DIFFORUM_SCHEDULE",),
                "positive_schedule": ("DIFFORUM_PROMPT",),
                "near": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0}),
                "far": ("FLOAT", {"default": 100.0, "min": 0.02, "max": 10000.0}),
                "invert_depth": ("BOOLEAN", {"default": False}),
                "translation_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "control_net": ("CONTROL_NET",),
                "control_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05}),
                "control_image": ("IMAGE",),
                "symmetry": (list(SYMMETRY_MODES), {"default": "none"}),
                "symmetry_segments": ("INT", {"default": 6, "min": 2, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, positive, negative, vae, params, camera, init_image,
            strength_schedule, steps, cfg, sampler_name, scheduler, color_coherence,
            color_mode="lab", depth=None, cfg_schedule=None, positive_schedule=None,
            near=1.0, far=100.0, invert_depth=False, translation_scale=1.0,
            control_net=None, control_strength=0.6, control_image=None,
            symmetry="none", symmetry_segments=6):
        import comfy.utils
        from nodes import common_ksampler

        cn_apply = None
        if control_net is not None and control_strength > 0.0:
            from nodes import ControlNetApplyAdvanced
            cn_apply = ControlNetApplyAdvanced().apply_controlnet
        ctrl_b = None
        if control_image is not None:
            ctrl_b = _resize_bhwc(control_image, params["width"], params["height"])

        w, h = params["width"], params["height"]
        n = params["max_frames"]
        seed = int(params.get("seed", 0))

        prev = _resize_bhwc(init_image, w, h)[:1]  # frame 0
        anchor = prev.clone()
        frames = [prev]

        depth_b = None
        if depth is not None:
            depth_b = _resize_bhwc(depth, w, h)
            if depth_b.shape[-1] == 3:
                depth_b = depth_b.mean(dim=-1, keepdim=True)

        pbar = comfy.utils.ProgressBar(n)
        pbar.update(1)

        for f in range(1, n):
            delta = torch.as_tensor(camera.deltas[f], dtype=torch.float32)
            zoom = float(camera.zoom[f])
            fov = float(camera.fov[f])

            if camera.mode == "3d" and depth_b is not None:
                warped, _mask = warp_3d(
                    prev, depth_b[..., 0], delta, fov_deg=fov,
                    near=float(near), far=float(far), invert_depth=bool(invert_depth),
                    translation_scale=float(translation_scale),
                )
            else:
                tx, ty = float(delta[0, 3]), float(delta[1, 3])
                warped, _mask = warp_2d(prev, tx, ty, _z_angle_deg(delta), zoom)

            # symmetry inside the loop: it compounds frame to frame and the
            # diffusion below heals the seams = a living kaleidoscope
            if symmetry != "none":
                warped = apply_symmetry(warped, mode=symmetry,
                                        segments=int(symmetry_segments))

            # img2img re-diffuse the warped frame
            denoise = max(0.0, min(1.0, float(strength_schedule.at(f))))
            cfg_f = float(cfg_schedule.at(f)) if cfg_schedule is not None else float(cfg)
            # prompt travel: pick this frame's blended conditioning if provided
            pos_f = positive
            if positive_schedule is not None and len(positive_schedule) > 0:
                pos_f = positive_schedule[min(f, len(positive_schedule) - 1)]
            neg_f = negative

            # ControlNet: hint = an external control video frame if given, else
            # the warped frame (keeps structure aligned to the camera per frame)
            if cn_apply is not None:
                if ctrl_b is not None:
                    hint = ctrl_b[min(f, ctrl_b.shape[0] - 1)].unsqueeze(0)
                else:
                    hint = warped[:, :, :, :3]
                pos_f, neg_f = cn_apply(
                    pos_f, neg_f, control_net, hint,
                    float(control_strength), 0.0, 1.0, vae=vae,
                )

            latent = {"samples": vae.encode(warped[:, :, :, :3])}
            out_latent = common_ksampler(
                model, seed + f, int(steps), cfg_f, sampler_name, scheduler,
                pos_f, neg_f, latent, denoise=denoise,
            )[0]
            image = vae.decode(out_latent["samples"])

            if color_coherence > 0.0 and color_mode != "none":
                image = match_color(image, anchor, strength=float(color_coherence), mode=color_mode)

            prev = image[:1]
            frames.append(prev)
            pbar.update(1)

        return (torch.cat(frames, dim=0),)


NODE_CLASS_MAPPINGS = {
    "DifforumFeedbackSampler": DifforumFeedbackSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumFeedbackSampler": "Difforum · Feedback Sampler (Classic+)",
}
