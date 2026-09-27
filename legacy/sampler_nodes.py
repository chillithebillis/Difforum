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

import torch


from ..core.color import COLOR_MODES, match_color  # noqa: E402
from ..core.detail import add_noise as _add_noise  # noqa: E402
from ..core.detail import sharpen as _sharpen  # noqa: E402
from ..core.fx import FxRunner  # noqa: E402
from ..core.symmetry import SYMMETRY_MODES, apply_symmetry  # noqa: E402
from ..core.warp import warp_2d, warp_3d  # noqa: E402

BORDER_MODES = ("reflection", "zeros", "border")

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
                "border": (list(BORDER_MODES), {"default": "reflection"}),
                "sharpen": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "noise": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005}),
                "cadence": ("INT", {"default": 1, "min": 1, "max": 12}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000000}),
                "seed_mode": (["fixed", "increment"], {"default": "fixed"}),
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
            symmetry="none", symmetry_segments=6, border="reflection",
            sharpen=0.2, noise=0.02, cadence=1, start_frame=0, end_frame=0,
            seed_mode="fixed"):
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

        fx = FxRunner()   # run the pixel-effect chain on cuda/mps when available

        # render range: chunked / resumable long videos. init_image = the frame
        # at start_frame; all schedules stay absolutely indexed so chunks align.
        f0 = max(0, int(start_frame))
        f1 = min(n, int(end_frame)) if int(end_frame) > 0 else n
        for f in range(f0 + 1, f1):
            delta = torch.as_tensor(camera.deltas[f], dtype=torch.float32)
            zoom = float(camera.zoom[f])
            fov = float(camera.fov[f])
            is_key = not (cadence > 1 and (f % cadence) != 0)

            def _chain(img, f=f, delta=delta, zoom=zoom, fov=fov, is_key=is_key):
                if camera.mode == "3d" and depth_b is not None:
                    d = depth_b[..., 0].to(img.device)
                    out, _m = warp_3d(
                        img, d, delta, fov_deg=fov,
                        near=float(near), far=float(far), invert_depth=bool(invert_depth),
                        translation_scale=float(translation_scale),
                    )
                else:
                    tx, ty = float(delta[0, 3]), float(delta[1, 3])
                    out, _m = warp_2d(img, tx, ty, _z_angle_deg(delta), zoom,
                                      padding_mode=border)
                # symmetry inside the loop: it compounds frame to frame and the
                # diffusion below heals the seams = a living kaleidoscope
                if symmetry != "none":
                    out = apply_symmetry(out, mode=symmetry,
                                         segments=int(symmetry_segments))
                # detail guard: every warp + VAE round-trip softens the frame,
                # so re-sharpen and inject fresh noise for the sampler to
                # resolve into detail (the classic anti-mush trick)
                if sharpen > 0.0:
                    out = _sharpen(out, float(sharpen))
                # noise only on frames that get diffused (nothing eats it on tweens)
                if noise > 0.0 and is_key:
                    out = _add_noise(out, float(noise), seed=seed + f)
                return out

            warped = fx(_chain, prev)

            # cadence: only diffuse every Nth frame; in-between frames are the
            # camera-warped feedback itself (the classic Deforum turbo mode).
            # Motion stays per-frame smooth while diffusion cost drops ~N times.
            if not is_key:
                prev = warped[:1].clamp(0.0, 1.0)
                frames.append(prev)
                pbar.update(1)
                continue

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
                    hint = ctrl_b[f % ctrl_b.shape[0]].unsqueeze(0)  # loop the sequence
                else:
                    hint = warped[:, :, :, :3]
                pos_f, neg_f = cn_apply(
                    pos_f, neg_f, control_net, hint,
                    float(control_strength), 0.0, 1.0, vae=vae,
                )

            latent = {"samples": vae.encode(warped[:, :, :, :3])}
            # fixed sampling seed keeps the diffusion noise identical every
            # frame, so texture stops re-rolling (the classic anti-boil trick)
            seed_f = seed if seed_mode == "fixed" else seed + f
            out_latent = common_ksampler(
                model, seed_f, int(steps), cfg_f, sampler_name, scheduler,
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
