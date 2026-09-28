"""Restyle (vid2vid look pass), Upscale, and the Workflow Switches panel."""

from __future__ import annotations

import math
import time

import torch

from ._common import (
    CAT_POST, CAT_RENDER, CAT_SETUP, PROMPT, call_comfy_node, check_interrupt, progress_bar,
)
from .direction import DIRECTION

# ---------------------------------------------------------------------------
# Restyle
# ---------------------------------------------------------------------------

RESTYLE_STYLES = {
    # feedback: how much of the previous stylized frame (carried along optical
    # flow) goes into the next one; hold: colour held to the source frame;
    # noise: extra grain before sampling; per_frame_seed: texture re-rolls.
    "clean restyle": dict(feedback=0.0, per_frame_seed=False, hold=0.7, denoise_mul=1.0, noise=0.0),
    "animatediff boil": dict(feedback=0.15, per_frame_seed=True, hold=0.6, denoise_mul=1.0, noise=0.01),
    "deforum morph": dict(feedback=0.55, per_frame_seed=False, hold=0.35, denoise_mul=1.0, noise=0.02),
    "disco flicker": dict(feedback=0.25, per_frame_seed=True, hold=0.15, denoise_mul=1.25, noise=0.04),
}


def _samplers():
    try:
        import comfy.samplers
        return comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS
    except Exception:
        return ["euler", "lcm"], ["normal", "sgm_uniform"]


def _gray_u8(img: torch.Tensor):
    import numpy as np
    return (img[0].mean(dim=-1).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _size(h: int, w: int, long_edge: int) -> tuple[int, int]:
    if long_edge <= 0:
        return w // 8 * 8, h // 8 * 8
    s = long_edge / max(h, w)
    return max(16, round(w * s / 8) * 8), max(16, round(h * s / 8) * 8)


class _Mapped:
    """Index a per-frame track made for another length."""

    def __init__(self, track, n_src: int, n: int):
        self.track, self.n_src, self.n = track, max(1, n_src), max(1, n)

    def _i(self, f: int) -> int:
        return min(self.n_src - 1, round(f * (self.n_src - 1) / max(1, self.n - 1)))

    def __getitem__(self, f):
        return self.track[self._i(f)]

    def __len__(self):
        return self.n


class DifforumRestyle:
    """Give a video-model render the Deforum / AnimateDiff / Disco look.

    MiniMax H3 or LTX supplies the motion; an image model (SDXL, SD1.5, Flux,
    turbo distills) re-paints every frame in your look, with the previous
    stylized frame carried along the video's optical flow and mixed in. That
    feedback is what made Deforum morph and AnimateDiff boil - here it rides on
    the video model's motion instead of a synthetic camera.

    Styles: clean restyle (steady), animatediff boil (texture re-rolls each
    frame), deforum morph (strong feedback smear), disco flicker (high denoise,
    per-frame seed, loose colour). `custom` uses feedback / seed_mode /
    color_hold as set. Connect a Director to use its scene prompts and energy.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        samplers, schedulers = _samplers()
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Frames from H3 / LTX / any video (Get Video Components)."}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING", {"tooltip": "The look, e.g. 'oil painting, thick brush strokes'."}),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "style": (list(RESTYLE_STYLES) + ["custom"], {"default": "deforum morph"}),
                "denoise": ("FLOAT", {"default": 0.45, "min": 0.05, "max": 1.0, "step": 0.01,
                            "tooltip": "How much each frame is re-painted. 0.3 keeps the render, 0.6+ re-imagines it."}),
                "steps": ("INT", {"default": 12, "min": 1, "max": 100,
                          "tooltip": "Full-denoise steps; each frame runs steps x denoise."}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "normal"}),
                "cadence": ("INT", {"default": 1, "min": 1, "max": 8,
                            "tooltip": "Re-paint every Nth frame; the rest follow the flow. 2 = twice as fast."}),
                "long_edge": ("INT", {"default": 1024, "min": 0, "max": 4096, "step": 64,
                              "tooltip": "Working size. 0 = the video's size. Upscale afterwards for 2K."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "direction": (DIRECTION, {"tooltip": "Director: scene prompts (with CLIP) and the energy curve."}),
                "prompts": (PROMPT,),
                "feedback": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 0.95, "step": 0.05,
                             "tooltip": "custom: share of the previous stylized frame in the next one."}),
                "seed_mode": (["fixed", "per frame"], {"default": "fixed",
                              "tooltip": "custom: per frame = texture re-rolls every frame (boil / flicker)."}),
                "color_hold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "custom: how much each frame keeps the source colours."}),
                "follow_energy": ("BOOLEAN", {"default": True,
                                  "tooltip": "With a Director: denoise follows its energy curve (0.5 = as set)."}),
                "flow_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.25}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "report")
    FUNCTION = "run"
    CATEGORY = CAT_RENDER

    def run(self, video, model, positive, negative, vae, style, denoise, steps, cfg, sampler_name,
            scheduler, cadence, long_edge, seed, direction=None, prompts=None, feedback=0.35,
            seed_mode="fixed", color_hold=0.5, follow_energy=True, flow_scale=0.5):
        from ..core.color import match_color
        from ..core.engine import resize_bhwc
        from .render import launch_warning, make_diffuser

        preset = RESTYLE_STYLES.get(style)
        if preset is None:
            preset = dict(feedback=float(feedback), per_frame_seed=seed_mode == "per frame",
                          hold=float(color_hold), denoise_mul=1.0, noise=0.0)
        n = int(video.shape[0])
        w, h = _size(int(video.shape[1]), int(video.shape[2]), int(long_edge))
        src = resize_bhwc(video[..., :3].float(), w, h)

        energy = None
        if direction is not None:
            if prompts is None and direction.prompts is not None and len(direction.prompts):
                prompts = direction.prompts
            if follow_energy and direction.strength is not None:
                nd = int(direction.params["max_frames"])
                energy = _Mapped([float(direction.strength.at(i)) for i in range(nd)], nd, n)
        if prompts is not None:
            prompts = _Mapped(prompts, len(prompts), n)

        base = float(denoise) * float(preset["denoise_mul"])

        def denoise_at(f):
            d = base * (energy[f] / 0.5 if energy is not None else 1.0)
            return max(0.02, min(1.0, d))

        diffuse = make_diffuser(model, vae, positive, negative, steps, cfg, sampler_name, scheduler,
                                denoise_at, lambda _f: float(cfg), prompts, int(seed),
                                "increment" if preset["per_frame_seed"] else "fixed",
                                step_scaling="by energy (fast)")
        try:
            import cv2  # noqa: F401

            from ..core.flow import _flow_cur_to_prev, _warp_by_flow
            has_flow = True
        except Exception:
            has_flow = False

        fb, hold, noise = float(preset["feedback"]), float(preset["hold"]), float(preset["noise"])
        cad = max(1, int(cadence))
        gen = torch.Generator().manual_seed(int(seed))
        pbar = progress_bar(n)
        out, prev_out, prev_g, keys = [], None, None, 0
        t0 = time.perf_counter()
        for f in range(n):
            check_interrupt()
            cur = src[f:f + 1]
            g = _gray_u8(cur) if has_flow else None
            aligned = None
            if prev_out is not None:
                aligned = _warp_by_flow(prev_out, _flow_cur_to_prev(g, prev_g, float(flow_scale))) \
                    if has_flow else prev_out
            if f % cad == 0 or f == n - 1 or aligned is None:
                init = cur if aligned is None or fb <= 0 else cur * (1.0 - fb) + aligned * fb
                if noise > 0:
                    init = (init + noise * torch.randn(init.shape, generator=gen)).clamp(0, 1)
                img = diffuse(init, f)[:1, ..., :3].float().cpu()
                img = resize_bhwc(img, w, h)
                if hold > 0:
                    img = match_color(img, cur, strength=hold)
                keys += 1
            else:
                img = aligned
            img = img.clamp(0, 1)
            out.append(img)
            prev_out, prev_g = img, g
            pbar.update_absolute(f + 1, n)
        dt = time.perf_counter() - t0
        report = (f"restyle '{style}': {n} frames {w}x{h}, {keys} painted, feedback {fb:.2f}, "
                  f"hold {hold:.2f}, {'per-frame' if preset['per_frame_seed'] else 'fixed'} seed; "
                  f"{dt:.0f}s ({dt / max(1, n):.2f}s/frame)"
                  + ("" if has_flow else "\n  ! OpenCV missing: feedback is not flow-aligned"))
        slow = launch_warning()
        if slow:
            report += f"\n  ! {slow}"
        return (torch.cat(out, dim=0), report)


# ---------------------------------------------------------------------------
# Upscale
# ---------------------------------------------------------------------------

UPSCALE_TARGETS = {
    "2K (2048 long edge)": ("edge", 2048),
    "1080p (1920 long edge)": ("edge", 1920),
    "1440p (2560 long edge)": ("edge", 2560),
    "4K (3840 long edge)": ("edge", 3840),
    "x1.5": ("mul", 1.5),
    "x2": ("mul", 2.0),
    "x4": ("mul", 4.0),
}


def target_size(h: int, w: int, target: str) -> tuple[int, int]:
    kind, v = UPSCALE_TARGETS[target]
    s = v / max(h, w) if kind == "edge" else v
    return max(2, round(w * s / 2) * 2), max(2, round(h * s / 2) * 2)


def _resize(x: torch.Tensor, w: int, h: int, method: str) -> torch.Tensor:
    if x.shape[2] == w and x.shape[1] == h:
        return x
    chw = x.movedim(-1, 1)
    try:
        import comfy.utils
        out = comfy.utils.common_upscale(chw, w, h, method, "disabled")
    except Exception:
        out = torch.nn.functional.interpolate(chw, size=(h, w), mode="bicubic", align_corners=False)
    return out.movedim(1, -1).clamp(0, 1)


def _unsharp(x: torch.Tensor, amount: float) -> torch.Tensor:
    if amount <= 0:
        return x
    chw = x.movedim(-1, 1)
    k = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=chw.dtype, device=chw.device)
    k = (k[:, None] * k[None, :]) / 256.0
    k = k.expand(chw.shape[1], 1, 5, 5)
    blur = torch.nn.functional.conv2d(torch.nn.functional.pad(chw, (2, 2, 2, 2), mode="replicate"),
                                      k, groups=chw.shape[1])
    return (chw + amount * (chw - blur)).clamp(0, 1).movedim(1, -1)


class DifforumUpscale:
    """Finish at 2K (or 1080p / 1440p / 4K / xN) for delivery.

    With an upscale model (Load Upscale Model: 4x-UltraSharp, RealESRGAN,
    4x_foolhardy_Remacri...) each frame is upscaled by the model in chunks,
    then resized to the exact target; without one it is a clean Lanczos
    resize. Aspect is kept and sizes stay even for video codecs. Bypass the
    node (or its group) to deliver at render size.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "target": (list(UPSCALE_TARGETS), {"default": "2K (2048 long edge)"}),
                "method": (["lanczos", "bicubic", "bilinear", "area"], {"default": "lanczos"}),
                "sharpen": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.5, "step": 0.05}),
                "chunk": ("INT", {"default": 16, "min": 1, "max": 256,
                          "tooltip": "Frames per model pass; lower it if VRAM runs out."}),
            },
            "optional": {"upscale_model": ("UPSCALE_MODEL",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "info")
    FUNCTION = "run"
    CATEGORY = CAT_POST

    def run(self, frames, target, method, sharpen, chunk, upscale_model=None):
        n, h, w = int(frames.shape[0]), int(frames.shape[1]), int(frames.shape[2])
        tw, th = target_size(h, w, target)
        step = max(1, int(chunk))
        pbar = progress_bar(math.ceil(n / step))
        out = []
        for k, i in enumerate(range(0, n, step)):
            check_interrupt()
            x = frames[i:i + step, ..., :3]
            if upscale_model is not None:
                x = call_comfy_node("ImageUpscaleWithModel", upscale_model=upscale_model, image=x)[0]
            out.append(_unsharp(_resize(x.float().cpu(), tw, th, method), float(sharpen)))
            pbar.update_absolute(k + 1)
        info = f"{n} frames {w}x{h} -> {tw}x{th} ({'model + ' if upscale_model is not None else ''}{method})"
        return (torch.cat(out, dim=0), info)


# ---------------------------------------------------------------------------
# Workflow Switches (UI only)
# ---------------------------------------------------------------------------

class DifforumSwitches:
    """Control panel for the workflow: one switch per group.

    Turning a group off mutes it when it holds an output (Previz, Render) and
    bypasses it when it sits in the middle of the chain (Live Preview, Fill
    Reveal, Restyle, Look Mix, Upscale), so the rest of the graph still runs.
    ⌖ jumps the canvas to the group. The node runs nothing itself.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = CAT_SETUP

    def run(self):
        return ()


NODE_CLASS_MAPPINGS = {
    "Difforum_Restyle": DifforumRestyle,
    "Difforum_Upscale": DifforumUpscale,
    "Difforum_Switches": DifforumSwitches,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_Restyle": "Difforum · Restyle (Deforum / AnimateDiff look)",
    "Difforum_Upscale": "Difforum · Upscale (2K / 4K)",
    "Difforum_Switches": "Difforum · Workflow Switches",
}
