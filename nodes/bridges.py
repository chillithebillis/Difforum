"""Bridges from Difforum direction to modern video models.

Difforum stays the director (camera, timing, prompts); the video model does
the temporally coherent rendering. Nothing here imports another pack's
internals: outputs are plain IMAGE / MASK / INT / STRING, and the LTX node
drives ComfyUI's own core `LTXVAddGuide`.
"""

from __future__ import annotations

import torch

from ..core.direction import blocks_in_range, describe_blocks, describe_timed, describe_track
from ..core.engine import EngineConfig, FeedbackEngine, resize_bhwc
from ..core.h3prompt import MODES as H3_MODES
from ..core.h3prompt import h3_prompt
from ._common import CAMERA, CAT_BRIDGE, PARAMS, call_comfy_node, progress_bar
from .direction import DIRECTION
from .setup import TARGETS, snap, snap_frames

GRIDS = {
    "LTX-2 / 2.5 (8k+1)": (8, 1),
    "MiniMax H3 (17k+5)": (17, 5),
    "Wan 2.x (4k+1)": (4, 1),
    "any": (1, 0),
}


def _params_camera(direction, params, camera):
    if direction is not None:
        params = params or direction.params
        camera = camera or direction.camera
    if params is None or camera is None:
        raise ValueError("Connect a Director `direction`, or params + camera.")
    return params, camera


def _parse_indices(text: str, n: int) -> list[int]:
    out = []
    for tok in str(text or "").replace(";", ",").split(","):
        tok = tok.strip()
        if tok.lstrip("-").isdigit():
            i = int(tok)
            out.append(i + n if i < 0 else i)
    return sorted({i for i in out if 0 <= i < n})


class DifforumGuideFrames:
    """Warp one anchor image along the Director's camera path.

    Produces a guide video whose motion is exactly the camera you drew, for
    video models that take control/guide frames (LTX guides, Wan VACE, H3
    first/last frame). Revealed areas are filled with neutral gray and marked
    in the mask, using the convention the target expects.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anchor_image": ("IMAGE",),
                "mask_convention": (["1 = generate (VACE / LTX / H3)", "1 = keep"],
                                    {"default": "1 = generate (VACE / LTX / H3)"}),
                "hole_fill": (["gray", "stretch edge", "black"], {"default": "gray"}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "params": (PARAMS,),
                "camera": (CAMERA,),
                "depth": ("IMAGE", {"tooltip": "Depth of the anchor, for real parallax in 3d mode."}),
                "translation_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("guide_frames", "masks", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, anchor_image, mask_convention, hole_fill, direction=None, params=None,
            camera=None, depth=None, translation_scale=1.0):
        params, camera = _params_camera(direction, params, camera)
        n = min(int(params["max_frames"]), len(camera.deltas))
        w, h = int(params["width"]), int(params["height"])
        border = "border" if hole_fill == "stretch edge" else "zeros"
        engine = FeedbackEngine(camera, EngineConfig(width=w, height=h, border=border,
                                translation_scale=float(translation_scale)), depth=depth)
        pbar = progress_bar(n)
        imgs, masks = [], []
        for f, (img, valid) in enumerate(engine.iter_from_anchor(anchor_image, n)):
            if hole_fill == "gray":
                img = img * valid + 0.5 * (1.0 - valid)
            imgs.append(img)
            masks.append(valid[..., 0])
            pbar.update_absolute(f + 1, n)
        mask = torch.cat(masks, dim=0)
        if mask_convention.startswith("1 = generate"):
            mask = 1.0 - mask
        info = (f"{n} guide frames {w}x{h} ({engine.report.mode}); "
                f"revealed area avg {float((1 - torch.cat(masks)).mean()) * 100:.1f}%")
        return (torch.cat(imgs, dim=0), mask, info)


class DifforumKeyframes:
    """Pick keyframes out of any frame batch (a Storyboard, a Feedback render,
    guide frames) for a video model, on that model's frame grid.

    Outputs the keyframes, their indices, a full-length *sparse* batch (black
    except at keyframes - what `LTXVAddGuidesFromBatch` expects), the first and
    last frame (for first-last-frame models) and the snapped clip length.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "grid": (list(GRIDS), {"default": "LTX-2 / 2.5 (8k+1)"}),
                "every_seconds": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 60.0, "step": 0.05,
                                  "tooltip": "Spacing between keyframes. 0 = first and last only."}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0}),
            },
            "optional": {
                "indices": ("STRING", {"default": "", "tooltip": "Explicit frame list, e.g. 0, 48, 96, -1 (overrides spacing)."}),
                "masks": ("MASK", {"tooltip": "Guide Frames masks, so Fill Reveal can repaint the keyframes."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "IMAGE", "INT", "MASK", "STRING")
    RETURN_NAMES = ("keyframes", "indices", "sparse_batch", "first_frame", "last_frame", "length",
                    "key_masks", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, frames, grid, every_seconds, fps, indices="", masks=None):
        rule = GRIDS.get(grid, (1, 0))
        total = int(frames.shape[0])
        length = snap_frames(total, rule)
        if length > total:                       # never ask for frames we do not have
            a, b = rule
            length = max(b if b else 1, total if a <= 1 else ((total - b) // a) * a + b)
        frames = frames[:length]

        if indices.strip():
            idx = _parse_indices(indices, length)
        else:
            step = int(round(float(every_seconds) * float(fps)))
            if rule[0] > 1 and step > 0:          # align to the model's latent stride
                step = max(rule[0], int(round(step / rule[0])) * rule[0])
            idx = list(range(0, length, step)) if step > 0 else [0]
            if idx[-1] != length - 1:
                idx.append(length - 1)
        idx = idx or [0]

        keys = frames[idx]
        sparse = torch.zeros_like(frames)
        sparse[idx] = frames[idx]
        info = (f"{len(idx)} keyframes on {grid}, length {length} "
                f"({length / fps:.2f}s): {', '.join(map(str, idx))}")
        if masks is not None:
            key_masks = masks[[min(i, masks.shape[0] - 1) for i in idx]]
        else:
            key_masks = torch.zeros(keys.shape[:3])
        return (keys, ",".join(map(str, idx)), sparse, frames[:1], frames[-1:], int(length),
                key_masks, info)


class DifforumCameraPrompt:
    """The camera, in words - for prompt-driven video models (MiniMax H3,
    LTX, Seedance, Veo...). Uses the Director's blocks when available, or
    reads any camera track and names its moves."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": (["sentence", "timed lines", "prompt suffix", "H3 structured"], {"default": "sentence",
                           "tooltip": "sentence: one line of camera direction (+ look) to append to any prompt. "
                                      "timed lines: [0.0s-2.5s] camera ... per block. prompt suffix: 'Camera: ...'. "
                                      "H3 structured: the whole Director timeline (scenes, camera, keys, look) "
                                      "written in MiniMax H3's native prompt format."}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "params": (PARAMS,),
                "camera": (CAMERA,),
                "prefix": ("STRING", {"default": "", "multiline": True,
                           "tooltip": "Your shot description; the camera sentence is appended."}),
                "include_look": ("BOOLEAN", {"default": True,
                                 "tooltip": "Append the Director's look sentence (deforum morph, stop-motion...)."}),
                "h3_mode": (list(H3_MODES), {"default": H3_MODES[3],
                            "tooltip": "H3 structured only. Which H3 task the prompt is for: sets the alignment "
                                       "line (I2VA / FL2VA) or the six full-reference sections (ref2va with "
                                       "H3 Guides, <Picture 1> as the first frame)."}),
                "soundscape": ("STRING", {"default": "", "multiline": True,
                               "tooltip": "H3 structured: overall_soundscape - ambience and physical sounds only "
                                          "(wind, footsteps, rain). Empty = soft natural ambience."}),
                "music": ("STRING", {"default": "", "multiline": True,
                          "tooltip": "H3 structured: non_diegetic_music - instruments, tempo, dynamics, no mood "
                                     "words. Empty = N/A (no score)."}),
                "cuts": ("BOOLEAN", {"default": False,
                         "tooltip": "H3 structured: off = one continuous take (scene changes become "
                                    "transformations). On = every scene starts a new [Shot N] at its cut time."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, format, direction=None, params=None, camera=None, prefix="", include_look=True,
            h3_mode=H3_MODES[3], soundscape="", music="", cuts=False):
        if format == "H3 structured":
            if direction is None:
                raise ValueError("H3 structured needs the Director's direction wire.")
            return (h3_prompt(direction.direction, direction.look_prompt if include_look else "", h3_mode,
                              prefix, soundscape, music, cuts=bool(cuts)),)
        if direction is not None and camera is None:
            d = direction.direction
            sentence = d.camera_text
            timed = describe_timed(d.camera_blocks, d.frames, d.fps)
        else:
            params, camera = _params_camera(direction, params, camera)
            sentence, timed = describe_track(camera, float(params["fps"]))
        if format == "timed lines":
            body = timed
        elif format == "prompt suffix":
            body = "Camera: " + sentence[len("The camera "):] if sentence.startswith("The camera ") else sentence
        else:
            body = sentence
        if include_look and direction is not None and format != "timed lines":
            body = f"{body} {direction.look_prompt}"
        text = (prefix.strip() + " " + body).strip() if prefix.strip() else body
        return (text,)


class DifforumLTXGuides:
    """Put Difforum keyframes into an LTX-2 / 2.5 latent as guides.

    Wraps ComfyUI's core `LTXVAddGuide` once per keyframe, so the camera you
    drew becomes the LTX shot. Feed keyframes + indices from the Keyframes
    node (from a Storyboard, Guide Frames or a Feedback render), and an empty
    LTX latent whose length matches (Setup target = LTX).
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "keyframes": ("IMAGE",),
                "indices": ("STRING", {"default": "0", "forceInput": True}),
                "strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01,
                             "tooltip": "Guide strength for in-between keyframes."}),
                "first_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "Frame 0 usually locks the look: keep it high."}),
                "last_strength": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, positive, negative, vae, latent, keyframes, indices, strength,
            first_strength, last_strength):
        idx = _parse_indices(indices, 10**9)
        if len(idx) != keyframes.shape[0]:
            raise ValueError(f"{keyframes.shape[0]} keyframes but {len(idx)} indices ({indices!r})")
        lines = []
        for k, (fi, img) in enumerate(zip(idx, keyframes)):
            s = first_strength if k == 0 else (last_strength if k == len(idx) - 1 else strength)
            if s <= 0:
                continue
            positive, negative, latent = call_comfy_node(
                "LTXVAddGuide", positive=positive, negative=negative, vae=vae, latent=latent,
                image=img.unsqueeze(0), frame_idx=int(fi), strength=float(s))[:3]
            lines.append(f"  guide @ {fi} strength {s:g}")
        return (positive, negative, latent, "LTX guides:\n" + "\n".join(lines))


class DifforumFillReveal:
    """Complete what the camera reveals, with an image model (AI hole fill).

    Guide Frames, H3 Shot and Keyframes mark the area the camera uncovers
    (outside the original picture) in a mask. This node repaints only that
    area by inpainting, so edges become new, coherent scenery instead of
    gray or stretched pixels. The known pixels are kept exactly.

    Works with any image model through ComfyUI's core `InpaintModelConditioning`:
    a dedicated inpaint model (SDXL inpainting, Flux Fill) gives the cleanest
    seams, a regular checkpoint works too. Run it only on the frames a video
    model will see (first / last frame, keyframes) - it is one diffusion per frame.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import comfy.samplers
            samplers, schedulers = comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS
        except Exception:
            samplers, schedulers = ["euler"], ["normal"]
        return {
            "required": {
                "images": ("IMAGE",),
                "masks": ("MASK", {"tooltip": "1 = area to fill (Guide Frames default convention)."}),
                "model": ("MODEL",),
                "positive": ("CONDITIONING", {"tooltip": "Describe the scene so the fill matches it."}),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "frames": (["all", "first", "last", "first + last"], {"default": "all"}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "euler" if "euler" in samplers else samplers[0]}),
                "scheduler": (schedulers, {"default": "normal" if "normal" in schedulers else schedulers[0]}),
                "grow": ("INT", {"default": 16, "min": 0, "max": 256,
                         "tooltip": "Pixels the mask is grown into the known image, to hide the seam."}),
                "feather": ("INT", {"default": 12, "min": 0, "max": 256}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    @staticmethod
    def _prepare_mask(m: torch.Tensor, h: int, w: int, grow: int, feather: int):
        """-> (area to diffuse, blend alpha). The revealed area itself is always
        fully replaced; grow + feather only soften the seam into known pixels."""
        m = m.float()
        if m.shape[-2:] != (h, w):
            m = torch.nn.functional.interpolate(m[None, None], size=(h, w), mode="bilinear")[0, 0]
        core = (m > 0.5).float()[None, None]
        hard = core
        if grow > 0:
            hard = torch.nn.functional.max_pool2d(core, 2 * grow + 1, stride=1, padding=grow)
        soft = hard
        if feather > 0:
            soft = torch.nn.functional.avg_pool2d(hard, 2 * feather + 1, stride=1, padding=feather,
                                                  count_include_pad=False)
            soft = torch.maximum(soft * hard, core)
        return hard[0, 0], soft[0, 0].clamp(0, 1)

    def run(self, images, masks, model, positive, negative, vae, frames, steps, cfg,
            sampler_name, scheduler, grow, feather, seed):
        from nodes import common_ksampler

        n = int(images.shape[0])
        pick = {"all": range(n), "first": [0], "last": [n - 1],
                "first + last": sorted({0, n - 1})}[frames]
        out = images.clone()
        lines = []
        for i in pick:
            h, w = int(images.shape[1]), int(images.shape[2])
            hard, soft = self._prepare_mask(masks[min(i, masks.shape[0] - 1)], h, w, int(grow), int(feather))
            if float(hard.sum()) < 1:
                lines.append(f"  frame {i}: nothing revealed, kept")
                continue
            img = images[i:i + 1, ..., :3]
            pos, neg, latent = call_comfy_node("InpaintModelConditioning", positive=positive,
                                               negative=negative, pixels=img, vae=vae,
                                               mask=hard.unsqueeze(0), noise_mask=True)
            sampled = common_ksampler(model, int(seed) + i, int(steps), float(cfg), sampler_name,
                                      scheduler, pos, neg, latent, denoise=1.0)[0]
            dec = vae.decode(sampled["samples"])
            if dec.dim() == 5:
                dec = dec.reshape(-1, *dec.shape[-3:])
            dec = resize_bhwc(dec[:1], w, h).to(img.dtype)
            a = soft.to(img.dtype)[None, ..., None]
            out[i:i + 1, ..., :3] = torch.where(a > 0, img * (1.0 - a) + dec * a, img)
            lines.append(f"  frame {i}: filled {float(hard.mean()) * 100:.1f}% of the image")
        return (out, "Fill Reveal:\n" + "\n".join(lines))


class DifforumH3Guides:
    """Anchor Difforum keyframes inside a MiniMax H3 generation.

    Wraps ComfyUI's core `MiniMaxH3AddGuide` once per keyframe, so the camera
    you drew becomes the H3 shot: feed keyframes + indices from the Keyframes
    node (grid H3) and the positive + AV latent from `MiniMax H3 Reference to
    Video` (or `Image to Video`). Optionally anchor a soundtrack at frame 0, so
    an audio-reactive direction and H3's own audio stay in sync.

    H3 is trained with a few guides per clip: `max_guides` keeps the first,
    the last and evenly spaced ones in between.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT", {"tooltip": "The MiniMax H3 AV latent."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "keyframes": ("IMAGE",),
                "indices": ("STRING", {"default": "0", "forceInput": True}),
                "max_guides": ("INT", {"default": 4, "min": 1, "max": 16}),
                "skip_first": ("BOOLEAN", {"default": False,
                               "tooltip": "Enable when frame 0 is already set (e.g. Image to Video first_frame)."}),
            },
            "optional": {
                "audio_vae": ("VAE", {"tooltip": "MiniMax H3 audio VAE, needed with audio."}),
                "audio": ("AUDIO", {"tooltip": "Soundtrack anchored at frame 0."}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, positive, latent, vae, keyframes, indices, max_guides, skip_first,
            audio_vae=None, audio=None):
        idx = _parse_indices(indices, 10**9)
        if len(idx) != keyframes.shape[0]:
            raise ValueError(f"{keyframes.shape[0]} keyframes but {len(idx)} indices ({indices!r})")
        pairs = list(zip(idx, keyframes))
        if skip_first and pairs and pairs[0][0] == 0:
            pairs = pairs[1:]
        if len(pairs) > max_guides:        # keep the keyframes nearest to evenly spaced times
            f0, f1 = pairs[0][0], pairs[-1][0]
            keep = set()
            for i in range(max_guides):
                t = f0 + (f1 - f0) * i / max(1, max_guides - 1)
                free = [k for k in range(len(pairs)) if k not in keep]
                keep.add(min(free, key=lambda k: abs(pairs[k][0] - t)))
            pairs = [pairs[k] for k in sorted(keep)]
        lines = []
        for k, (fi, img) in enumerate(pairs):
            kw = {"positive": positive, "latent": latent, "frame_idx": int(fi), "vae": vae,
                  "image": img.unsqueeze(0)}
            if k == 0 and audio is not None and fi == 0:
                kw.update(audio=audio, audio_vae=audio_vae)
            positive = call_comfy_node("MiniMaxH3AddGuide", **kw)[0]
            _stash_guide_pixels(positive, img.unsqueeze(0))
            lines.append(f"  guide @ frame {fi}" + (" + audio" if "audio" in kw else ""))
        if audio is not None and not any("audio" in x for x in lines):
            positive = call_comfy_node("MiniMaxH3AddGuide", positive=positive, latent=latent,
                                       frame_idx=0, audio=audio, audio_vae=audio_vae)[0]
            lines.append("  audio @ frame 0")
        return (positive, "H3 guides:\n" + "\n".join(lines))


def _stash_guide_pixels(positive, image):
    """Keep the full-size pixels of the guide just added, so H3 Refine Guides can
    re-encode it at a two-stage refine size instead of stretching a small latent."""
    for item in positive:
        extra = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else None
        kfs = extra.get("minimax_keyframes") if isinstance(extra, dict) else None
        if kfs and kfs[-1].get("latent") is not None:
            kfs[-1]["difforum_image"] = image.detach().cpu()


def _h3_video_latent(latent):
    samples = latent["samples"]
    if getattr(samples, "is_nested", False):
        return samples.tensors[0]
    return samples


def _decode_frames(vae, z):
    img = vae.decode(z)
    return img.reshape(-1, *img.shape[-3:])


class DifforumH3RefineGuides:
    """Make H3 guides fit a two-stage refine.

    MiniMax H3 guides (first / last frame, Add Guide, Difforum H3 Guides) are
    encoded at the size of the first render and shared with the target's
    spatial grid, so the refine pass after a latent upscale fails with a
    *shape mismatch* if it reuses the same conditioning. This node re-encodes
    every guide at the upscaled latent's size: from the original full-size
    pixels when Difforum H3 Guides added it, otherwise by decoding the small
    guide, resizing it and encoding it again. `drop guides` removes the video
    guides instead and lets the refine follow the upscaled render alone.
    References (ref2va pictures) keep their own size and are left untouched.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT", {"tooltip": "The upscaled MiniMax H3 AV latent the refine samples."}),
                "vae": ("VAE", {"tooltip": "MiniMax H3 video VAE."}),
                "mode": (["re-encode", "drop guides"], {"default": "re-encode"}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING")
    RETURN_NAMES = ("positive", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, positive, latent, vae, mode):
        target = _h3_video_latent(latent)
        lh, lw = int(target.shape[-2]), int(target.shape[-1])
        width, height = lw * 16, lh * 16
        cache, out, lines = {}, [], []
        for emb, extra in positive:
            extra = dict(extra)
            kfs = extra.get("minimax_keyframes")
            if kfs:
                new = []
                for kf in kfs:
                    z = kf.get("latent")
                    if z is None or tuple(z.shape[-2:]) == (lh, lw):
                        new.append(kf)
                        continue
                    kf = dict(kf)
                    fi = kf.get("resolved_frame_index", 0)
                    if mode == "drop guides":
                        kf.pop("latent")
                        kf.pop("difforum_image", None)
                        if kf.get("audio_latent") is not None:
                            new.append(kf)
                        lines.append(f"  guide @ frame {fi}: dropped")
                        continue
                    key = id(z)
                    if key not in cache:
                        src = kf.get("difforum_image")
                        how = "from full-size pixels"
                        if src is None:
                            src, how = _decode_frames(vae, z), "decoded and resized"
                        frames = resize_bhwc(src[..., :3].float(), width, height)
                        cache[key] = (vae.encode(frames), how)
                    kf["latent"], how = cache[key]
                    lines.append(f"  guide @ frame {fi}: {tuple(z.shape[-2:])} -> {(lh, lw)} latent, {how}")
                    new.append(kf)
                extra["minimax_keyframes"] = new
            out.append([emb, extra])
        info = f"H3 refine guides at {width}x{height}:\n" + ("\n".join(lines) if lines else "  nothing to change")
        return (out, info)


H3_LENGTHS = {"auto (from frames)": 0, "124 (~5s)": 124, "243 (~10s)": 243,
              "362 (~15s)": 362, "481 (~20s)": 481}


class DifforumH3Shot:
    """Everything a MiniMax H3 first-last-frame shot needs, from Difforum.

    Give it the frames of a Storyboard / Feedback render / Guide Frames and it
    returns the first and last frame of the chosen segment, the length on the
    H3 17k+5 grid, the size on the 32 px grid, and the camera move of that
    segment in words for the prompt. Clips longer than one H3 generation are
    split into segments: render segment 0, then feed its last frame as the
    next segment's first frame.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "segment_length": (list(H3_LENGTHS), {"default": "auto (from frames)"}),
                "segment": ("INT", {"default": 0, "min": 0, "max": 999}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "params": (PARAMS,),
                "camera": (CAMERA,),
                "shot_description": ("STRING", {"default": "", "multiline": True}),
                "masks": ("MASK", {"tooltip": "Guide Frames masks: returns the last frame's revealed area "
                                              "for Fill Reveal."}),
                "include_look": ("BOOLEAN", {"default": True}),
                "prompt_style": (["H3 structured", "sentence"], {"default": "H3 structured",
                                 "tooltip": "H3 structured: this segment's scenes, camera, keys and look in "
                                            "MiniMax H3's native FL2VA format (alignment line + fields). "
                                            "sentence: description + camera sentence + look, as plain text."}),
                "soundscape": ("STRING", {"default": "", "multiline": True,
                               "tooltip": "overall_soundscape: ambience and physical sounds only."}),
                "music": ("STRING", {"default": "", "multiline": True,
                          "tooltip": "non_diegetic_music: instruments, tempo, dynamics. Empty = N/A."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "STRING", "INT", "MASK", "STRING")
    RETURN_NAMES = ("first_frame", "last_frame", "length", "width", "height", "prompt",
                    "segments", "last_mask", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, frames, segment_length, segment, direction=None, params=None, camera=None,
            shot_description="", masks=None, include_look=True, prompt_style="H3 structured",
            soundscape="", music=""):
        total = int(frames.shape[0])
        rule = TARGETS["MiniMax H3"][0]
        seg_len = H3_LENGTHS.get(segment_length, 0) or min(481, snap_frames(total, rule))
        seg_len = min(seg_len, max(5, total))
        seg_len = max(5, ((seg_len - 5) // 17) * 17 + 5)
        stride = seg_len - 1                        # segments share their boundary frame
        segments = max(1, -(-(total - 1) // stride))
        s = min(int(segment), segments - 1)
        start = s * stride
        end = min(total - 1, start + seg_len - 1)

        h, w = int(frames.shape[1]), int(frames.shape[2])
        w32, h32 = snap(w, 32), snap(h, 32)
        first = resize_bhwc(frames[start:start + 1], w32, h32)
        last = resize_bhwc(frames[end:end + 1], w32, h32)

        cam_text = ""
        if direction is not None:
            d = direction.direction
            cam_text = describe_blocks(blocks_in_range(d.camera_blocks, start, end + 1),
                                       end + 1 - start, d.fps)
        elif camera is not None and params is not None:
            from ..core.camera import CameraTrack
            sub = CameraTrack(deltas=camera.deltas[start:end + 1], poses=camera.poses[start:end + 1],
                              zoom=camera.zoom[start:end + 1], fov=camera.fov[start:end + 1],
                              mode=camera.mode)
            cam_text = describe_track(sub, float(params["fps"]))[0]
        look = direction.look_prompt if (include_look and direction is not None) else ""
        if prompt_style == "H3 structured" and direction is not None:
            prompt = h3_prompt(direction.direction, look, H3_MODES[2], shot_description, soundscape, music,
                               start=start, end=end + 1)
        else:
            prompt = " ".join(x for x in (shot_description.strip(), cam_text, look) if x)
        if masks is not None:
            m = masks[min(end, masks.shape[0] - 1)].unsqueeze(0).unsqueeze(-1)
            last_mask = resize_bhwc(m, w32, h32)[..., 0]
        else:
            last_mask = torch.zeros((1, h32, w32))
        info = (f"H3 segment {s + 1}/{segments}: frames {start}-{end} "
                f"(length {seg_len}, {seg_len / 24:.2f}s @ 24fps), {w32}x{h32}\n"
                f"  mode fl2va - wire first/last into MiniMax H3 Image to Video")
        return (first, last, int(seg_len), int(w32), int(h32), prompt, int(segments), last_mask, info)


NODE_CLASS_MAPPINGS = {
    "Difforum_GuideFrames": DifforumGuideFrames,
    "Difforum_Keyframes": DifforumKeyframes,
    "Difforum_CameraPrompt": DifforumCameraPrompt,
    "Difforum_LTXGuides": DifforumLTXGuides,
    "Difforum_H3Shot": DifforumH3Shot,
    "Difforum_H3Guides": DifforumH3Guides,
    "Difforum_H3RefineGuides": DifforumH3RefineGuides,
    "Difforum_FillReveal": DifforumFillReveal,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_GuideFrames": "Difforum · Guide Frames",
    "Difforum_Keyframes": "Difforum · Keyframes",
    "Difforum_CameraPrompt": "Difforum · Camera → Prompt",
    "Difforum_LTXGuides": "Difforum · LTX Guides",
    "Difforum_H3Shot": "Difforum · H3 Shot",
    "Difforum_H3Guides": "Difforum · H3 Guides",
    "Difforum_H3RefineGuides": "Difforum · H3 Refine Guides",
    "Difforum_FillReveal": "Difforum · Fill Reveal (AI)",
}
