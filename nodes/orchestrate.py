"""Orchestration from outside the editor: shot scripts (typed, from a file or
from an LLM node) that write the Director timeline, and keyframe assets loaded
from a folder so styled pictures made elsewhere drive the video model directly,
without rendering a look pass first."""

from __future__ import annotations

import hashlib
import json
import os
import re

import numpy as np
import torch

from ..core.camera_presets import CAMERA_PRESETS
from ..core.script import LLM_GUIDE, script_to_timeline
from ._common import CAT_DIRECT, PARAMS
from .direction import DIRECTION, _parse_times

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

_EXAMPLE = """# TIME | MOOD | CAMERA | PROMPT     (one beat per line, # comments)
0s   | calm    | dolly_in slow small | misty ancient forest at dawn, light shafts through the canopy
2s   | build   | orbit_left          | glowing moss and roots in the foreground
3.5s | key: the light breaks
4s   | resolve | crane_up slow       | the canopy opens to a pale sky"""


def _input_dir():
    import folder_paths
    return folder_paths.get_input_directory()


def _safe(name):
    from ..core.video import safe_input_path
    return safe_input_path(_input_dir(), name)


EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "example_scripts")


def _script_path(name):
    """A file in ComfyUI/input, or `examples/<name>` for the scripts shipped with the pack."""
    name = str(name).strip().replace("\\", "/")
    if name.startswith("examples/"):
        from ..core.video import safe_input_path
        return safe_input_path(EXAMPLES_DIR, name[len("examples/"):])
    return _safe(name)


def _natural(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ---------------------------------------------------------------------------
# Shot Script
# ---------------------------------------------------------------------------

class DifforumShotScript:
    """Write the Director timeline from text instead of drawing it.

    One beat per line: `TIME | MOOD | CAMERA | PROMPT` (`0s | calm | dolly_in
    slow | misty forest`), `4s | key: the light breaks` for a key marker, or
    just prompts without times (spread evenly). A CSV with a header row
    (`time,mood,camera,prompt,key,energy`) or a Director JSON works too.

    The text can come from the box, from a `.txt` / `.csv` / `.json` file in
    ComfyUI's input folder (re-read when it changes; `examples/01_infinite_zoom.txt`
    and the other shipped scripts work too), or from any node that
    outputs a STRING - an LLM node, a text loader, a spreadsheet export - into
    `script_in`. Wire `timeline` into the Director's `timeline_in`.
    `llm_instructions` is a ready prompt that asks an LLM for this format.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script": ("STRING", {"multiline": True, "default": _EXAMPLE}),
                "file": ("STRING", {"default": ""}),
            },
            "optional": {
                "params": (PARAMS,),
                "script_in": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("timeline", "llm_instructions", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    @classmethod
    def IS_CHANGED(cls, script, file, params=None, script_in=None):
        if str(file).strip():
            try:
                return os.path.getmtime(_script_path(file))
            except Exception:
                return float("nan")
        return ""

    def run(self, script, file, params=None, script_in=None):
        fps = float(params["fps"]) if params else 24.0
        frames = int(params["max_frames"]) if params else None
        if script_in is not None and str(script_in).strip():
            text, source = str(script_in), "script_in"
        elif str(file).strip():
            path = _script_path(file)
            if not os.path.isfile(path):
                raise ValueError(f"Shot Script: {file!r} not found in ComfyUI/input "
                                 "(or use examples/01_infinite_zoom.txt ... for the shipped scripts)")
            with open(path, encoding="utf-8-sig") as fh:
                text, source = fh.read(), file.strip()
        else:
            text, source = script, "text box"
        tl, notes = script_to_timeline(text, fps, frames)
        secs = (frames or 120) / fps
        moves = ", ".join(sorted(m for m in CAMERA_PRESETS if m != "custom"))
        guide = LLM_GUIDE.format(seconds=secs, moves=moves)
        info = "\n".join([
            f"[Shot Script] from {source}: {len(tl['scenes'])} scenes, {len(tl['camera'])} camera blocks, "
            f"{len(tl['keys'])} keys",
            *(f"  {s['start'] / fps:6.2f}s  {s['mood']:<8} {s['prompt'][:70]}" for s in tl["scenes"]),
            *(f"  {c['start'] / fps:6.2f}s  camera {c['move']}" for c in tl["camera"]),
            *(f"  {k['start'] / fps:6.2f}s  key {k['label']}" for k in tl["keys"]),
            *(f"  ! {n}" for n in notes),
        ])
        return {"ui": {"text": [info]}, "result": (json.dumps(tl), guide, info)}


# ---------------------------------------------------------------------------
# Keyframe Assets
# ---------------------------------------------------------------------------

def _fit(img: torch.Tensor, w: int, h: int, mode: str):
    """[1,H,W,3] -> ([1,h,w,3], [1,h,w] mask of padding)."""
    import torch.nn.functional as F
    _b, ih, iw, _c = img.shape
    chw = img.permute(0, 3, 1, 2)
    mask = torch.zeros(1, h, w)
    if mode.startswith("stretch"):
        out = F.interpolate(chw, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        return out.permute(0, 2, 3, 1), mask
    s = max(w / iw, h / ih) if mode.startswith("cover") else min(w / iw, h / ih)
    nw, nh = max(1, round(iw * s)), max(1, round(ih * s))
    rs = F.interpolate(chw, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    if mode.startswith("cover"):
        y, x = (nh - h) // 2, (nw - w) // 2
        return rs[:, :, y:y + h, x:x + w].permute(0, 2, 3, 1), mask
    out = torch.full((1, 3, h, w), 0.5)                   # gray = "fill me" for Fill Reveal
    y, x = (h - nh) // 2, (w - nw) // 2
    out[:, :, y:y + nh, x:x + nw] = rs
    mask[:] = 1.0
    mask[:, y:y + nh, x:x + nw] = 0.0
    return out.permute(0, 2, 3, 1), mask


def _time_from_name(stem: str, fps: float):
    s = stem.lower()
    m = re.search(r"(?:^|[^a-z0-9.])(\d+(?:\.\d+)?)s(?:$|[^a-z0-9])", s)
    if m:
        return round(float(m.group(1)) * fps)
    m = re.search(r"(?:^|[^a-z0-9])f(\d+)(?:$|[^0-9])", s)
    if m:
        return int(m.group(1))
    m = re.match(r"^(\d+)(?:$|[^0-9.])", s)
    if m:
        return int(m.group(1))
    return None


class DifforumKeyframeAssets:
    """Your own styled pictures as the keyframes - no look pass to render.

    Point `folder` at a folder inside ComfyUI/input (stills made in any image
    model, Photoshop, a photo shoot...). Each picture lands on a moment:

    - **Director keys**: in order, on the Keys track markers.
    - **filename**: the name says when - `0s_wide.png`, `4.5s_close.png`,
      `f096.png`, or a leading frame number `0096_rise.png`.
    - **spread evenly**: first at frame 0, last at the end.

    `fit` matches the canvas: cover crops, contain pads with gray (and the
    `masks` output tells Fill Reveal what to paint), stretch distorts. Wire
    `keyframes` + `indices` into H3 Guides / LTX Guides / Animatic, and
    `first` / `last` into a first-last-frame model. An `images` batch from
    other nodes can replace the folder.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "difforum_keys"}),
                "timing": (["Director keys", "filename", "spread evenly"], {"default": "filename"}),
                "fit": (["cover (crop)", "contain (pad gray)", "stretch"], {"default": "cover (crop)"}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "params": (PARAMS,),
                "images": ("IMAGE",),
                "times": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("keyframes", "indices", "first", "last", "masks", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    @classmethod
    def IS_CHANGED(cls, folder, timing, fit, direction=None, params=None, images=None, times=""):
        try:
            d = _safe(folder.strip())
            sig = [(f, os.path.getmtime(os.path.join(d, f))) for f in sorted(os.listdir(d))]
            return hashlib.sha1(repr(sig).encode()).hexdigest()
        except Exception:
            return ""

    def _load_folder(self, folder):
        from PIL import Image, ImageOps
        d = _safe(folder.strip())
        if not os.path.isdir(d):
            raise ValueError(f"Keyframe Assets: folder {folder!r} not found in ComfyUI/input")
        names = sorted((f for f in os.listdir(d) if f.lower().endswith(IMAGE_EXT)), key=_natural)
        if not names:
            raise ValueError(f"Keyframe Assets: no images in input/{folder}")
        imgs = []
        for f in names:
            with Image.open(os.path.join(d, f)) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                imgs.append(torch.from_numpy(np.asarray(im).copy()).float()[None] / 255.0)
        return imgs, [os.path.splitext(f)[0] for f in names]

    def run(self, folder, timing, fit, direction=None, params=None, images=None, times=""):
        params = params or (direction.params if direction is not None else None)
        if params is None:
            raise ValueError("Keyframe Assets needs a direction or params.")
        n, fps = int(params["max_frames"]), float(params["fps"])
        w, h = int(params["width"]), int(params["height"])
        if images is not None:
            pics, names = [images[i:i + 1, ..., :3] for i in range(images.shape[0])], None
        else:
            pics, names = self._load_folder(folder)
        notes = []
        if str(times).strip():
            idx = _parse_times(times, fps, n)
        elif timing == "Director keys" and direction is not None and direction.direction.keys:
            idx = [k["start"] for k in direction.direction.keys]
        elif timing == "filename" and names:
            idx = [_time_from_name(s, fps) for s in names]
            if any(i is None for i in idx):
                bad = [s for s, i in zip(names, idx) if i is None]
                notes.append(f"  no time in {', '.join(bad[:4])}: spread evenly instead")
                idx = None
        else:
            idx = None
        if idx is None:
            m = len(pics)
            idx = [round(i * (n - 1) / max(1, m - 1)) for i in range(m)]
            if timing != "spread evenly":
                notes.append("  spread evenly")
        count = min(len(idx), len(pics))
        if count < max(len(idx), len(pics)):
            notes.append(f"  {len(idx)} times for {len(pics)} pictures: using {count}")
        pairs = sorted((max(0, min(n - 1, int(f))), i) for f, i in zip(idx[:count], range(count)))
        fitted = [_fit(pics[i].float(), w, h, fit) for _f, i in pairs]
        keys = torch.cat([f[0] for f in fitted]).clamp(0, 1)
        masks = torch.cat([f[1] for f in fitted])
        frames = [f for f, _i in pairs]
        label = (lambda i: names[i]) if names else (lambda i: f"image {i}")
        info = "\n".join([f"[Keyframe Assets] {count} keyframes at {w}x{h} ({fit})",
                          *(f"  {f / fps:6.2f}s  f{f:<5} {label(i)}" for f, i in pairs), *notes])
        return {"ui": {"text": [info]},
                "result": (keys, ",".join(map(str, frames)), keys[:1], keys[-1:], masks, info)}


# ---------------------------------------------------------------------------
# Scene Stills
# ---------------------------------------------------------------------------

class DifforumSceneStills:
    """One still per scene of the timeline: the storyboard, painted by the image model.

    Each scene prompt (with your `style` in front) becomes a picture at the
    frame where the scene starts. `continuity` paints every still over the one
    before it, so palette, light and layout carry from beat to beat instead of
    jumping; 0 makes each one from scratch. Feed `keyframes` + `indices` to H3
    Guides / LTX Guides and a video model animates between pictures that
    already tell the story: a handful of images instead of a look pass.

    Stills are made at `long_edge` (the image model's own size) in the canvas
    aspect, so they stay sharp when the video model renders smaller.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        from .render import _samplers
        samplers, schedulers = _samplers()
        return {
            "required": {
                "direction": (DIRECTION,),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "style": ("STRING", {"multiline": True, "default": ""}),
                "negative": ("STRING", {"multiline": True,
                             "default": "blurry, low quality, watermark, text, deformed"}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "dpmpp_2m" if "dpmpp_2m" in samplers else samplers[0]}),
                "scheduler": (schedulers, {"default": "karras" if "karras" in schedulers else schedulers[0]}),
                "seed": ("INT", {"default": 7, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "continuity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "long_edge": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
            },
            "optional": {"first_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("keyframes", "indices", "first", "last", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, direction, model, clip, vae, style, negative, steps, cfg, sampler_name, scheduler,
            seed, continuity, long_edge, first_image=None):
        from nodes import common_ksampler

        from ._common import check_interrupt, progress_bar
        from .render import resident_models
        params = direction.params
        n, fps = int(params["max_frames"]), float(params["fps"])
        cw, ch = int(params["width"]), int(params["height"])
        k = float(long_edge) / max(cw, ch)
        w, h = max(64, round(cw * k / 64) * 64), max(64, round(ch * k / 64) * 64)
        scenes = [s for s in direction.direction.scenes if s["start"] < n and s["prompt"].strip()]
        if not scenes:
            raise ValueError("Scene Stills: the timeline has no scene prompts.")

        def encode(text):
            return clip.encode_from_tokens_scheduled(clip.tokenize(text))

        neg = encode(negative)
        look = " ".join(str(style).split()).strip().rstrip(".,")
        cont = max(0.0, min(1.0, float(continuity)))
        pbar = progress_bar(len(scenes))
        stills, lines, prev = [], [], None
        for i, sc in enumerate(scenes):
            check_interrupt()
            if i == 0 and first_image is not None:
                img = _fit(first_image[:1, ..., :3].float(), w, h, "cover")[0]
                how = "given"
            else:
                pos = encode(f"{look}. {sc['prompt']}" if look else sc["prompt"])
                with resident_models():
                    if prev is not None and cont > 0.0:
                        latent, denoise = {"samples": vae.encode(prev)}, 1.0 - 0.45 * cont
                    else:
                        latent, denoise = {"samples": torch.zeros(1, 4, h // 8, w // 8)}, 1.0
                    out = common_ksampler(model, int(seed), int(steps), float(cfg), sampler_name, scheduler,
                                          pos, neg, latent, denoise=denoise)[0]
                    img = vae.decode(out["samples"])
                img = img.reshape(-1, *img.shape[-3:])[:1, ..., :3].float().cpu()
                how = "new" if denoise >= 1.0 else f"over the previous, denoise {denoise:.2f}"
            if tuple(img.shape[1:3]) != (h, w):          # VAEs that round the size differently
                img = _fit(img, w, h, "stretch")[0]
            prev = img
            stills.append(img)
            lines.append(f"  {sc['start'] / fps:6.2f}s  f{sc['start']:<5} {how}: {sc['prompt'][:60]}")
            pbar.update_absolute(i + 1)
        keys = torch.cat(stills).clamp(0, 1)
        idx = ",".join(str(int(s["start"])) for s in scenes)
        info = "\n".join([f"[Scene Stills] {len(scenes)} stills at {w}x{h}", *lines])
        return {"ui": {"text": [info]}, "result": (keys, idx, keys[:1], keys[-1:], info)}


# ---------------------------------------------------------------------------
# Travel Conditioning
# ---------------------------------------------------------------------------

class DifforumTravelConditioning:
    """The Director's prompt travel as one CONDITIONING with a prompt per frame.

    Samplers that render a whole clip at once - AnimateDiff, or any batch
    KSampler - take a single conditioning. This node stacks the timeline's
    scene prompts, blended from scene to scene, into a batch as long as the
    clip, so frame 40 is sampled with the prompt of frame 40: the Deforum
    prompt schedule, on AnimateDiff. Connect a CLIP to the Director (it encodes
    the scenes). The batch is as long as the Director's clip, or as the
    `images` / `latent` you connect (a rendered video of another length).
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {},
                "optional": {"direction": (DIRECTION,), "prompts": ("DIFFORUM_PROMPT",),
                             "images": ("IMAGE",), "latent": ("LATENT",)}}

    RETURN_TYPES = ("CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("positive", "frames", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, direction=None, prompts=None, images=None, latent=None):
        track = prompts if prompts is not None else (direction.prompts if direction is not None else None)
        if track is None or not len(track):
            raise ValueError("Travel Conditioning: connect a CLIP to the Director (or a Prompt Travel node) "
                             "so the scene prompts are encoded.")
        total = len(track)
        if images is not None:                       # match the clip being sampled, whatever its length
            n = int(images.shape[0])
        elif latent is not None:
            n = int(latent["samples"].shape[0])
        else:
            n = int(direction.params["max_frames"]) if direction is not None else total
        conds = [track[min(total - 1, round(f * (total - 1) / max(1, n - 1)) if n != total else f)]
                 for f in range(n)]
        tokens = max(c[0][0].shape[1] for c in conds)
        rows, pooled = [], []
        for c in conds:
            t = c[0][0]
            if t.shape[1] < tokens:                      # prompts of different length: pad the short ones
                t = torch.cat([t, torch.zeros(t.shape[0], tokens - t.shape[1], t.shape[2], dtype=t.dtype)], 1)
            rows.append(t[:1])
            po = c[0][1].get("pooled_output")
            if po is not None:
                pooled.append(po[:1])
        extra = {k: v for k, v in conds[0][0][1].items() if k != "pooled_output"}
        if len(pooled) == n:
            extra["pooled_output"] = torch.cat(pooled)
        keys = getattr(track, "keyframes", [])
        info = "\n".join([f"[Travel Conditioning] {n} frames, {len(keys)} prompts",
                          *(f"  f{f:<5} {t[:70]}" for f, t in keys)])
        return ([[torch.cat(rows), extra]], n, info)


NODE_CLASS_MAPPINGS = {
    "Difforum_ShotScript": DifforumShotScript,
    "Difforum_KeyframeAssets": DifforumKeyframeAssets,
    "Difforum_SceneStills": DifforumSceneStills,
    "Difforum_TravelConditioning": DifforumTravelConditioning,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_ShotScript": "Difforum · Shot Script (timeline from text)",
    "Difforum_KeyframeAssets": "Difforum · Keyframe Assets (folder)",
    "Difforum_SceneStills": "Difforum · Scene Stills (storyboard)",
    "Difforum_TravelConditioning": "Difforum · Travel Conditioning (prompt per frame)",
}
