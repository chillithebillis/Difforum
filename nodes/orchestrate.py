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
    ComfyUI's input folder (re-read when it changes), or from any node that
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
                return os.path.getmtime(_safe(file.strip()))
            except Exception:
                return float("nan")
        return ""

    def run(self, script, file, params=None, script_in=None):
        fps = float(params["fps"]) if params else 24.0
        frames = int(params["max_frames"]) if params else None
        if script_in is not None and str(script_in).strip():
            text, source = str(script_in), "script_in"
        elif str(file).strip():
            path = _safe(file.strip())
            if not os.path.isfile(path):
                raise ValueError(f"Shot Script: {file!r} not found in ComfyUI/input")
            with open(path, encoding="utf-8-sig") as fh:
                text, source = fh.read(), f"input/{file.strip()}"
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


NODE_CLASS_MAPPINGS = {
    "Difforum_ShotScript": DifforumShotScript,
    "Difforum_KeyframeAssets": DifforumKeyframeAssets,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_ShotScript": "Difforum · Shot Script (timeline from text)",
    "Difforum_KeyframeAssets": "Difforum · Keyframe Assets (folder)",
}
