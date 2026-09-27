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
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "IMAGE", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("keyframes", "indices", "sparse_batch", "first_frame", "last_frame", "length", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, frames, grid, every_seconds, fps, indices=""):
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
        return (keys, ",".join(map(str, idx)), sparse, frames[:1], frames[-1:], int(length), info)


class DifforumCameraPrompt:
    """The camera, in words - for prompt-driven video models (MiniMax H3,
    LTX, Seedance, Veo...). Uses the Director's blocks when available, or
    reads any camera track and names its moves."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "format": (["sentence", "timed lines", "prompt suffix"], {"default": "sentence"}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "params": (PARAMS,),
                "camera": (CAMERA,),
                "prefix": ("STRING", {"default": "", "multiline": True,
                           "tooltip": "Your shot description; the camera sentence is appended."}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, format, direction=None, params=None, camera=None, prefix=""):
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
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT", "INT", "STRING", "INT", "STRING")
    RETURN_NAMES = ("first_frame", "last_frame", "length", "width", "height", "prompt",
                    "segments", "info")
    FUNCTION = "run"
    CATEGORY = CAT_BRIDGE

    def run(self, frames, segment_length, segment, direction=None, params=None, camera=None,
            shot_description=""):
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
        prompt = " ".join(x for x in (shot_description.strip(), cam_text) if x)
        info = (f"H3 segment {s + 1}/{segments}: frames {start}-{end} "
                f"(length {seg_len}, {seg_len / 24:.2f}s @ 24fps), {w32}x{h32}\n"
                f"  mode fl2va - wire first/last into MiniMax H3 Image to Video")
        return (first, last, int(seg_len), int(w32), int(h32), prompt, int(segments), info)


NODE_CLASS_MAPPINGS = {
    "Difforum_GuideFrames": DifforumGuideFrames,
    "Difforum_Keyframes": DifforumKeyframes,
    "Difforum_CameraPrompt": DifforumCameraPrompt,
    "Difforum_LTXGuides": DifforumLTXGuides,
    "Difforum_H3Shot": DifforumH3Shot,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_GuideFrames": "Difforum · Guide Frames",
    "Difforum_Keyframes": "Difforum · Keyframes",
    "Difforum_CameraPrompt": "Difforum · Camera → Prompt",
    "Difforum_LTXGuides": "Difforum · LTX Guides",
    "Difforum_H3Shot": "Difforum · H3 Shot",
}
