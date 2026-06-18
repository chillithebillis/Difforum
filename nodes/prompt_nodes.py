"""
Difforum Prompt Schedule node - prompt travel across keyframes.

Encodes each keyframe prompt with CLIP once, then builds a per-frame blended
CONDITIONING list (DIFFORUM_PROMPT). Feed that into the Feedback Sampler's
optional `positive_schedule` input to morph the prompt over the animation.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.prompt import (  # noqa: E402
    blend_conditioning, parse_prompt_schedule, plan_blend, scenes_to_keyframes,
)
from core.schedule import EASINGS  # noqa: E402

CATEGORY = "Difforum/schedule"

_DEFAULT = "0: a serene misty forest, soft light\n30: a stormy ocean, dramatic clouds\n59: a vast starry galaxy, vivid colors"


def _build_per_frame(clip, keyframes, max_frames, easing):
    """Encode each keyframe prompt once, then blend per frame -> conditioning list."""
    frames = [f for f, _ in keyframes]
    encoded = [clip.encode_from_tokens_scheduled(clip.tokenize(t)) for _, t in keyframes]
    per_frame = []
    for (li, ri, w) in plan_blend(frames, max_frames, easing):
        if li == ri or w <= 0.0:
            per_frame.append(encoded[li])
        elif w >= 1.0:
            per_frame.append(encoded[ri])
        else:
            per_frame.append(blend_conditioning(encoded[ri], encoded[li], w))
    return per_frame


class DifforumPromptSchedule:
    """Build a per-frame blended CONDITIONING list from a prompt schedule."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "clip": ("CLIP",),
                "prompts": ("STRING", {"multiline": True, "default": _DEFAULT}),
                "easing": (list(EASINGS), {"default": "ease_in_out"}),
            }
        }

    RETURN_TYPES = ("DIFFORUM_PROMPT", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive_schedule", "first_frame_cond", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, clip, prompts, easing):
        kfs = parse_prompt_schedule(prompts)
        n = params["max_frames"]
        per_frame = _build_per_frame(clip, kfs, n, easing)
        info = (
            f"prompt travel: {len(kfs)} keyframes over {n} frames, easing={easing}\n"
            + "\n".join(f"  {f}: {t[:48]}" for f, t in kfs)
        )
        return (per_frame, per_frame[0], info)


class DifforumPromptScenes:
    """Prompt travel by scenes - one text box per scene, spaced automatically."""

    @classmethod
    def INPUT_TYPES(cls):
        boxes = {}
        defaults = [
            "a serene misty forest, soft volumetric light",
            "a glowing crystal cave, bioluminescent",
            "a vast starry galaxy, swirling nebula",
            "",
        ]
        for idx in range(1, 5):
            boxes[f"scene_{idx}"] = ("STRING", {"multiline": True, "default": defaults[idx - 1]})
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "clip": ("CLIP",),
                **boxes,
                "easing": (list(EASINGS), {"default": "ease_in_out"}),
            }
        }

    RETURN_TYPES = ("DIFFORUM_PROMPT", "CONDITIONING", "STRING")
    RETURN_NAMES = ("positive_schedule", "first_frame_cond", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, clip, scene_1, scene_2, scene_3, scene_4, easing):
        n = params["max_frames"]
        kfs = scenes_to_keyframes([scene_1, scene_2, scene_3, scene_4], n)
        per_frame = _build_per_frame(clip, kfs, n, easing)
        info = (
            f"{len(kfs)} scenes over {n} frames (every ~{n // max(1, len(kfs))} frames)\n"
            + "\n".join(f"  frame {f}: {t[:44]}" for f, t in kfs)
        )
        return (per_frame, per_frame[0], info)


class DifforumPromptBatch:
    """Stack a per-frame prompt schedule into ONE batched CONDITIONING.

    The feedback sampler uses a per-frame conditioning list (DIFFORUM_PROMPT).
    AnimateDiff (and any batch sampler) instead wants a single CONDITIONING whose
    cond tensor has batch dim = number of frames, so frame i gets prompt i. This
    converts one to the other - the bridge for AnimateDiff prompt travel.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"positive_schedule": ("DIFFORUM_PROMPT",)}}

    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, positive_schedule):
        import torch

        conds = list(positive_schedule)
        if not conds:
            raise ValueError("empty positive_schedule")
        tensors = [c[0][0] for c in conds]  # each [1, T, D]
        max_t = max(t.shape[1] for t in tensors)
        padded = []
        for t in tensors:
            if t.shape[1] < max_t:
                pad = torch.zeros((t.shape[0], max_t - t.shape[1], t.shape[2]),
                                  dtype=t.dtype, device=t.device)
                t = torch.cat([t, pad], dim=1)
            padded.append(t)
        batched = torch.cat(padded, dim=0)  # [N, T, D]

        meta = dict(conds[0][0][1])
        pooled = [c[0][1].get("pooled_output") for c in conds]
        if all(p is not None for p in pooled):
            meta["pooled_output"] = torch.cat(pooled, dim=0)  # [N, D]
        return ([[batched, meta]],)


NODE_CLASS_MAPPINGS = {
    "DifforumPromptSchedule": DifforumPromptSchedule,
    "DifforumPromptScenes": DifforumPromptScenes,
    "DifforumPromptBatch": DifforumPromptBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumPromptSchedule": "Difforum · Prompt Schedule (travel)",
    "DifforumPromptScenes": "Difforum · Prompt Scenes (travel)",
    "DifforumPromptBatch": "Difforum · Prompt Batch (→ AnimateDiff)",
}
