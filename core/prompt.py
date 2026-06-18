"""
Prompt travel for Difforum - interpolate the text prompt across keyframes.

Parse a Deforum/FizzNodes-style prompt schedule:

    0: a serene misty forest, soft light
    30: a stormy ocean, dramatic clouds
    59: a vast starry galaxy

and produce, per frame, a blended CONDITIONING between the two surrounding
keyframe prompts. The blend replicates ComfyUI's ConditioningAverage
(`addWeighted`) so it behaves exactly like the stock node.

The parser + blend plan are pure (testable); CLIP encoding happens in the node.
"""

from __future__ import annotations

import re

from .schedule import EASINGS, _ease

# matches a line starting with  <frame> :  then the prompt text
_LINE_RE = re.compile(r"^\s*(-?\d+)\s*:\s*(.*?)\s*$")


def parse_prompt_schedule(text: str) -> list[tuple[int, str]]:
    """
    Parse `frame: prompt` lines into sorted (frame, text) pairs.

    Tolerates `0:(text)`, surrounding quotes and trailing commas (so a Deforum
    JSON body can be pasted). Blank lines are ignored.
    """
    out: dict[int, str] = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        frame = int(m.group(1))
        prompt = m.group(2).strip().rstrip(",").strip()
        # strip one layer of (...) or quotes
        if prompt.startswith("(") and prompt.endswith(")"):
            prompt = prompt[1:-1].strip()
        if len(prompt) >= 2 and prompt[0] in "\"'" and prompt[-1] == prompt[0]:
            prompt = prompt[1:-1]
        out[frame] = prompt
    if not out:
        raise ValueError("no `frame: prompt` lines found in prompt schedule")
    return [(f, out[f]) for f in sorted(out)]


def scenes_to_keyframes(scenes: list[str], max_frames: int) -> list[tuple[int, str]]:
    """
    Turn a list of scene prompts into evenly-spaced keyframes.

    Empty/whitespace scenes are skipped. One scene -> held the whole time; N
    scenes -> spread from frame 0 to the last frame so each gets equal screen
    time with smooth transitions between them.
    """
    texts = [s.strip() for s in scenes if s and s.strip()]
    if not texts:
        raise ValueError("no non-empty scenes provided")
    if len(texts) == 1:
        return [(0, texts[0])]
    last = max(1, max_frames - 1)
    n = len(texts)
    return [(round(idx * last / (n - 1)), t) for idx, t in enumerate(texts)]


def plan_blend(frames: list[int], max_frames: int, easing: str = "linear"):
    """
    For each output frame, return (left_idx, right_idx, weight_right) into the
    keyframe list, where weight_right is the blend amount toward the right
    keyframe (0 = fully left, 1 = fully right).
    """
    if easing not in EASINGS:
        raise ValueError(f"unknown easing {easing!r}")
    n = len(frames)
    plan = []
    for f in range(max_frames):
        # locate surrounding keyframes
        li = 0
        ri = 0
        for i in range(n):
            if frames[i] <= f:
                li = i
                ri = i + 1 if i + 1 < n else i
            else:
                break
        a, b = frames[li], frames[ri]
        if li == ri or f <= a:
            w = 0.0
        elif f >= b:
            w = 1.0
        else:
            w = _ease((f - a) / (b - a), easing)
        plan.append((li, ri, w))
    return plan


def blend_conditioning(cond_to, cond_from, to_strength: float):
    """
    Weighted blend of two CONDITIONINGs (ComfyUI addWeighted semantics).

    result = cond_to * to_strength + cond_from * (1 - to_strength)
    Shapes are reconciled by truncating/zero-padding the token dimension.
    """
    import torch

    if to_strength >= 1.0:
        return cond_to
    if to_strength <= 0.0:
        return cond_from

    cf = cond_from[0][0]
    pooled_from = cond_from[0][1].get("pooled_output", None)
    out = []
    for i in range(len(cond_to)):
        t1 = cond_to[i][0]
        pooled_to = cond_to[i][1].get("pooled_output", pooled_from)
        t0 = cf[:, : t1.shape[1]]
        if t0.shape[1] < t1.shape[1]:
            pad = torch.zeros(
                (t0.shape[0], t1.shape[1] - t0.shape[1], t0.shape[2]),
                dtype=t0.dtype, device=t0.device,
            )
            t0 = torch.cat([t0, pad], dim=1)
        tw = t1 * to_strength + t0 * (1.0 - to_strength)
        meta = cond_to[i][1].copy()
        if pooled_from is not None and pooled_to is not None:
            meta["pooled_output"] = pooled_to * to_strength + pooled_from * (1.0 - to_strength)
        out.append([tw, meta])
    return out
