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
from collections.abc import Sequence

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


class PromptTrack(Sequence):
    """Per-frame prompt travel, blended lazily.

    Holds one encoded CONDITIONING per keyframe plus the blend plan, and
    builds a frame's blended conditioning only when it is asked for (with a
    small cache). A 0.x `DIFFORUM_PROMPT` was a fully materialised list, which
    for long clips on T5/Gemma-sized encoders meant gigabytes of RAM. Indexing
    and len() behave exactly like that list, so older consumers keep working.
    """

    def __init__(self, encoded: list, keyframes: list[tuple[int, str]],
                 max_frames: int, easing: str = "linear"):
        if not encoded:
            raise ValueError("PromptTrack needs at least one keyframe")
        self.encoded = list(encoded)
        self.keyframes = list(keyframes)
        self.n = max(1, int(max_frames))
        self.easing = easing
        self.plan = plan_blend([f for f, _ in keyframes], self.n, easing)
        self._cache: dict[int, object] = {}

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(self.n))]
        i = int(i)
        if i < 0:
            i += self.n
        i = max(0, min(i, self.n - 1))
        li, ri, w = self.plan[i]
        key = (li, ri, round(w, 4))
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if li == ri or w <= 0.0:
            out = self.encoded[li]
        elif w >= 1.0:
            out = self.encoded[ri]
        else:
            out = blend_conditioning(self.encoded[ri], self.encoded[li], w)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = out
        return out

    def scene_index(self, f: int) -> int:
        """Index of the keyframe the frame is travelling away from."""
        return self.plan[max(0, min(int(f), self.n - 1))][0]

    def transition_weight(self, f: int) -> float:
        """0 = holding a keyframe prompt, ->1 = arriving at the next one."""
        li, ri, w = self.plan[max(0, min(int(f), self.n - 1))]
        return 0.0 if li == ri else float(w)

    def text_at(self, f: int) -> str:
        return self.keyframes[self.scene_index(f)][1]


def batch_conditioning(track) -> list:
    """Stack a per-frame prompt track into ONE CONDITIONING whose batch dim is
    the frame count (frame i = prompt i) - what AnimateDiff-style batch
    samplers expect."""
    import torch

    conds = [track[i] for i in range(len(track))]
    if not conds:
        raise ValueError("empty prompt track")
    tensors = [c[0][0] for c in conds]
    max_t = max(t.shape[1] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[1] < max_t:
            pad = torch.zeros((t.shape[0], max_t - t.shape[1], t.shape[2]),
                              dtype=t.dtype, device=t.device)
            t = torch.cat([t, pad], dim=1)
        padded.append(t)
    meta = dict(conds[0][0][1])
    pooled = [c[0][1].get("pooled_output") for c in conds]
    if all(p is not None for p in pooled):
        meta["pooled_output"] = torch.cat(pooled, dim=0)
    return [[torch.cat(padded, dim=0), meta]]
