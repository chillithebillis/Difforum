"""
Keyframe schedule parsing and evaluation for Difforum.

Deforum-compatible syntax:   0:(expr), 30:(expr), 120:(expr)
Each keyframe holds a math expression (see expr.py). A Schedule turns those
sparse keyframes into a dense per-frame value array of length `max_frames`.

Interpolation between two keyframes A (frame a) and B (frame b) at frame f:

    va = exprA(t=f) ; vb = exprB(t=f)
    u  = ease( (f - a) / (b - a) )
    value = lerp(va, vb, u)

Evaluating BOTH endpoints at the current frame `f` is deliberate: it lets a
pure oscillator like `0:(sin(t/10))` keep oscillating across the whole range,
while a constant like `0:(0), 60:(100)` interpolates linearly. This matches
how people expect Deforum/Parseq schedules to behave.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping

from .expr import CompiledExpr, compile_expr

# matches the start of a keyframe:  <frame> : (
# the expression body (which may contain nested parens) is read separately
# by scanning for the balanced closing paren.
_KF_HEAD_RE = re.compile(r"(-?\d+)\s*:\s*\(")


def _scan_keyframes(text: str) -> list[tuple[int, str]]:
    """Yield (frame, expr) pairs, honouring nested parentheses in exprs."""
    out: list[tuple[int, str]] = []
    pos = 0
    for m in _KF_HEAD_RE.finditer(text):
        if m.start() < pos:
            continue  # inside a previous expression body, skip
        frame = int(m.group(1))
        i = m.end()  # char right after the opening '('
        depth = 1
        start = i
        while i < len(text) and depth > 0:
            c = text[i]
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
            i += 1
        if depth != 0:
            raise ValueError(f"unbalanced parentheses in schedule near {text[m.start():m.start()+20]!r}")
        expr = text[start:i - 1]
        out.append((frame, expr))
        pos = i
    return out


EASINGS = ("linear", "ease_in", "ease_out", "ease_in_out", "step")


def _ease(u: float, mode: str) -> float:
    if u <= 0.0:
        return 0.0
    if u >= 1.0:
        return 1.0
    if mode == "linear":
        return u
    if mode == "ease_in":
        return u * u
    if mode == "ease_out":
        return 1.0 - (1.0 - u) * (1.0 - u)
    if mode == "ease_in_out":
        return u * u * (3.0 - 2.0 * u)  # smoothstep
    if mode == "step":
        return 0.0  # hold the left keyframe until the next one
    return u


@dataclass
class Keyframe:
    frame: int
    expr: CompiledExpr


@dataclass
class Schedule:
    """A dense, evaluated per-frame curve plus the metadata that produced it."""

    values: list[float]
    fps: float
    source: str = ""
    easing: str = "linear"
    keyframes: list[Keyframe] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.values)

    def at(self, frame: int) -> float:
        if not self.values:
            return 0.0
        i = max(0, min(frame, len(self.values) - 1))
        return self.values[i]

    def as_list(self) -> list[float]:
        return list(self.values)


def parse_keyframes(text: str) -> list[Keyframe]:
    """Parse `0:(expr), 60:(expr)` into sorted Keyframe objects."""
    text = (text or "").strip()
    if not text:
        raise ValueError("empty schedule string")
    matches = _scan_keyframes(text)
    if not matches:
        # allow a bare expression as shorthand for "0:(expr)"
        return [Keyframe(0, compile_expr(text))]
    kfs = [Keyframe(frame, compile_expr(expr)) for frame, expr in matches]
    kfs.sort(key=lambda k: k.frame)
    # de-duplicate identical frame numbers, last one wins
    dedup: dict[int, Keyframe] = {}
    for k in kfs:
        dedup[k.frame] = k
    return [dedup[f] for f in sorted(dedup)]


def build_schedule(
    text: str,
    max_frames: int,
    fps: float = 24.0,
    easing: str = "linear",
    extra_vars: Mapping[str, list[float]] | None = None,
) -> Schedule:
    """
    Evaluate `text` into a dense Schedule of length `max_frames`.

    `extra_vars` maps a variable name to a per-frame list (e.g. {"amp": [...]})
    so audio-reactive expressions like `0:(0.3 + 0.7*amp)` work. Lists shorter
    than max_frames are clamped at their last value.
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be >= 1")
    if easing not in EASINGS:
        raise ValueError(f"unknown easing {easing!r}, pick from {EASINGS}")

    kfs = parse_keyframes(text)
    extra_vars = extra_vars or {}

    def frame_vars(f: int) -> dict[str, float]:
        v = {
            "t": float(f),
            "f": float(f),
            "fps": float(fps),
            "s": f / fps if fps else 0.0,
            "max_f": float(max_frames),
        }
        for name, series in extra_vars.items():
            if series:
                idx = min(f, len(series) - 1)
                v[name] = float(series[idx])
            else:
                v[name] = 0.0
        return v

    values: list[float] = []
    n = len(kfs)
    for f in range(max_frames):
        env = frame_vars(f)
        # locate the segment [left, right] surrounding frame f
        left = kfs[0]
        right = kfs[0]
        for i in range(n):
            if kfs[i].frame <= f:
                left = kfs[i]
                right = kfs[i + 1] if i + 1 < n else kfs[i]
            else:
                break
        if left.frame == right.frame or f <= left.frame:
            values.append(left.expr.eval(env))
            continue
        if f >= right.frame:
            values.append(right.expr.eval(env))
            continue
        span = right.frame - left.frame
        u = _ease((f - left.frame) / span, easing)
        va = left.expr.eval(env)
        vb = right.expr.eval(env)
        values.append(va + (vb - va) * u)

    return Schedule(
        values=values, fps=fps, source=text, easing=easing, keyframes=kfs
    )
