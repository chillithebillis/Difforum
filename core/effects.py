"""
Temporal effects for Difforum frame batches. Pure torch, testable.
Frames are [N,H,W,C] in 0..1.
"""

from __future__ import annotations

import torch


def echo_trails(frames: torch.Tensor, decay: float = 0.6, mix: float = 0.5) -> torch.Tensor:
    """Long-exposure style motion trails: blend a decaying echo of past frames
    into each frame. `decay` = how long the trail lasts (0..1), `mix` = how
    strong the trail shows. Smooth, hypnotic motion blur without interpolation."""
    if mix <= 0.0:
        return frames
    out = frames.clone()
    echo = frames[0].clone()
    d = float(max(0.0, min(0.999, decay)))
    m = float(max(0.0, min(1.0, mix)))
    for i in range(frames.shape[0]):
        echo = frames[i] * (1.0 - d) + echo * d
        out[i] = frames[i] * (1.0 - m) + echo * m
    return out.clamp(0.0, 1.0)


def pingpong(frames: torch.Tensor) -> torch.Tensor:
    """Seamless loop: forward then reversed (endpoints not duplicated)."""
    if frames.dim() == 3 or frames.shape[0] < 3:
        return frames
    return torch.cat([frames, frames.flip(0)[1:-1]], dim=0)


def loop_blend(frames: torch.Tensor, blend: int = 12, flow: bool = True) -> torch.Tensor:
    """Perfect FORWARD loop: the last `blend` frames are morph-faded into the
    first ones (optical-flow aligned when available), then dropped. Output has
    N - blend frames and plays seamlessly head-to-tail without reversing."""
    n = frames.shape[0]
    k = int(min(max(0, blend), n // 3))
    if k < 1:
        return frames
    warp = None
    if flow:
        try:
            import numpy as _np

            from .flow import _flow_cur_to_prev, _warp_by_flow

            def warp(tail, head, a):
                hg = (head[0].mean(dim=-1).cpu().numpy() * 255).astype(_np.uint8)
                tg = (tail[0].mean(dim=-1).cpu().numpy() * 255).astype(_np.uint8)
                fl = _flow_cur_to_prev(hg, tg, 0.5)          # head -> tail sampling
                return _warp_by_flow(tail, fl * a)
        except Exception:
            warp = None
    out = frames[: n - k].clone()
    for i in range(k):
        a = (i + 1) / (k + 1)
        tail = frames[n - k + i : n - k + i + 1]
        head = frames[i : i + 1]
        moved = warp(tail, head, a) if warp is not None else tail
        out[i] = (moved * (1.0 - a) + head * a).clamp(0.0, 1.0)[0]
    return out
