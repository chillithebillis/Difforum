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
