"""
Optical-flow temporal stabilizer (anti-flicker). Feedback animations flicker
because each frame re-rolls texture; naive frame blending fixes that but ghosts
anything that moves. This aligns the previous stabilized frame onto the current
one along dense optical flow first, then blends only where the alignment is
photometrically trustworthy - flicker dies, motion stays crisp.

Flow comes from OpenCV Farneback (lazy import, computed at reduced scale for
speed); the warp and blend are torch. Frames are [N,H,W,3] in 0..1.
"""

from __future__ import annotations

import numpy as np
import torch


def _flow_cur_to_prev(cur_g: np.ndarray, prev_g: np.ndarray, scale: float) -> np.ndarray:
    import cv2
    h, w = cur_g.shape
    if scale < 1.0:
        sw, sh = max(16, int(w * scale)), max(16, int(h * scale))
        a = cv2.resize(cur_g, (sw, sh), interpolation=cv2.INTER_AREA)
        b = cv2.resize(prev_g, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        a, b = cur_g, prev_g
    flow = cv2.calcOpticalFlowFarneback(
        a, b, None, pyr_scale=0.5, levels=3, winsize=21,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    if scale < 1.0:
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR)
        flow[..., 0] *= w / a.shape[1]
        flow[..., 1] *= h / a.shape[0]
    return flow


def _warp_by_flow(img: torch.Tensor, flow: np.ndarray) -> torch.Tensor:
    """Sample `img` [1,H,W,3] at pos+flow (flow maps current -> previous)."""
    _, h, w, _ = img.shape
    dev, dt = img.device, img.dtype
    fl = torch.from_numpy(flow).to(dev, dt)
    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev, dtype=dt),
        torch.arange(w, device=dev, dtype=dt), indexing="ij")
    gx = ((xs + fl[..., 0]) / (w - 1)) * 2.0 - 1.0
    gy = ((ys + fl[..., 1]) / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
    chw = img.permute(0, 3, 1, 2)
    out = torch.nn.functional.grid_sample(
        chw, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.permute(0, 2, 3, 1)


def stabilize(frames: torch.Tensor, strength: float = 0.5,
              flow_scale: float = 0.5, error_gate: float = 0.15) -> torch.Tensor:
    """Flow-consistent temporal smoothing over a frame batch.

    strength: how much history to keep (0 = off, ~0.5 = strong deflicker).
    flow_scale: resolution factor for the flow computation (0.5 = fast).
    error_gate: photometric error above which blending shuts off (occlusions,
    new content) so moving/changing areas never ghost.
    """
    try:
        import cv2  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"OpenCV needed for flow stabilization ({e}). pip install opencv-python")
    if strength <= 0.0 or frames.shape[0] < 2:
        return frames

    s = float(min(0.95, strength))
    out = [frames[0:1]]
    prev_stab = frames[0:1]
    prev_g = (frames[0].mean(dim=-1).cpu().numpy() * 255).astype(np.uint8)
    for i in range(1, frames.shape[0]):
        cur = frames[i:i + 1]
        cur_g = (frames[i].mean(dim=-1).cpu().numpy() * 255).astype(np.uint8)
        flow = _flow_cur_to_prev(cur_g, prev_g, float(flow_scale))
        aligned = _warp_by_flow(prev_stab, flow)
        err = (aligned - cur).abs().mean(dim=-1, keepdim=True)          # [1,H,W,1]
        # judge mismatch over a neighbourhood, not per pixel, so grain/boil
        # (the thing we are removing) does not shut the blend off itself
        err = torch.nn.functional.avg_pool2d(
            err.permute(0, 3, 1, 2), 5, stride=1, padding=2).permute(0, 2, 3, 1)
        conf = (1.0 - err / max(1e-4, float(error_gate))).clamp(0.0, 1.0)
        w = s * conf
        stab = (cur * (1.0 - w) + aligned * w).clamp(0.0, 1.0)
        out.append(stab)
        prev_stab = stab
        prev_g = cur_g
    return torch.cat(out, dim=0)


MOSH_MODES = ("grid", "melt", "edge")


def datamosh(frames: torch.Tensor, intensity: float = 0.7, mode: str = "grid",
             block_size: int = 16, flow_scale: float = 0.5) -> torch.Tensor:
    """Datamosh: keep the motion vectors, drop the refresh. Each output frame
    is the previous OUTPUT warped by the real motion field, mixed with the true
    frame by intensity - the classic I-frame-removal smear, controllable."""
    import cv2
    if mode not in MOSH_MODES:
        raise ValueError(f"unknown mosh mode {mode!r}, pick from {MOSH_MODES}")
    n = frames.shape[0]
    if n < 2 or intensity <= 0.0:
        return frames
    h, w = frames.shape[1], frames.shape[2]
    out = [frames[0:1]]
    prev_out = frames[0:1]
    prev_g = (frames[0].mean(dim=-1).cpu().numpy() * 255).astype(np.uint8)
    for i in range(1, n):
        cur = frames[i:i + 1]
        cur_g = (frames[i].mean(dim=-1).cpu().numpy() * 255).astype(np.uint8)
        flow = _flow_cur_to_prev(cur_g, prev_g, float(flow_scale))
        if mode == "grid":
            bs = max(4, int(block_size))
            small = cv2.resize(flow, (max(1, w // bs), max(1, h // bs)),
                               interpolation=cv2.INTER_AREA)
            flow = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif mode == "melt":
            flow = cv2.GaussianBlur(flow, (0, 0), 9) * 1.6
        else:  # edge: mosh only along contours
            edges = cv2.Canny(cur_g, 60, 120)
            mask = cv2.dilate(edges, np.ones((5, 5), np.uint8)) / 255.0
            flow = flow * mask[..., None]
        pred = _warp_by_flow(prev_out, flow)
        mixed = (cur * (1.0 - intensity) + pred * intensity).clamp(0.0, 1.0)
        out.append(mixed)
        prev_out = mixed
        prev_g = cur_g
    return torch.cat(out, dim=0)
