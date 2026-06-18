"""
Symmetry / kaleidoscope for Difforum.

Mirror and radial-fold an image (or a whole batch of frames). Applied to the
final frames it makes mesmerizing symmetric video; fed back inside the feedback
loop it compounds each frame and the diffusion heals the seams, giving a living
kaleidoscope. Pure torch, testable. Images are [H,W,C] or [B,H,W,C] in 0..1.
"""

from __future__ import annotations

import math

import torch

SYMMETRY_MODES = ("none", "mirror_h", "mirror_v", "mirror_quad", "kaleidoscope")


def _as_bhwc(image):
    return (image.unsqueeze(0), True) if image.dim() == 3 else (image, False)


def _mirror_axis(img: torch.Tensor, dim: int, flip: bool) -> torch.Tensor:
    """Reflect an axis about its centre using one half as the source."""
    length = img.shape[dim]
    idx = torch.arange(length, device=img.device)
    rev = (length - 1) - idx
    src = torch.maximum(idx, rev) if flip else torch.minimum(idx, rev)
    return img.index_select(dim, src)


def _kaleidoscope(img, segments, center_x, center_y, angle_deg, flip):
    b, h, w, c = img.shape
    dev, dt = img.device, img.dtype
    cx, cy = center_x * (w - 1), center_y * (h - 1)
    a0 = math.radians(angle_deg)
    seg = 2.0 * math.pi / max(2, int(segments))

    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev, dtype=dt),
        torch.arange(w, device=dev, dtype=dt),
        indexing="ij",
    )
    dx, dy = xs - cx, ys - cy
    r = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx) - a0
    # fold the angle into one wedge with a mirror (triangle wave)
    m = torch.remainder(theta, 2.0 * seg)
    folded = torch.abs(m - seg)
    if flip:
        folded = seg - folded
    src_t = folded + a0
    sx = cx + r * torch.cos(src_t)
    sy = cy + r * torch.sin(src_t)

    gx = (sx / (w - 1)) * 2.0 - 1.0
    gy = (sy / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, h, w, 2)
    chw = img.permute(0, 3, 1, 2)
    out = torch.nn.functional.grid_sample(
        chw, grid, mode="bilinear", padding_mode="reflection", align_corners=True
    )
    return out.permute(0, 2, 3, 1)


def apply_symmetry(
    image: torch.Tensor,
    mode: str = "mirror_h",
    segments: int = 6,
    flip: bool = False,
    mix: float = 1.0,
    center_x: float = 0.5,
    center_y: float = 0.5,
    angle: float = 0.0,
) -> torch.Tensor:
    """Return `image` made symmetric. `mix` blends original (0) to symmetric (1)."""
    if mode == "none" or mix <= 0.0:
        return image
    if mode not in SYMMETRY_MODES:
        raise ValueError(f"unknown mode {mode!r}, pick from {SYMMETRY_MODES}")
    img, squeezed = _as_bhwc(image)

    if mode == "mirror_h":
        sym = _mirror_axis(img, 2, flip)
    elif mode == "mirror_v":
        sym = _mirror_axis(img, 1, flip)
    elif mode == "mirror_quad":
        sym = _mirror_axis(_mirror_axis(img, 2, flip), 1, flip)
    else:  # kaleidoscope
        sym = _kaleidoscope(img, segments, center_x, center_y, angle, flip)

    out = img + (sym - img) * float(mix) if mix < 1.0 else sym
    return out[0] if squeezed else out
