"""
Detail preservation for the feedback loop. Every warp + VAE round-trip softens
the frame a little; over dozens of frames that compounds into mush. The classic
Deforum antidote, rebuilt here: re-sharpen the warped frame, then inject a touch
of noise so the sampler has fresh high-frequency signal to resolve into detail
instead of amplifying blur. Pure torch, [H,W,C] or [B,H,W,C] in 0..1.
"""

from __future__ import annotations

import torch

from .look import _bhwc, _gaussian_blur

NOISE_MODES = ("gaussian", "plasma")


def sharpen(image: torch.Tensor, amount: float = 0.3, radius: int = 2) -> torch.Tensor:
    """Unsharp mask: image + amount * (image - blur(image)). amount 0 = off."""
    if amount <= 0.0:
        return image
    x, sq = _bhwc(image)
    rgb = x[..., :3]
    blurred = _gaussian_blur(rgb.permute(0, 3, 1, 2), radius).permute(0, 2, 3, 1)
    out = (rgb + float(amount) * (rgb - blurred)).clamp(0.0, 1.0)
    return out[0] if sq else out


def _plasma(b, h, w, gen, device, dtype):
    """Multi-scale smooth noise (perlin-ish): sum of upscaled random grids."""
    out = torch.zeros(b, 1, h, w, dtype=dtype)
    total = 0.0
    for cell, weight in ((4, 0.5), (8, 0.3), (16, 0.2)):
        gh, gw = max(2, h // cell), max(2, w // cell)
        g = torch.randn(b, 1, gh, gw, generator=gen)
        out += weight * torch.nn.functional.interpolate(
            g, size=(h, w), mode="bilinear", align_corners=False)
        total += weight
    out = out / total
    return out.to(device)


def add_noise(image: torch.Tensor, amount: float = 0.03, seed: int = 0,
              mode: str = "gaussian") -> torch.Tensor:
    """Inject noise before re-diffusion (the Deforum noise schedule trick)."""
    if amount <= 0.0:
        return image
    if mode not in NOISE_MODES:
        raise ValueError(f"unknown noise mode {mode!r}, pick from {NOISE_MODES}")
    x, sq = _bhwc(image)
    rgb = x[..., :3]
    b, h, w, _ = rgb.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    if mode == "gaussian":
        n = torch.randn(b, h, w, 1, generator=gen).to(rgb.device, rgb.dtype)
    else:
        n = _plasma(b, h, w, gen, rgb.device, rgb.dtype).permute(0, 2, 3, 1)
    out = (rgb + n * float(amount)).clamp(0.0, 1.0)
    return out[0] if sq else out


def adjust_contrast(image: torch.Tensor, factor: float = 1.0) -> torch.Tensor:
    """Mid-pivot contrast. 1.0 = off; ~1.02-1.05 counters feedback gray-drift."""
    if factor == 1.0:
        return image
    x, sq = _bhwc(image)
    rgb = x[..., :3]
    out = ((rgb - 0.5) * float(factor) + 0.5).clamp(0.0, 1.0)
    return out[0] if sq else out


def detail_guard(image: torch.Tensor, sharpen_amount: float = 0.0,
                 noise_amount: float = 0.0, contrast: float = 1.0,
                 seed: int = 0, noise_mode: str = "gaussian") -> torch.Tensor:
    """Sharpen -> contrast -> noise, the order that feeds a sampler best."""
    out = sharpen(image, sharpen_amount)
    out = adjust_contrast(out, contrast)
    out = add_noise(out, noise_amount, seed=seed, mode=noise_mode)
    return out
