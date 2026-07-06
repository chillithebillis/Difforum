"""
Procedural glitch for Difforum: convolution kernels, DSP databending (the
frame treated as a 1D signal with a delay line), bitcrush, VHS damage and RGB
split. Pure torch, batch-native ([N,H,W,3] in 0..1), deterministic via seed -
no cv2, no per-instance state, so it stays headless- and cloud-safe.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

KERNELS = ("none", "blur", "sharpen", "edge", "emboss")

_K = {
    "blur": [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    "sharpen": [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    "edge": [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
    "emboss": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
}


def _bhwc(x):
    return (x.unsqueeze(0), True) if x.dim() == 3 else (x, False)


def convolve(frames: torch.Tensor, kernel: str = "blur", mix: float = 1.0) -> torch.Tensor:
    """Classic convolution-art kernels, blended over the original by `mix`."""
    if kernel == "none" or mix <= 0.0:
        return frames
    if kernel not in _K:
        raise ValueError(f"unknown kernel {kernel!r}, pick from {KERNELS}")
    x, sq = _bhwc(frames)
    rgb = x[..., :3]
    k = torch.tensor(_K[kernel], dtype=rgb.dtype, device=rgb.device)
    if kernel == "blur":
        k = k / 9.0
    k = k.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    chw = rgb.permute(0, 3, 1, 2)
    out = F.conv2d(chw, k, padding=1, groups=3).permute(0, 2, 3, 1)
    if kernel == "edge":
        out = out.abs()
    out = (rgb + (out - rgb) * float(mix)).clamp(0.0, 1.0)
    return out[0] if sq else out


def dsp_delay(frames: torch.Tensor, delay: int = 4096, feedback: float = 0.5) -> torch.Tensor:
    """Databending: flatten each frame to a 1D signal and run a delay line."""
    if delay <= 0 or feedback <= 0.0:
        return frames
    x, sq = _bhwc(frames)
    rgb = x[..., :3]
    n = rgb.shape[0]
    flat = rgb.reshape(n, -1)
    echo = torch.roll(flat, shifts=int(delay), dims=1)
    out = ((1.0 - feedback) * flat + feedback * echo).reshape(rgb.shape).clamp(0.0, 1.0)
    return out[0] if sq else out


def bitcrush(frames: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Reduce colour bit depth (0 = off, 1..7 = crush)."""
    if bits <= 0 or bits >= 8:
        return frames
    q = float(2 ** int(bits) - 1)
    return (frames.clamp(0.0, 1.0) * q).floor() / q


def vhs(frames: torch.Tensor, jitter: int = 0, band: float = 0.0,
        chroma_blur: int = 0, seed: int = 0) -> torch.Tensor:
    """Analog tape damage: per-row horizontal jitter, a scrolling tracking band
    and horizontal chroma blur on the R/B channels."""
    x, sq = _bhwc(frames)
    rgb = x[..., :3].clone()
    n, h, w, _ = rgb.shape
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)

    if jitter > 0:
        shifts = torch.randint(-int(jitter), int(jitter) + 1, (n, h), generator=gen)
        base = torch.arange(w)
        idx = (base.view(1, 1, w) - shifts.unsqueeze(-1)) % w          # [N,H,W]
        rgb = torch.gather(rgb, 2, idx.unsqueeze(-1).expand(n, h, w, 3).to(rgb.device))

    if band > 0.0:
        bh = max(2, h // 24)
        for i in range(n):
            y0 = int((i * 7 + int(seed)) % max(1, h - bh))
            noise = torch.rand(bh, w, 1, generator=gen).to(rgb.device, rgb.dtype)
            rgb[i, y0:y0 + bh] = (rgb[i, y0:y0 + bh] * (1.0 - band)
                                  + noise * band).clamp(0.0, 1.0)

    if chroma_blur > 1:
        k = int(chroma_blur) | 1
        ker = torch.ones(1, 1, 1, k, dtype=rgb.dtype, device=rgb.device) / k
        for c in (0, 2):                                                # R and B only
            ch = rgb[..., c].unsqueeze(1)
            rgb[..., c] = F.conv2d(ch, ker, padding=(0, k // 2)).squeeze(1)

    out = rgb.clamp(0.0, 1.0)
    return out[0] if sq else out


def apply_glitch(frames: torch.Tensor, kernel: str = "none", kernel_mix: float = 0.5,
                 delay: int = 0, feedback: float = 0.5, bits: int = 0,
                 jitter: int = 0, band: float = 0.0, chroma: int = 0,
                 rgb_split: int = 0, seed: int = 0) -> torch.Tensor:
    """The full ConvoLab-style chain in a stable order."""
    from .look import chroma_shift
    out = convolve(frames, kernel, kernel_mix)
    out = dsp_delay(out, delay, feedback)
    out = bitcrush(out, bits)
    out = vhs(out, jitter=jitter, band=band, chroma_blur=chroma, seed=seed)
    if rgb_split:
        out = chroma_shift(out, amount=float(rgb_split))
    return out
