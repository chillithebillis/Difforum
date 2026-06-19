"""
VJ 'look' grade for Difforum: cinematic colour grade, neon glow, chroma shift,
vignette and film grain, plus one-shot presets. Pure torch, testable. Operates
on [H,W,C] or [B,H,W,C] in 0..1 (RGB). Built to polish video footage for live
visuals: stack the pieces, or use a preset and dial intensity.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

LOOK_PRESETS = ("none", "neon", "cinematic", "vaporwave", "film", "noir", "psychedelic")


def _bhwc(x):
    return (x.unsqueeze(0), True) if x.dim() == 3 else (x, False)


def _luma(x):
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    return (0.2126 * r + 0.7152 * g + 0.0722 * b).unsqueeze(-1)


def color_grade(image, exposure=0.0, contrast=1.0, saturation=1.0,
                temperature=0.0, tint=0.0, lift=0.0, gamma=1.0, gain=1.0):
    """Exposure / contrast / saturation / white balance / lift-gamma-gain."""
    x, sq = _bhwc(image)
    out = x[..., :3].clone()
    out = out * (2.0 ** exposure)
    if temperature != 0.0:
        out[..., 0] = out[..., 0] * (1.0 + 0.3 * temperature)
        out[..., 2] = out[..., 2] * (1.0 - 0.3 * temperature)
    if tint != 0.0:
        out[..., 1] = out[..., 1] * (1.0 + 0.3 * tint)
    out = out * gain + lift * (1.0 - out)              # gain (mult) + lift (shadows)
    out = out.clamp(min=1e-6) ** (1.0 / max(1e-3, gamma))
    out = (out - 0.5) * contrast + 0.5                  # contrast around mid
    if saturation != 1.0:
        lum = _luma(out)
        out = lum + (out - lum) * saturation
    out = out.clamp(0.0, 1.0)
    return out[0] if sq else out


def _gaussian_blur(chw, radius):
    r = max(1, int(radius))
    sigma = r / 2.0
    k = torch.arange(-r, r + 1, dtype=chw.dtype, device=chw.device)
    k = torch.exp(-(k * k) / (2.0 * sigma * sigma))
    k = k / k.sum()
    c = chw.shape[1]
    kx = k.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
    ky = k.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
    x = F.conv2d(chw, kx, padding=(0, r), groups=c)
    x = F.conv2d(x, ky, padding=(r, 0), groups=c)
    return x


def glow(image, threshold=0.7, radius=8, intensity=0.6):
    """Bloom: blur the bright areas and screen-blend them back (neon look)."""
    if intensity <= 0.0:
        return image
    x, sq = _bhwc(image)
    rgb = x[..., :3]
    mask = (_luma(rgb) > threshold).to(rgb.dtype)
    bright = (rgb * mask)
    blurred = _gaussian_blur(bright.permute(0, 3, 1, 2), radius).permute(0, 2, 3, 1)
    g = (blurred * intensity).clamp(0.0, 1.0)
    out = 1.0 - (1.0 - rgb) * (1.0 - g)                 # screen blend
    out = out.clamp(0.0, 1.0)
    return out[0] if sq else out


def chroma_shift(image, amount=3.0, angle=0.0):
    """Split R and B channels in opposite directions (chromatic aberration)."""
    if amount == 0.0:
        return image
    x, sq = _bhwc(image)
    rgb = x[..., :3]
    dx = int(round(amount * math.cos(math.radians(angle))))
    dy = int(round(amount * math.sin(math.radians(angle))))
    r = torch.roll(rgb[..., 0], shifts=(dy, dx), dims=(1, 2))
    b = torch.roll(rgb[..., 2], shifts=(-dy, -dx), dims=(1, 2))
    out = torch.stack([r, rgb[..., 1], b], dim=-1)
    return out[0] if sq else out


def vignette(image, amount=0.3, softness=0.5):
    """Darken toward the edges."""
    if amount <= 0.0:
        return image
    x, sq = _bhwc(image)
    rgb = x[..., :3]
    _b, h, w, _c = rgb.shape
    ys = torch.linspace(-1, 1, h, device=rgb.device, dtype=rgb.dtype).view(1, h, 1)
    xs = torch.linspace(-1, 1, w, device=rgb.device, dtype=rgb.dtype).view(1, 1, w)
    d = torch.sqrt(xs * xs + ys * ys) / math.sqrt(2.0)
    edge = torch.clamp((d - (1.0 - softness)) / max(1e-3, softness), 0.0, 1.0)
    mask = (1.0 - amount * edge).unsqueeze(-1)
    out = (rgb * mask).clamp(0.0, 1.0)
    return out[0] if sq else out


def grain(image, amount=0.05, seed=0):
    """Add film grain (gaussian noise)."""
    if amount <= 0.0:
        return image
    x, sq = _bhwc(image)
    rgb = x[..., :3]
    gen = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    noise = torch.randn(rgb.shape, generator=gen).to(rgb.device, rgb.dtype) * amount
    out = (rgb + noise).clamp(0.0, 1.0)
    return out[0] if sq else out


# preset = staged params; values are at intensity 1.0 and scale with intensity
_PRESETS = {
    "neon": {
        "grade": {"contrast": 1.15, "saturation": 1.4, "exposure": 0.05},
        "chroma": {"amount": 2.0}, "glow": {"threshold": 0.6, "radius": 10, "intensity": 0.7},
        "vignette": {"amount": 0.25}, "grain": {"amount": 0.02},
    },
    "cinematic": {
        "grade": {"contrast": 1.1, "saturation": 0.95, "temperature": 0.1, "lift": 0.02},
        "glow": {"threshold": 0.8, "radius": 6, "intensity": 0.25},
        "vignette": {"amount": 0.35}, "grain": {"amount": 0.03},
    },
    "vaporwave": {
        "grade": {"saturation": 1.3, "temperature": -0.15, "tint": 0.1, "contrast": 1.05},
        "chroma": {"amount": 4.0}, "glow": {"threshold": 0.65, "radius": 12, "intensity": 0.6},
        "vignette": {"amount": 0.2},
    },
    "film": {
        "grade": {"contrast": 1.08, "saturation": 0.9, "gamma": 1.05},
        "glow": {"threshold": 0.85, "radius": 4, "intensity": 0.15},
        "vignette": {"amount": 0.3}, "grain": {"amount": 0.06},
    },
    "noir": {
        "grade": {"saturation": 0.0, "contrast": 1.3},
        "glow": {"threshold": 0.8, "radius": 5, "intensity": 0.2},
        "vignette": {"amount": 0.45}, "grain": {"amount": 0.05},
    },
    "psychedelic": {
        "grade": {"saturation": 1.8, "contrast": 1.2, "exposure": 0.1},
        "chroma": {"amount": 6.0}, "glow": {"threshold": 0.5, "radius": 14, "intensity": 0.8},
        "grain": {"amount": 0.02},
    },
}


def _mul(v, k):    # scale a "multiplier" param (neutral 1.0) by intensity k
    return 1.0 + (v - 1.0) * k


def apply_look(image, preset="neon", intensity=1.0, seed=0):
    """Apply a named VJ look. intensity 0 = off, 1 = preset, >1 = stronger."""
    if preset == "none" or intensity <= 0.0:
        return image
    if preset not in _PRESETS:
        raise ValueError(f"unknown preset {preset!r}, pick from {LOOK_PRESETS}")
    p = _PRESETS[preset]
    k = float(intensity)
    out = image
    if "grade" in p:
        g = p["grade"]
        out = color_grade(
            out, exposure=g.get("exposure", 0.0) * k, contrast=_mul(g.get("contrast", 1.0), k),
            saturation=_mul(g.get("saturation", 1.0), k), temperature=g.get("temperature", 0.0) * k,
            tint=g.get("tint", 0.0) * k, lift=g.get("lift", 0.0) * k,
            gamma=_mul(g.get("gamma", 1.0), k), gain=_mul(g.get("gain", 1.0), k),
        )
    if "chroma" in p:
        out = chroma_shift(out, amount=p["chroma"].get("amount", 0.0) * k)
    if "glow" in p:
        gl = p["glow"]
        out = glow(out, threshold=gl.get("threshold", 0.7), radius=gl.get("radius", 8),
                   intensity=gl.get("intensity", 0.0) * k)
    if "vignette" in p:
        out = vignette(out, amount=min(0.95, p["vignette"].get("amount", 0.0) * k))
    if "grain" in p:
        out = grain(out, amount=p["grain"].get("amount", 0.0) * k, seed=seed)
    return out
