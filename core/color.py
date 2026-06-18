"""
Color coherence for Difforum - anti-drift across a feedback animation.

The classic Deforum feedback loop slowly drifts in color/brightness as each
frame is re-diffused from the last. Matching every frame's statistics back to an
anchor frame keeps the palette stable.

Two modes:
  "lab"  match mean/std in CIELAB - preserves perceptual luminance/chroma, the
         better default (what pro color-match tools do).
  "rgb"  match mean/std per RGB channel - cheaper, slightly more aggressive.

Pure torch, testable. Images are [H,W,C] or [B,H,W,C] float in 0..1.
"""

from __future__ import annotations

import torch

COLOR_MODES = ("lab", "rgb", "none")


def _rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    mask = rgb > 0.04045
    lin = torch.where(mask, ((rgb + 0.055) / 1.055) ** 2.4, rgb / 12.92)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = r * 0.2126 + g * 0.7152 + b * 0.0722
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883
    xyz = torch.stack([x, y, z], dim=-1)
    eps = 0.008856
    f = torch.where(xyz > eps, xyz.clamp(min=0) ** (1.0 / 3.0), 7.787 * xyz + 16.0 / 116.0)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return torch.stack([L, a, bb], dim=-1)


def _lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    L, a, bb = lab[..., 0], lab[..., 1], lab[..., 2]
    fy = (L + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - bb / 200.0
    f = torch.stack([fx, fy, fz], dim=-1)
    eps = 0.008856
    xyz = torch.where(f ** 3 > eps, f ** 3, (f - 16.0 / 116.0) / 7.787)
    x = xyz[..., 0] * 0.95047
    y = xyz[..., 1]
    z = xyz[..., 2] * 1.08883
    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570
    lin = torch.stack([r, g, b], dim=-1)
    mask = lin > 0.0031308
    rgb = torch.where(mask, 1.055 * lin.clamp(min=0) ** (1.0 / 2.4) - 0.055, 12.92 * lin)
    return rgb.clamp(0.0, 1.0)


def _match_stats(img: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    img_mean = img.mean(dim=(1, 2), keepdim=True)
    img_std = img.std(dim=(1, 2), keepdim=True) + 1e-5
    ref_mean = ref.mean(dim=(1, 2), keepdim=True)
    ref_std = ref.std(dim=(1, 2), keepdim=True) + 1e-5
    return (img - img_mean) / img_std * ref_std + ref_mean


def match_color(
    image: torch.Tensor,
    reference: torch.Tensor,
    strength: float = 1.0,
    mode: str = "lab",
) -> torch.Tensor:
    """
    Match `image`'s color statistics to `reference`'s.

    strength in [0,1] blends original (0) to fully matched (1). `mode` is
    "lab" (default, perceptual), "rgb", or "none" (passthrough). Reference may
    be a single image broadcast over an image batch.
    """
    if mode == "none" or strength <= 0.0:
        return image
    if mode not in COLOR_MODES:
        raise ValueError(f"unknown color mode {mode!r}, pick from {COLOR_MODES}")

    squeezed = image.dim() == 3
    img = image.unsqueeze(0) if squeezed else image
    ref = reference.unsqueeze(0) if reference.dim() == 3 else reference

    if mode == "lab":
        img_l = _rgb_to_lab(img)
        ref_l = _rgb_to_lab(ref)
        matched = _lab_to_rgb(_match_stats(img_l, ref_l))
    else:  # rgb
        matched = _match_stats(img, ref)

    out = (img + (matched - img) * float(strength)).clamp(0.0, 1.0)
    return out[0] if squeezed else out
