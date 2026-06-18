"""
Warp engine for Difforum - the heart of the Deforum look.

Two warps, both producing a warped image + an occlusion (hole) mask. The mask
marks pixels the warp could not fill (newly revealed areas); re-diffusing only
those is the classic Deforum feedback mechanism.

    warp_2d  affine: translation + rotation (angle) + zoom about center.
             The original 2D Deforum feel. No depth needed. Inverse sampling,
             so it is hole-free except at the edges that scroll into view.

    warp_3d  perspective reprojection using a depth map and a 4x4 camera delta.
             The "fly through the scene" parallax look. Forward splat with a
             z-buffer -> disocclusion holes appear in the mask.

torch-based so it runs fast on GPU inside ComfyUI; tests exercise it on CPU
with tiny tensors. Images are [H,W,3] or [B,H,W,3] float in 0..1 (ComfyUI's
IMAGE layout). Depth is [H,W] / [B,H,W] / [B,H,W,1].
"""

from __future__ import annotations

import math

import torch


def _focal(width: int, fov_deg: float) -> float:
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def _as_bhwc(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.dim() == 3:
        return image.unsqueeze(0), True
    return image, False


def warp_2d(
    image: torch.Tensor,
    translation_x: float = 0.0,
    translation_y: float = 0.0,
    angle: float = 0.0,
    zoom: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    2D affine warp (the classic Deforum 2D mode), inverse-sampled.

    translation in pixels, angle in degrees (CCW), zoom > 1 magnifies. Returns
    (warped [.,H,W,3], mask [.,H,W,1]) where mask=1 means a valid sampled pixel.
    """
    img, squeezed = _as_bhwc(image)
    b, h, w, c = img.shape
    dev, dt = img.device, img.dtype

    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    a = math.radians(angle)
    ca, sa = math.cos(a), math.sin(a)
    s = max(zoom, 1e-6)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev, dtype=dt),
        torch.arange(w, device=dev, dtype=dt),
        indexing="ij",
    )
    # forward: p_out = S*R*(p_in - c) + c + t  ->  invert to find p_in
    xo = xs - cx - translation_x
    yo = ys - cy - translation_y
    xr = (ca * xo + sa * yo) / s
    yr = (-sa * xo + ca * yo) / s
    src_x = xr + cx
    src_y = yr + cy

    gx = (src_x / (w - 1)) * 2.0 - 1.0
    gy = (src_y / (h - 1)) * 2.0 - 1.0
    grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(b, h, w, 2)

    chw = img.permute(0, 3, 1, 2)
    warped = torch.nn.functional.grid_sample(
        chw, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    # validity: where the source coord fell inside the image
    inside = (
        (src_x >= 0) & (src_x <= w - 1) & (src_y >= 0) & (src_y <= h - 1)
    ).to(dt)
    mask = inside.unsqueeze(0).expand(b, h, w).unsqueeze(-1)
    out = warped.permute(0, 2, 3, 1)
    if squeezed:
        out, mask = out[0], mask[0]
    return out, mask


def _fill_holes(img: torch.Tensor, filled: torch.Tensor, iters: int = 4) -> torch.Tensor:
    """Fill empty (filled==0) pixels from nearest filled neighbours.

    Cleans the salt-and-pepper holes a forward splat leaves when the scene
    expands, without touching the occlusion mask. img [B,H,W,C], filled [B,H,W].
    Small `iters` so only thin gaps close; large disocclusions stay for the
    diffusion step to repaint.
    """
    out = img.clone()
    have = filled.clone()
    for _ in range(int(iters)):
        holes = have < 0.5
        if not bool(holes.any()):
            break
        acc = torch.zeros_like(out)
        wsum = torch.zeros_like(have)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            acc += torch.roll(out, shifts=(dy, dx), dims=(1, 2)) * \
                torch.roll(have, shifts=(dy, dx), dims=(1, 2)).unsqueeze(-1)
            wsum += torch.roll(have, shifts=(dy, dx), dims=(1, 2))
        fill = holes & (wsum > 0)
        newv = acc / wsum.clamp(min=1e-6).unsqueeze(-1)
        out = torch.where(fill.unsqueeze(-1), newv, out)
        have = torch.where(fill, torch.ones_like(have), have)
    return out


def _depth_to_z(depth: torch.Tensor, near: float, far: float, invert: bool) -> torch.Tensor:
    d = depth.clamp(0.0, 1.0)
    if invert:
        d = 1.0 - d
    # d=1 -> near (closest), d=0 -> far
    return near + (1.0 - d) * (far - near)


def warp_3d(
    image: torch.Tensor,
    depth: torch.Tensor,
    transform: torch.Tensor,
    fov_deg: float = 40.0,
    near: float = 1.0,
    far: float = 100.0,
    invert_depth: bool = False,
    translation_scale: float = 1.0,
    fill_holes: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Perspective warp via depth + a 4x4 scene-point transform.

    `transform` is applied to unprojected scene points (P' = R@P + T). A
    positive translation_z therefore pushes the scene away (camera dollies
    back); flip the sign upstream for "into the scene". Forward splat with a
    z-buffer: nearest point wins, unfilled pixels are holes (mask=0).

    `translation_scale` multiplies the translation part of the transform so the
    motion magnitude can be tuned to the depth range (depth estimators output
    relative/disparity values, not metric units - this is the calibration knob).

    Returns (warped [.,H,W,3], mask [.,H,W,1]).
    """
    img, squeezed = _as_bhwc(image)
    b, h, w, c = img.shape
    dev, dt = img.device, img.dtype

    if depth.dim() == 2:
        depth = depth.unsqueeze(0)
    if depth.dim() == 4:
        depth = depth[..., 0]
    depth = depth.expand(b, h, w)

    f = _focal(w, fov_deg)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    R = transform[:3, :3].to(dev, dt)
    T = transform[:3, 3].to(dev, dt) * float(translation_scale)

    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev, dtype=dt),
        torch.arange(w, device=dev, dtype=dt),
        indexing="ij",
    )

    out = torch.zeros_like(img)
    mask = torch.zeros((b, h, w), device=dev, dtype=dt)

    for i in range(b):
        z = _depth_to_z(depth[i], near, far, invert_depth)
        X = (xs - cx) * z / f
        Y = (ys - cy) * z / f
        P = torch.stack([X, Y, z], dim=-1).reshape(-1, 3)  # [N,3]
        Pp = P @ R.T + T                                    # [N,3]
        Zp = Pp[:, 2].clamp(min=1e-4)
        up = f * Pp[:, 0] / Zp + cx
        vp = f * Pp[:, 1] / Zp + cy

        ui = up.round().long()
        vi = vp.round().long()
        valid = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

        ui, vi, Zp = ui[valid], vi[valid], Zp[valid]
        cols = img[i].reshape(-1, c)[valid]
        dest = vi * w + ui

        # paint far -> near so the nearest point ends up on top (z-buffer)
        order = torch.argsort(Zp, descending=True)
        dest_o, cols_o = dest[order], cols[order]

        flat = out[i].reshape(-1, c)
        flat[dest_o] = cols_o
        mflat = mask[i].reshape(-1)
        mflat[dest_o] = 1.0

    # clean speckle holes in the colour (keep `mask` as the true occlusion)
    if fill_holes:
        out = _fill_holes(out, mask)

    mask = mask.unsqueeze(-1)
    if squeezed:
        out, mask = out[0], mask[0]
    return out, mask
