"""
Storyboard: a diffusion-free dry run of the whole clip.

Runs the exact camera-warp feedback chain the Feedback Sampler uses, but skips
every sampler / VAE call. What comes out is not a render - it is the *motion
skeleton*: where the camera goes, how fast the frame drifts, when the strength
curve peaks. Cheap enough (a few hundred ms) to sit in the edit loop, so the
direction gets fixed before any diffusion time is spent.
"""

from __future__ import annotations

import math

import torch

from .symmetry import apply_symmetry
from .warp import warp_2d, warp_3d


def _z_angle_deg(delta: torch.Tensor) -> float:
    """Recover the in-plane rotation (degrees) from a 4x4 transform."""
    return math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0])))


def _resize(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if image.dim() == 3:
        image = image.unsqueeze(0)
    b, h, w, c = image.shape
    if h == height and w == width:
        return image
    chw = image.permute(0, 3, 1, 2)
    out = torch.nn.functional.interpolate(
        chw, size=(height, width), mode="bilinear", align_corners=False
    )
    return out.permute(0, 2, 3, 1)


def simulate(
    init_image: torch.Tensor,
    camera,
    max_frames: int,
    width: int,
    height: int,
    depth: torch.Tensor | None = None,
    border: str = "reflection",
    symmetry: str = "none",
    symmetry_segments: int = 6,
    translation_scale: float = 1.0,
    near: float = 1.0,
    far: float = 100.0,
    invert_depth: bool = False,
) -> list[torch.Tensor]:
    """Accumulate the camera warp frame by frame. Returns a list of [1,H,W,3]."""
    prev = _resize(init_image, width, height)[:1]
    frames = [prev]

    depth_b = None
    if depth is not None:
        depth_b = _resize(depth, width, height)
        if depth_b.shape[-1] == 3:
            depth_b = depth_b.mean(dim=-1, keepdim=True)

    n = min(int(max_frames), len(camera.deltas))
    for f in range(1, n):
        delta = torch.as_tensor(camera.deltas[f], dtype=torch.float32)

        if camera.mode == "3d" and depth_b is not None:
            d = depth_b[..., 0].to(prev.device)
            out, _ = warp_3d(
                prev, d, delta,
                fov_deg=float(camera.fov[f]),
                near=float(near), far=float(far),
                invert_depth=bool(invert_depth),
                translation_scale=float(translation_scale),
            )
        else:
            out, _ = warp_2d(
                prev,
                float(delta[0, 3]), float(delta[1, 3]),
                _z_angle_deg(delta),
                float(camera.zoom[f]),
                padding_mode=border,
            )

        if symmetry != "none":
            out = apply_symmetry(out, mode=symmetry, segments=int(symmetry_segments))

        frames.append(out)
        prev = out

    return frames


def drift_curve(frames: list[torch.Tensor]) -> list[float]:
    """Mean absolute change between consecutive frames.

    A flat curve means the camera is barely moving (the sampler will have
    nothing to bite on); a spike means the warp is outrunning the diffusion and
    the frame will smear. Useful as a pacing readout.
    """
    out = [0.0]
    for a, b in zip(frames, frames[1:]):
        out.append(float((b - a).abs().mean()))
    return out


def contact_sheet(
    frames: list[torch.Tensor],
    columns: int = 6,
    cell_width: int = 192,
    labels: list[str] | None = None,
    gap: int = 4,
    bg: float = 0.08,
) -> torch.Tensor:
    """Tile frames into one image [1,H,W,3], row-major, with optional labels."""
    if not frames:
        return torch.zeros(1, 64, 64, 3)

    b, h, w, c = frames[0].shape
    cw = int(cell_width)
    ch = max(1, int(round(cw * h / max(w, 1))))
    label_h = 12 if labels else 0

    cols = max(1, int(columns))
    rows = math.ceil(len(frames) / cols)

    sheet_w = cols * cw + (cols + 1) * gap
    sheet_h = rows * (ch + label_h) + (rows + 1) * gap
    sheet = torch.full((1, sheet_h, sheet_w, 3), float(bg))

    for i, frame in enumerate(frames):
        r, col = divmod(i, cols)
        cell = _resize(frame, cw, ch)[0].clamp(0.0, 1.0)
        y = gap + r * (ch + label_h + gap)
        x = gap + col * (cw + gap)
        sheet[0, y:y + ch, x:x + cw, :] = cell.to(sheet.dtype)
        if labels and i < len(labels):
            _stamp(sheet, labels[i], x + 2, y + ch + 2)

    return sheet


# ---------------------------------------------------------------------------
# Tiny 3x5 bitmap font - enough for frame numbers and shot names, no PIL needed
# ---------------------------------------------------------------------------

_FONT = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "010", "010"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "f": ("111", "100", "110", "100", "100"),
    "s": ("111", "100", "111", "001", "111"),
    ".": ("000", "000", "000", "000", "010"),
    "-": ("000", "000", "111", "000", "000"),
    ":": ("000", "010", "000", "010", "000"),
    " ": ("000", "000", "000", "000", "000"),
}


def _stamp(sheet: torch.Tensor, text: str, x0: int, y0: int, scale: int = 2) -> None:
    """Draw `text` into the sheet in place (white on whatever is there)."""
    _, H, W, _ = sheet.shape
    cx = x0
    for chn in str(text).lower():
        glyph = _FONT.get(chn)
        if glyph is None:
            cx += 4 * scale
            continue
        for ry, row in enumerate(glyph):
            for rx, bit in enumerate(row):
                if bit != "1":
                    continue
                ys, xs = y0 + ry * scale, cx + rx * scale
                ye, xe = min(ys + scale, H), min(xs + scale, W)
                if ys < H and xs < W and ye > ys and xe > xs:
                    sheet[0, ys:ye, xs:xe, :] = 1.0
        cx += 4 * scale
