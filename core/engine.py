"""
The Difforum feedback engine - one implementation shared by the Feedback
Sampler, the Live Sampler and the Storyboard.

Per frame:  warp (camera) -> symmetry -> detail guard -> [diffuse] -> colour.

What it fixes over the 0.x loop:

* **Depth follows the image.** The depth map is re-projected with every 3D
  warp, so parallax stays attached to what is on screen instead of to frame 0.
* **No silent freeze without depth.** A 3D camera with no depth map runs a
  pseudo-3D affine (dolly -> zoom, orbit/pan -> shift, roll -> roll) and the
  run report says so.
* **Pans agree between 2D and 3D.** Lateral 3D translation is measured in
  pixels at the reference depth, the same unit the 2D warp uses.
* **Cadence without pops.** In-between frames are a crossfade of the previous
  key warped forward and the next key warped *back*, the way Deforum does it.
* **Holes get repainted.** The occlusion mask from the warp gets extra noise, so
  the sampler invents new content where the camera revealed the unknown.
* **Colour anchor modes.** `first` (0.x behaviour), `scene` (re-anchors at every
  prompt keyframe and lets go during the transition), `rolling` (previous key)
  or `none`.

The engine is torch-only and knows nothing about ComfyUI: diffusion is a
callback, so the whole loop is testable with a stub.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Iterator

import torch

from .color import match_color
from .detail import add_noise, sharpen
from .fx import FxRunner
from .symmetry import apply_symmetry
from .warp import (
    _focal, affine_2d, pseudo_3d_params, warp_2d, warp_3d, warp_affine,
)

ANCHOR_MODES = ("scene", "first", "rolling", "none")
DEPTH_TRACKING = ("follow", "static")
BORDER_MODES = ("reflection", "border", "zeros")


def resize_bhwc(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """[H,W,C] or [B,H,W,C] -> [B,height,width,C] (bilinear)."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    _b, h, w, _c = image.shape
    if h == height and w == width:
        return image
    chw = image.permute(0, 3, 1, 2)
    chw = torch.nn.functional.interpolate(
        chw, size=(height, width), mode="bilinear", align_corners=False)
    return chw.permute(0, 2, 3, 1)


def depth_to_single(depth: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Any IMAGE/MASK-ish depth -> [B,H,W] in 0..1 at the render size."""
    d = depth
    if d.dim() == 2:
        d = d.unsqueeze(0)
    if d.dim() == 3:          # MASK [B,H,W]
        d = d.unsqueeze(-1)
    d = resize_bhwc(d, width, height)
    if d.shape[-1] >= 3:
        d = d[..., :3].mean(dim=-1)
    else:
        d = d[..., 0]
    return d.clamp(0.0, 1.0)


def z_angle_deg(delta) -> float:
    return math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0])))


@dataclass
class EngineConfig:
    width: int
    height: int
    border: str = "reflection"
    symmetry: str = "none"
    symmetry_segments: int = 6
    sharpen: float = 0.2
    noise: float = 0.02
    hole_noise: float = 0.25          # extra noise where the warp revealed holes
    cadence: int = 1
    color_coherence: float = 0.8
    color_mode: str = "lab"
    anchor_mode: str = "scene"
    near: float = 1.0
    far: float = 100.0
    invert_depth: bool = False
    translation_scale: float = 1.0
    depth_tracking: str = "follow"
    reference_depth: float = 0.5      # where pseudo-3D / lateral units are measured
    seed: int = 0


@dataclass
class RunReport:
    mode: str = "2d"                  # what actually ran: 2d / 3d / pseudo3d
    keys: int = 0
    tweens: int = 0
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        head = f"engine: {self.mode}  keys={self.keys}  tweens={self.tweens}"
        return "\n".join([head, *[f"  {n}" for n in self.notes]])


class FeedbackEngine:
    """Stateful camera warp chain. `step()` advances one frame."""

    def __init__(self, camera, cfg: EngineConfig, depth: torch.Tensor | None = None,
                 fx: FxRunner | None = None):
        self.camera = camera
        self.cfg = cfg
        self.fx = fx or FxRunner()
        self.report = RunReport()
        self.depth_log: dict[int, torch.Tensor] = {}
        self.depth_seq = None        # per-frame depth batch (video depth)
        self.depth = None            # current tracked depth [1,H,W]
        if depth is not None:
            d = depth_to_single(depth, cfg.width, cfg.height)
            if d.shape[0] > 1:
                self.depth_seq = d
                self.depth = d[:1]
            else:
                self.depth = d
        self.use_depth = camera.mode == "3d" and self.depth is not None and not camera.flat
        if camera.mode == "3d":
            self.report.mode = "3d" if self.use_depth else "pseudo3d"
            if self.depth is None and not camera.flat:
                self.report.notes.append(
                    "3D camera without a depth map: ran the pseudo-3D fallback "
                    "(dolly -> zoom, orbit -> pan). Connect a depth map for real parallax."
                )
        else:
            self.report.mode = "2d"

    # -- geometry ------------------------------------------------------------

    def _lateral_scale(self, fov: float) -> float:
        """3D x/y translation units -> pixels at the reference depth."""
        c = self.cfg
        z_ref = c.near + (1.0 - c.reference_depth) * (c.far - c.near)
        return z_ref / _focal(c.width, fov)

    def _delta3d(self, f: int, fov: float) -> torch.Tensor:
        d = torch.as_tensor(self.camera.deltas[f], dtype=torch.float32).clone()
        lat = self._lateral_scale(fov)
        d[0, 3] *= lat
        d[1, 3] *= lat
        return d

    def _affine_for(self, f: int) -> torch.Tensor:
        """3x3 pixel-space map for frame f in 2D / pseudo-3D."""
        c = self.cfg
        zoom = float(self.camera.zoom[f])
        fov = float(self.camera.fov[f])
        if self.camera.mode == "3d":
            tx, ty, ang, z = pseudo_3d_params(
                self._delta3d(f, fov), fov, c.width, c.near, c.far,
                c.translation_scale, c.reference_depth)
            zoom = zoom * z
        else:
            delta = self.camera.deltas[f]
            tx, ty, ang = float(delta[0][3]), float(delta[1][3]), z_angle_deg(delta)
        return affine_2d(tx, ty, ang, zoom, c.width, c.height)

    def _depth_for(self, f: int):
        if self.depth_seq is not None:
            return self.depth_seq[min(f, self.depth_seq.shape[0] - 1)].unsqueeze(0)
        return self.depth

    def warp(self, img: torch.Tensor, f: int, track_depth: bool = True):
        """Warp one frame forward by the camera step at frame f.

        Returns (image, hole_mask) where hole_mask is 1 where the warp revealed
        pixels it could not fill."""
        c = self.cfg
        fov = float(self.camera.fov[f])
        depth = self._depth_for(f)
        if self.use_depth:
            out, m, new_d = warp_3d(
                img, depth.to(img.device), self._delta3d(f, fov), fov_deg=fov,
                near=c.near, far=c.far, invert_depth=c.invert_depth,
                translation_scale=c.translation_scale, return_depth=True)
            zoom = float(self.camera.zoom[f])
            if abs(zoom - 1.0) > 1e-6:   # 3D mode still honours the zoom channel
                out, m2 = warp_2d(out, 0.0, 0.0, 0.0, zoom, padding_mode=c.border)
                m = m * m2
                new_d, _ = warp_2d(new_d.unsqueeze(-1), 0.0, 0.0, 0.0, zoom,
                                   padding_mode="border")
                new_d = new_d[..., 0]
            if track_depth and c.depth_tracking == "follow" and self.depth_seq is None:
                self.depth = new_d.detach().to("cpu")
            if track_depth:
                self.depth_log[f] = (new_d if c.depth_tracking == "follow" else depth).detach().to("cpu")
        else:
            out, m = warp_affine(img, self._affine_for(f), padding_mode=c.border)
        return out, 1.0 - m

    def warp_back(self, img: torch.Tensor, frm: int, to: int, depth=None) -> torch.Tensor:
        """Carry an image rendered at frame `frm` back to earlier frame `to`
        (inverse of the camera steps to+1..frm). Used for cadence tweens."""
        c = self.cfg
        if frm <= to:
            return img
        if self.use_depth and depth is not None:
            acc = torch.eye(4, dtype=torch.float64)
            for f in range(to + 1, frm + 1):
                fov = float(self.camera.fov[f])
                acc = self._delta3d(f, fov).to(torch.float64) @ acc
            inv = torch.linalg.inv(acc).to(torch.float32)
            out, _m = warp_3d(
                img, depth.to(img.device), inv, fov_deg=float(self.camera.fov[frm]),
                near=c.near, far=c.far, invert_depth=c.invert_depth,
                translation_scale=c.translation_scale)
            return out
        acc = torch.eye(3, dtype=torch.float64)
        for f in range(to + 1, frm + 1):
            acc = self._affine_for(f) @ acc
        out, _m = warp_affine(img, torch.linalg.inv(acc), padding_mode=c.border)
        return out

    def iter_from_anchor(self, anchor: torch.Tensor, n: int):
        """Warp ONE anchor image along the accumulated camera path (no
        feedback, no blur build-up) - the guide-frame generator for video
        models. Yields (image [1,H,W,3], valid_mask [1,H,W,1]) per frame."""
        c = self.cfg
        img = resize_bhwc(anchor, c.width, c.height)[:1, ..., :3].float()
        depth0 = self.depth
        use3d = self.use_depth
        acc3 = torch.eye(4, dtype=torch.float64)
        acc2 = torch.eye(3, dtype=torch.float64)
        zoom_cum = 1.0
        for f in range(n):
            if f > 0:
                if use3d:
                    fov_f = float(self.camera.fov[f])
                    acc3 = self._delta3d(f, fov_f).to(torch.float64) @ acc3
                    zoom_cum *= float(self.camera.zoom[f])
                else:
                    acc2 = self._affine_for(f) @ acc2
            if f == 0:
                yield img, torch.ones_like(img[..., :1])
                continue
            if use3d:
                out, m = warp_3d(img, depth0, acc3.to(torch.float32),
                                 fov_deg=float(self.camera.fov[f]), near=c.near, far=c.far,
                                 invert_depth=c.invert_depth,
                                 translation_scale=c.translation_scale)
                if abs(zoom_cum - 1.0) > 1e-6:
                    out, m2 = warp_2d(out, 0.0, 0.0, 0.0, zoom_cum, padding_mode=c.border)
                    m = m * m2
            else:
                out, m = warp_affine(img, acc2, padding_mode=c.border)
            yield out.clamp(0.0, 1.0), m

    # -- per-frame pixel chain -------------------------------------------------

    def prepare(self, img: torch.Tensor, f: int, is_key: bool):
        """Warp + symmetry + detail guard (+ noise on key frames)."""
        c = self.cfg

        def chain(x):
            out, holes = self.warp(x, f)
            if c.symmetry != "none":
                out = apply_symmetry(out, mode=c.symmetry, segments=int(c.symmetry_segments))
            if c.sharpen > 0.0:
                out = sharpen(out, c.sharpen)
            if is_key and c.noise > 0.0:
                out = add_noise(out, c.noise, seed=c.seed + f)
            if is_key and c.hole_noise > 0.0 and float(holes.max()) > 0.0:
                extra = add_noise(out, c.hole_noise, seed=c.seed + 7919 * (f + 1))
                out = out * (1.0 - holes) + extra * holes
            return out

        return self.fx(chain, img).clamp(0.0, 1.0)


def _anchor_strength(cfg: EngineConfig, prompt_track, f: int) -> float:
    """Colour-lock strength at frame f. In `scene` mode the lock releases while
    the prompt is travelling, so the palette can follow the new scene."""
    s = float(cfg.color_coherence)
    if cfg.anchor_mode != "scene" or prompt_track is None:
        return s
    # lock fades out as the prompt travels towards the next scene; the key
    # rendered once that scene is reached becomes the new anchor
    return s * (1.0 - prompt_track.transition_weight(f))


def iter_feedback(
    engine: FeedbackEngine,
    init_image: torch.Tensor,
    n_frames: int,
    diffuse: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
    prompt_track=None,
    start_frame: int = 0,
    end_frame: int = 0,
    on_progress: Callable[[int], None] | None = None,
    pre_warp: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
) -> Iterator[tuple[int, torch.Tensor]]:
    """Run the loop, yielding (frame_index, [1,H,W,3]) in order.

    `diffuse(image, f)` re-diffuses a key frame (img2img); None = storyboard
    mode (camera only). `pre_warp(prev, f)` may alter the previous frame
    before it is warped (live camera input). With cadence > 1, in-between frames are buffered until
    the next key is known, then crossfaded, so output lags by `cadence` frames.
    """
    cfg = engine.cfg
    cad = max(1, int(cfg.cadence))
    prev = resize_bhwc(init_image, cfg.width, cfg.height)[:1, ..., :3].float()
    anchor = prev.clone()
    f0 = max(0, int(start_frame))
    f1 = min(n_frames, int(end_frame)) if int(end_frame) > 0 else n_frames

    if engine.depth is not None:
        engine.depth_log[f0] = engine.depth
    yield f0, prev
    if on_progress:
        on_progress(f0)

    last_key_f = f0
    pending: list[tuple[int, torch.Tensor]] = []
    scene_idx = prompt_track.scene_index(f0) if prompt_track is not None else 0

    for f in range(f0 + 1, f1):
        is_key = diffuse is not None and (cad == 1 or (f - f0) % cad == 0 or f == f1 - 1)
        if pre_warp is not None:
            prev = pre_warp(prev, f)
        warped = engine.prepare(prev, f, is_key=is_key)

        if not is_key:
            prev = warped
            if diffuse is None or cad == 1:
                yield f, prev
                if on_progress:
                    on_progress(f)
            else:
                pending.append((f, warped))
            continue

        image = diffuse(warped, f)[:1, ..., :3].float().cpu()
        engine.report.keys += 1

        # colour anchoring
        if cfg.color_mode != "none" and cfg.anchor_mode != "none":
            if cfg.anchor_mode == "scene" and prompt_track is not None:
                si = prompt_track.scene_index(f)
                if si != scene_idx:
                    scene_idx = si
                    anchor = image.clone()
            strength = _anchor_strength(cfg, prompt_track, f)
            if strength > 0.0:
                image = match_color(image, anchor, strength=strength, mode=cfg.color_mode)
            if cfg.anchor_mode == "rolling":
                anchor = image.clone()

        # cadence: crossfade the buffered tweens between the two keys
        if pending:
            span = f - last_key_f
            depth = engine.depth
            for (tf, fwd) in pending:
                back = engine.warp_back(image, f, tf, depth=depth)
                u = (tf - last_key_f) / span
                w = u * u * (3.0 - 2.0 * u)
                engine.report.tweens += 1
                yield tf, (fwd * (1.0 - w) + back * w).clamp(0.0, 1.0)
                if on_progress:
                    on_progress(tf)
            pending = []

        prev = image
        last_key_f = f
        yield f, prev
        if on_progress:
            on_progress(f)

    for (tf, fwd) in pending:   # defensive: the last frame is always a key
        yield tf, fwd


def key_pull(keys: list[tuple[int, torch.Tensor]], pull: float = 0.65, approach: int = 12):
    """pre_warp hook that steers the feedback toward image keyframes.

    `keys` = [(frame, image [1,H,W,3])]. Over the `approach` frames before a key
    the previous frame is blended toward the key image (quadratic ramp), fully
    reaching `pull` on the key frame itself, so a Deforum-style travel passes
    through the pictures you chose instead of drifting freely."""
    keys = sorted(keys, key=lambda k: k[0])
    ap = max(1, int(approach))

    def hook(prev: torch.Tensor, f: int) -> torch.Tensor:
        for idx, img in keys:
            if idx < f:
                continue
            if idx - f >= ap:
                break
            w = 1.0 - (idx - f) / ap
            a = float(pull) * w * w
            return prev * (1.0 - a) + img.to(prev.device, prev.dtype) * a
        return prev

    return hook
