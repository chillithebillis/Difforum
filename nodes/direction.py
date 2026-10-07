"""Direction nodes: the Director timeline, keyframed/expression cameras and the
Storyboard dry run."""

from __future__ import annotations

import json

from ..core.camera import CAMERA_MODES, build_camera
from ..core.camera_keys import keys_to_axis_values, lens_note, parse_camera_keys
from ..core.camera_presets import flat_presets, needs_depth
from ..core.direction import (
    LOOKS, DirectionBundle, build_direction, default_timeline, describe_timed, parse_timeline,
)
from ..core.script import EXTERNAL_MODES, merge_timelines, script_to_timeline
from ..core.loop import LOOP_MODES, close_axis_values, tile_laps
from ..core.schedule import Schedule, build_schedule
from ._common import (
    AUDIO, CAMERA, CAT_DIRECT, PARAMS, PROMPT, SCHEDULE, audio_vars,
)

DIRECTION = "DIFFORUM_DIRECTION"


def encode_prompt_track(clip, keyframes, frames, easing="ease_in_out"):
    from ..core.prompt import PromptTrack
    encoded = [clip.encode_from_tokens_scheduled(clip.tokenize(t)) for _, t in keyframes]
    return PromptTrack(encoded, keyframes, frames, easing)


def _track(axes: dict, lens: list, n: int, mode: str, moves: list):
    """Camera track for the chosen mode. Moves through space in a 2d shot are
    kept as a 3D track rendered flat, so they become pseudo-3D (dolly -> zoom,
    orbit -> pan) instead of being dropped."""
    values = dict(axes)
    values["fov"] = lens
    if mode == "2d" and any(needs_depth(m) for m in moves):
        cam = build_camera(values, max_frames=n, mode="3d", fov=lens[0])
        cam.flat = True
        return cam
    return build_camera(values, max_frames=n, mode=mode, fov=lens[0])


def _fit(series: list, n: int, pad):
    series = list(series)
    if len(series) < n:
        series.extend([series[-1] if series else pad] * (n - len(series)))
    return series[:n]


IMAGE_SLOTS = 6
IMAGES_AT = ("scene starts", "Keys markers", "spread evenly")


def _cover(img, w: int, h: int):
    """[1,H,W,C] -> [1,h,w,3], scaled to cover the canvas and centre-cropped."""
    import torch.nn.functional as F
    chw = img[..., :3].float().permute(0, 3, 1, 2)
    ih, iw = chw.shape[-2:]
    if (ih, iw) == (h, w):
        return img[..., :3].float()
    k = max(w / iw, h / ih)
    nw, nh = max(w, round(iw * k)), max(h, round(ih * k))
    rs = F.interpolate(chw, size=(nh, nw), mode="bilinear", align_corners=False, antialias=True)
    y, x = (nh - h) // 2, (nw - w) // 2
    return rs[:, :, y:y + h, x:x + w].permute(0, 2, 3, 1)


def _pin_images(images, slots, images_at, d, params):
    """Director image inputs -> (keyframes at the canvas size, 'f0,f1,...', info lines)."""
    import torch
    pics = [] if images is None else [images[i:i + 1] for i in range(images.shape[0])]
    for i in range(1, IMAGE_SLOTS + 1):
        im = slots.get(f"image_{i}")
        if im is not None:
            pics.extend(im[j:j + 1] for j in range(im.shape[0]))
    if not pics:
        return None, "", []
    n, fps = int(params["max_frames"]), float(params["fps"])
    w, h = int(params["width"]), int(params["height"])
    if images_at.startswith("scene") and d.scenes:
        frames, where = [s["start"] for s in d.scenes], "scene starts"
    elif images_at.startswith("Keys") and d.keys:
        frames, where = [k["start"] for k in d.keys], "Keys markers"
    else:
        m = len(pics)
        frames, where = [round(i * (n - 1) / max(1, m - 1)) for i in range(m)], "spread evenly"
    count = min(len(frames), len(pics))
    frames = [max(0, min(n - 1, int(f))) for f in frames[:count]]
    keys = torch.cat([_cover(p, w, h) for p in pics[:count]]).clamp(0, 1)
    lines = [f"  {count} image(s) pinned to {where}: " + ", ".join(f"{f / fps:.2f}s" for f in frames)]
    if len(pics) != count or (where != "spread evenly" and count < len(frames)):
        lines.append(f"  ! {len(pics)} image(s) for the timeline's moments: using {count}")
    return keys, ",".join(map(str, frames)), lines


def _thumbs(keys, key_idx, size: int = 320):
    """Small JPEG previews of the pinned images for the timeline editor."""
    if keys is None:
        return []
    import base64
    import io

    import numpy as np
    from PIL import Image
    out = []
    for f, img in zip(str(key_idx).split(","), keys):
        arr = (img[..., :3].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        im = Image.fromarray(arr)
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70)
        out.append({"frame": int(f), "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
    return out


# ---------------------------------------------------------------------------
# Director
# ---------------------------------------------------------------------------

class DifforumDirector:
    """Direct the whole clip on a visual timeline.

    Three tracks, edited with the mouse: **Scenes** (prompt + mood), **Camera**
    (pick a move, set speed / amplitude / lens / easing / audio reaction) and
    **Energy** (the denoise curve - how much each moment is re-imagined). The
    preview panel plays the camera back with the same engine that renders it.

    One `direction` wire carries camera, strength, cfg and prompt travel into
    the Feedback Sampler or a video-model bridge. The separate outputs are
    there for custom graphs.

    Plug pictures into `images` / `image_1..6` and each one is pinned to a
    moment (the scene starts by default): the Feedback Sampler travels through
    them, H3 / LTX Guides anchor them, the Animatic shows them - all through
    the same wire. The **Script** button edits the whole timeline as text, one
    line per frame range.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": (PARAMS,),
                "timeline": ("STRING", {"multiline": True,
                             "default": json.dumps(default_timeline(120))}),
                "camera_mode": (list(CAMERA_MODES), {"default": "2d",
                                "tooltip": "3d = real parallax when a depth map reaches the sampler "
                                           "(pseudo-3D otherwise)."}),
                "look": (list(LOOKS), {"default": "cinematic",
                         "tooltip": "Render aesthetic. Feedback Sampler: colour lock, detail, grain and energy. "
                                    "H3 / LTX: a look sentence added to the prompt."}),
                "transition": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                               "tooltip": "How much of each camera block is spent easing into the next. 0 = hard cuts."}),
                "camera_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                 "tooltip": "Master multiplier on every move."}),
                "energy_bias": ("FLOAT", {"default": 0.0, "min": -0.3, "max": 0.3, "step": 0.01,
                                "tooltip": "Shifts the whole denoise curve up or down."}),
                "variation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                              "tooltip": "0 = exactly as drawn. Above 0, seeded nudges per block."}),
                "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "clip": ("CLIP", {"tooltip": "Connect to encode the scene prompts (prompt travel)."}),
                "audio": (AUDIO, {"tooltip": "From Audio Analyzer - enables per-block audio reactions."}),
                "timeline_in": ("STRING", {"forceInput": True}),
                "external": (list(EXTERNAL_MODES), {"default": EXTERNAL_MODES[0]}),
                "images": ("IMAGE",),
                **{f"image_{i}": ("IMAGE",) for i in range(1, IMAGE_SLOTS + 1)},
                "images_at": (list(IMAGES_AT), {"default": IMAGES_AT[0]}),
            },
        }

    RETURN_TYPES = (DIRECTION, CAMERA, SCHEDULE, PROMPT, "STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("direction", "camera", "strength", "prompts", "camera_text", "info", "keyframes",
                    "indices")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, params, timeline, camera_mode, look, transition, camera_scale,
            energy_bias, variation, variation_seed, clip=None, audio=None, timeline_in=None,
            external=EXTERNAL_MODES[0], images=None, images_at=IMAGES_AT[0], **image_slots):
        n = int(params["max_frames"])
        fps = float(params["fps"])
        ext_notes = []
        if timeline_in is not None and str(timeline_in).strip():
            ext, notes = script_to_timeline(timeline_in, fps, n)
            timeline = json.dumps(merge_timelines(parse_timeline(timeline), ext, external))
            ext_notes = [f"  timeline from timeline_in ({external})", *(f"  ! {x}" for x in notes)]
        d = build_direction(
            timeline, n, fps, mode=camera_mode, camera_scale=camera_scale,
            strength_bias=energy_bias + LOOKS.get(look, LOOKS["cinematic"])["energy"], blend=transition, variation=variation,
            variation_seed=variation_seed, audio_curves=audio_vars(audio),
            width=int(params["width"]), height=int(params["height"]),
        )
        camera = _track(d.axes, d.lens, n, camera_mode, [b["move"] for b in d.camera_blocks])
        strength = Schedule(values=d.strength, fps=fps, source="director energy")
        cfg = Schedule(values=d.cfg, fps=fps, source="director guidance") if d.cfg else None
        prompts = encode_prompt_track(clip, d.prompts, n) if clip is not None else None

        keys, key_idx, key_lines = _pin_images(images, image_slots, images_at, d, params)
        bundle = DirectionBundle(params=params, camera=camera, strength=strength, cfg=cfg,
                                 prompts=prompts, direction=d, look_name=look,
                                 key_images=keys, key_indices=key_idx)
        info = "\n".join([
            f"[Difforum Director]  {n} frames  {n / fps:.2f}s  camera {camera_mode}  look {look}",
            *ext_notes,
            *key_lines,
            *d.summary,
            *([""] + [f"  ! {w}" for w in d.warnings] if d.warnings else []),
            "" if clip is not None else "  (connect a CLIP to get prompt travel on the direction wire)",
        ])
        timed = describe_timed(d.camera_blocks, n, fps)
        ui = {"text": [info]}
        if ext_notes:                    # the editor shows what was rendered
            ui["difforum_timeline"] = [timeline]
        ui["difforum_thumbs"] = _thumbs(keys, key_idx)
        return {
            "ui": ui,
            "result": (bundle, camera, strength, prompts,
                       d.camera_text + " " + bundle.look_prompt + "\n\n" + timed, info, keys, key_idx),
        }


# ---------------------------------------------------------------------------
# Camera (keys, text) and Camera (expressions)
# ---------------------------------------------------------------------------

_DEFAULT_KEYS = (
    "0: zoom_in 0.6 0.5 40 ease_in_out\n"
    "30: pan_right 1.0 0.8 40 ease_in_out\n"
    "60: spiral 1.4 1.1 55 ease_in\n"
    "90: zoom_out 0.5 0.45 40 ease_out"
)


class DifforumCamera:
    """Keyframed camera as text, one key per line:

        frame: move [speed] [intensity] [lens] [ease]

    Moves blend into each other (`transition` 1.0 = continuous crane move,
    0.0 = hard cuts). `loop_mode` makes the path periodic over `cycle_frames`
    so the clip loops without a crossfade. The Director timeline writes the
    same thing visually - use this node when you prefer typing.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": (PARAMS,),
                "keys": ("STRING", {"multiline": True, "default": _DEFAULT_KEYS}),
                "camera_mode": (list(CAMERA_MODES), {"default": "2d"}),
                "transition": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "loop_mode": (list(LOOP_MODES), {"default": "off"}),
                "cycle_frames": ("INT", {"default": 0, "min": 0, "max": 100000,
                                 "tooltip": "0 = whole clip is one cycle."}),
                "harmonics": ("INT", {"default": 3, "min": 1, "max": 16}),
            },
            "optional": {"audio": (AUDIO,)},
        }

    RETURN_TYPES = (CAMERA, "INT", "STRING")
    RETURN_NAMES = ("camera", "cycle_frames", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, params, keys, camera_mode, transition, speed, loop_mode, cycle_frames,
            harmonics, audio=None):
        total = int(params["max_frames"])
        fps = float(params["fps"])
        looping = loop_mode != "off"
        cyc = int(cycle_frames) if int(cycle_frames) > 0 else total
        cyc = max(8, min(cyc, total))
        span = cyc if looping else total

        values, lens, summary = keys_to_axis_values(
            keys, max_frames=span, fps=fps, extra_vars=audio_vars(audio), blend=transition)
        if abs(speed - 1.0) > 1e-6:
            for ax, series in values.items():
                values[ax] = ([1.0 + (v - 1.0) * speed for v in series] if ax == "zoom"
                              else [v * speed for v in series])

        notes = []
        if looping:
            values, residual = close_axis_values(values, mode=loop_mode,
                                                 keep_harmonics=int(harmonics))
            laps = max(1, total // cyc)
            values = tile_laps(values, laps)
            lens = list(lens) * laps
            err = max((abs(v - (1.0 if a == "zoom" else 0.0)) for a, v in residual.items()),
                      default=0.0)
            notes.append(f"  loop: {laps} lap(s) of {cyc}f, closure error {err:.1e}")
            if laps < 2:
                notes.append(f"  ! one lap cannot settle - set duration to {cyc * 3} frames "
                             "and keep the last lap with the Loop node")

        values = {ax: _fit(s, total, 1.0 if ax == "zoom" else 0.0) for ax, s in values.items()}
        moves = [k["move"] for k in parse_camera_keys(keys)]
        cam = _track(values, _fit(lens, total, 40.0), total, camera_mode, moves)

        depth_moves = sorted({m for m in moves if needs_depth(m)})
        if depth_moves:
            notes.append(f"  {', '.join(depth_moves)}: real parallax needs camera_mode=3d + a depth map; "
                         "otherwise they run as pseudo-3D (dolly->zoom, orbit->pan).")
            notes.append(f"  flat moves: {', '.join(flat_presets()[:10])}...")
        lo, hi = min(cam.fov), max(cam.fov)
        info = "\n".join([f"camera ({camera_mode}, transition {transition:g})",
                          f"  lens {lo:.0f}-{hi:.0f} deg ({lens_note(lo)})", summary, *notes])
        return (cam, cyc if looping else total, info)


_AXES = ("translation_x", "translation_y", "translation_z",
         "rotation_3d_x", "rotation_3d_y", "rotation_3d_z", "zoom")
_EXPR_DEFAULTS = {
    "translation_x": "0:(0)", "translation_y": "0:(0)", "translation_z": "0:(0)",
    "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)", "rotation_3d_z": "0:(0.3)",
    "zoom": "0:(1.0 + 0.004*sin(2*pi*t/48))",
}


class DifforumCameraExpr:
    """Deforum-style camera from math expressions, per axis.

    `0:(0), 60:(2*sin(2*pi*t/30))`, with `t` (frame), `s` (seconds) and any audio
    band (`low`, `beat`...) when audio is connected. Each axis also has an
    optional schedule socket (e.g. an Audio Curve) that overrides its text.
    Values are per-frame increments; zoom is a per-frame scale factor.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        req = {"params": (PARAMS,), "mode": (list(CAMERA_MODES), {"default": "2d"}),
               "fov": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 170.0, "step": 1.0})}
        for ax in _AXES:
            req[ax] = ("STRING", {"multiline": False, "default": _EXPR_DEFAULTS[ax]})
        opt = {"audio": (AUDIO,)}
        for ax in _AXES:
            opt[f"{ax}_curve"] = (SCHEDULE,)
        return {"required": req, "optional": opt}

    RETURN_TYPES = (CAMERA, "STRING")
    RETURN_NAMES = ("camera", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, params, mode, fov, audio=None, **kw):
        n = int(params["max_frames"])
        extra = audio_vars(audio)
        values, used = {}, []
        for ax in _AXES:
            curve = kw.get(f"{ax}_curve")
            if curve is not None:
                values[ax] = _fit(curve.as_list(), n, 1.0 if ax == "zoom" else 0.0)
                used.append(f"{ax} <- curve")
            else:
                values[ax] = build_schedule(kw.get(ax, _EXPR_DEFAULTS[ax]), max_frames=n,
                                            fps=params["fps"], extra_vars=extra).as_list()
        cam = build_camera(values, max_frames=n, mode=mode, fov=fov)
        info = f"camera expressions ({mode}, fov {fov:g})" + (
            "\n  " + "\n  ".join(used) if used else "")
        return (cam, info)


# ---------------------------------------------------------------------------
# Storyboard
# ---------------------------------------------------------------------------

class DifforumStoryboard:
    """The whole clip warped by the camera, no diffusion - in about a second.

    Same engine as the Feedback Sampler with the sampler switched off, so what
    you see is exactly the motion the render will get: contact sheet, the
    frames themselves, the camera path from above, and a pacing verdict.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "init_image": ("IMAGE",),
                "every_nth": ("INT", {"default": 8, "min": 1, "max": 240}),
                "columns": ("INT", {"default": 6, "min": 1, "max": 16}),
                "preview_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "params": (PARAMS,),
                "camera": (CAMERA,),
                "depth": ("IMAGE",),
                "symmetry": (["none", "mirror_h", "mirror_v", "mirror_quad", "kaleidoscope"],
                             {"default": "none"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "frames", "camera_path", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, init_image, every_nth, columns, preview_scale, direction=None,
            params=None, camera=None, depth=None, symmetry="none"):
        import torch

        from ..core.engine import EngineConfig, FeedbackEngine, iter_feedback
        from ..core.plot import render_camera_path
        from ..core.storyboard import contact_sheet, drift_curve

        if direction is not None:
            params = params or direction.params
            camera = camera or direction.camera
        if params is None or camera is None:
            raise ValueError("Storyboard needs a direction, or params + camera.")

        n = min(int(params["max_frames"]), len(camera.deltas))
        fps = float(params["fps"])
        w = max(64, int(params["width"] * preview_scale) // 8 * 8)
        h = max(64, int(params["height"] * preview_scale) // 8 * 8)
        cfg = EngineConfig(width=w, height=h, symmetry=symmetry, sharpen=0.0, noise=0.0,
                           hole_noise=0.0, color_mode="none", anchor_mode="none")
        engine = FeedbackEngine(camera, cfg, depth=depth)
        frames = [img for _f, img in iter_feedback(engine, init_image, n)]

        drift = drift_curve(frames)
        step = max(1, int(every_nth))
        picks = list(range(0, len(frames), step))
        if picks[-1] != len(frames) - 1:
            picks.append(len(frames) - 1)
        sheet = contact_sheet([frames[i] for i in picks], columns=int(columns), cell_width=192,
                              labels=[f"{i / fps:.1f}s" for i in picks])
        path = torch.from_numpy(render_camera_path(camera.poses, camera.zoom, mode=camera.mode,
                                                   width=384, height=384)).unsqueeze(0)

        body = drift[1:] or [0.0]
        avg = sum(body) / len(body)
        peak = max(body)
        peak_at = body.index(peak) + 1
        verdict = []
        if avg < 0.004:
            verdict.append("  camera is nearly still - the feedback will stagnate; add motion")
        elif avg > 0.05:
            verdict.append("  camera is very fast - expect smearing unless energy is high")
        if avg > 0 and peak > avg * 4:
            verdict.append(f"  spike at {peak_at / fps:.2f}s ({peak / avg:.1f}x average) - "
                           "raise the Director's transition to ease that cut")
        if not verdict:
            verdict.append("  motion is evenly paced")
        info = "\n".join([
            f"storyboard: {len(frames)} frames at {w}x{h} ({engine.report.mode}), "
            f"drift avg {avg:.4f} peak {peak:.4f}",
            *verdict, *[f"  {x}" for x in engine.report.notes],
        ])
        return (sheet, torch.cat(frames, dim=0), path, info)


# ---------------------------------------------------------------------------
# Keyframe Images (multikeyframing) and Animatic (previz)
# ---------------------------------------------------------------------------

def _parse_times(text: str, fps: float, n: int) -> list[int]:
    """'0, 4s, 9.5s, 200' -> frame indices (seconds with an s suffix)."""
    out = []
    for tok in str(text or "").replace(";", ",").split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        try:
            f = round(float(tok[:-1]) * fps) if tok.endswith("s") else int(float(tok))
        except ValueError:
            continue
        out.append(max(0, min(n - 1, f)))
    return out


class DifforumKeyframeImages:
    """Pin your own pictures to moments of the clip (multikeyframing).

    Place markers on the Director's **Keys** track (or type times: `0, 4s,
    9.5s`) and feed a batch of images in the same order. The keyframes then
    drive every renderer: the Feedback Sampler travels *through* them, H3 /
    LTX Guides anchor them, the Animatic shows them. Made for installations and
    experimental pieces where the image has to hit a picture on a beat.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"images": ("IMAGE", {"tooltip": "One image per key, in order (use a Batch Images node)."})},
            "optional": {
                "direction": (DIRECTION,),
                "params": (PARAMS,),
                "times": ("STRING", {"default": "", "tooltip": "Override: '0, 4s, 9.5s' or frame numbers."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("keyframes", "indices", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, images, direction=None, params=None, times=""):
        from ..core.engine import resize_bhwc
        params = params or (direction.params if direction is not None else None)
        if params is None:
            raise ValueError("Keyframe Images needs a direction or params.")
        n, fps = int(params["max_frames"]), float(params["fps"])
        notes = []
        if str(times).strip():
            idx = _parse_times(times, fps, n)
        elif direction is not None and direction.direction.keys:
            idx = [k["start"] for k in direction.direction.keys]
        else:
            m = images.shape[0]
            idx = [round(i * (n - 1) / max(1, m - 1)) for i in range(m)]
            notes.append("  no Keys markers or times: spread evenly")
        count = min(len(idx), images.shape[0])
        if count < max(len(idx), images.shape[0]):
            notes.append(f"  {len(idx)} times for {images.shape[0]} images: using {count}")
        pairs = sorted(zip(idx[:count], range(count)))
        keys = resize_bhwc(images[[i for _f, i in pairs]][..., :3], int(params["width"]), int(params["height"]))
        frames = [f for f, _i in pairs]
        info = "\n".join([f"{count} keyframes: " + ", ".join(f"{f / fps:.2f}s" for f in frames), *notes])
        return (keys, ",".join(map(str, frames)), info)


def _synthetic_plate(w: int, h: int):
    """A readable stand-in image (gradient, grid, circle) when no picture is given."""
    import torch
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    img = torch.stack([0.12 + 0.35 * xs, 0.10 + 0.20 * ys, 0.30 + 0.25 * (1 - xs)], dim=-1)
    grid = ((torch.remainder(xs * 12, 1) < 0.03) | (torch.remainder(ys * 12 * h / w, 1) < 0.03)).float()
    r = ((xs - 0.5) * w / h) ** 2 + (ys - 0.5) ** 2
    ring = ((r > 0.16 ** 2) & (r < 0.175 ** 2)).float()
    img = img + 0.25 * grid[..., None] + 0.6 * ring[..., None]
    return img.clamp(0, 1).unsqueeze(0)


class DifforumAnimatic:
    """Previz the whole shot before rendering anything heavy.

    Plays the Director's camera over your first frame (or a stand-in plate) at
    low resolution, in about a second, with the timecode, the active scene
    prompt, the camera move, the energy level and the keyframe markers burnt
    in. Image keyframes, when connected, show up at their moments. Send it to
    Create Video + Save Video; the Director's "Previz only" button mutes the
    render outputs so only this runs.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "direction": (DIRECTION,),
                "preview_scale": ("FLOAT", {"default": 0.4, "min": 0.1, "max": 1.0, "step": 0.05}),
                "overlay": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "init_image": ("IMAGE",),
                "depth": ("IMAGE",),
                "key_images": ("IMAGE",),
                "key_indices": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "STRING")
    RETURN_NAMES = ("frames", "fps", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, direction, preview_scale, overlay, init_image=None, depth=None,
            key_images=None, key_indices=""):
        import torch

        from ..core.engine import EngineConfig, FeedbackEngine, iter_feedback
        from .render import key_hook

        params, camera, d = direction.params, direction.camera, direction.direction
        n, fps = min(int(params["max_frames"]), len(camera.deltas)), float(params["fps"])
        w = max(64, int(params["width"] * preview_scale) // 8 * 8)
        h = max(64, int(params["height"] * preview_scale) // 8 * 8)
        plate = init_image if init_image is not None else _synthetic_plate(w, h)
        engine = FeedbackEngine(camera, EngineConfig(width=w, height=h, sharpen=0.0, noise=0.0,
                                hole_noise=0.0, color_mode="none", anchor_mode="none"), depth=depth)
        if key_images is None and direction.key_images is not None:      # pictures pinned on the Director
            key_images, key_indices = direction.key_images, direction.key_indices
        if init_image is None and key_images is not None and str(key_indices).split(",")[0].strip() == "0":
            plate = key_images[:1]
        hook = key_hook(key_images, key_indices, w, h, 1.0, 6)
        frames = [img for _f, img in iter_feedback(engine, plate, n, pre_warp=hook)]
        video = torch.cat(frames, dim=0)
        key_frames = [int(t) for t in str(key_indices or "").split(",") if t.strip().isdigit()]
        key_frames = key_frames or [k["start"] for k in d.keys]
        if overlay:
            video = _burn_overlay(video, d, direction.strength, fps, key_frames)
        info = (f"animatic {n} frames {w}x{h} ({engine.report.mode}), "
                f"{len(key_frames)} key(s); {n / fps:.2f}s")
        return (video, fps, info)


def _burn_overlay(video, d, strength, fps, key_frames):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw, ImageFont

    from ..core.camera_presets import MOVE_INFO

    n, h, w = video.shape[0], video.shape[1], video.shape[2]
    size = max(10, h // 22)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:            # Pillow < 10.1
        font = ImageFont.load_default()
    out = []
    for f in range(n):
        img = Image.fromarray((video[f].clamp(0, 1).numpy() * 255).astype("uint8"))
        dr = ImageDraw.Draw(img, "RGBA")
        scene = next((s for s in reversed(d.scenes) if s["start"] <= f), None)
        cam = next((c for c in reversed(d.camera_blocks) if c["start"] <= f), None)
        sec = f / fps
        head = f"{int(sec // 60):02d}:{sec % 60:05.2f}  f{f}"
        move = MOVE_INFO[cam["move"]][0] if cam else "Still"
        dr.rectangle([0, 0, w, size * 2 + 8], fill=(0, 0, 0, 150))
        dr.text((6, 3), f"{head}   CAM {move}", fill=(255, 255, 255, 255), font=font)
        tag = "PREVIZ"
        tw = dr.textlength(tag, font=font)
        dr.rectangle([w - tw - 14, 3, w - 4, size + 7], fill=(220, 60, 60, 220))
        dr.text((w - tw - 9, 4), tag, fill=(255, 255, 255, 255), font=font)
        if scene:
            text = f"[{scene['mood']}] {scene['prompt']}"
            while len(text) > 8 and dr.textlength(text, font=font) > w - 12:
                text = text[:-4] + "..."
            dr.text((6, size + 6), text, fill=(210, 210, 210, 255), font=font)
        # bottom strip: energy + progress + key markers
        bar = max(6, h // 40)
        y0 = h - bar - 4
        dr.rectangle([0, y0 - 2, w, h], fill=(0, 0, 0, 150))
        e = float(strength.at(f)) if strength is not None else 0.0
        dr.rectangle([4, y0, 4 + int((w - 8) * min(1.0, e)), y0 + bar // 2], fill=(240, 181, 58, 230))
        px = int(4 + (w - 8) * f / max(1, n - 1))
        dr.rectangle([4, y0 + bar // 2 + 1, px, y0 + bar], fill=(79, 142, 247, 230))
        for k in key_frames:
            kx = int(4 + (w - 8) * k / max(1, n - 1))
            dr.polygon([(kx, y0 - 2), (kx + 4, y0 + bar // 2), (kx, y0 + bar + 2), (kx - 4, y0 + bar // 2)],
                       fill=(255, 255, 255, 230))
        if any(abs(f - k) <= 1 for k in key_frames):
            dr.rectangle([0, 0, w - 1, h - 1], outline=(255, 255, 255, 255), width=max(2, h // 90))
        out.append(torch.from_numpy(np.asarray(img).astype("float32") / 255.0))
    return torch.stack(out, dim=0)


NODE_CLASS_MAPPINGS = {
    "Difforum_Director": DifforumDirector,
    "Difforum_Camera": DifforumCamera,
    "Difforum_CameraExpr": DifforumCameraExpr,
    "Difforum_Storyboard": DifforumStoryboard,
    "Difforum_KeyframeImages": DifforumKeyframeImages,
    "Difforum_Animatic": DifforumAnimatic,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_Director": "Difforum · Director (timeline)",
    "Difforum_Camera": "Difforum · Camera (keys)",
    "Difforum_CameraExpr": "Difforum · Camera (expressions)",
    "Difforum_Storyboard": "Difforum · Storyboard",
    "Difforum_KeyframeImages": "Difforum · Keyframe Images",
    "Difforum_Animatic": "Difforum · Animatic (previz)",
}
