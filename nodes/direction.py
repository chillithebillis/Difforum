"""Direction nodes: the Director timeline, keyframed/expression cameras and the
Storyboard dry run."""

from __future__ import annotations

import json

from ..core.camera import CAMERA_MODES, build_camera
from ..core.camera_keys import keys_to_axis_values, lens_note, parse_camera_keys
from ..core.camera_presets import flat_presets, needs_depth
from ..core.direction import (
    LOOKS, DirectionBundle, build_direction, default_timeline, describe_timed,
)
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
            },
        }

    RETURN_TYPES = (DIRECTION, CAMERA, SCHEDULE, PROMPT, "STRING", "STRING")
    RETURN_NAMES = ("direction", "camera", "strength", "prompts", "camera_text", "info")
    FUNCTION = "run"
    CATEGORY = CAT_DIRECT

    def run(self, params, timeline, camera_mode, look, transition, camera_scale,
            energy_bias, variation, variation_seed, clip=None, audio=None):
        n = int(params["max_frames"])
        fps = float(params["fps"])
        d = build_direction(
            timeline, n, fps, mode=camera_mode, camera_scale=camera_scale,
            strength_bias=energy_bias + LOOKS.get(look, LOOKS["cinematic"])["energy"], blend=transition, variation=variation,
            variation_seed=variation_seed, audio_curves=audio_vars(audio),
        )
        camera = _track(d.axes, d.lens, n, camera_mode, [b["move"] for b in d.camera_blocks])
        strength = Schedule(values=d.strength, fps=fps, source="director energy")
        cfg = Schedule(values=d.cfg, fps=fps, source="director guidance") if d.cfg else None
        prompts = encode_prompt_track(clip, d.prompts, n) if clip is not None else None

        bundle = DirectionBundle(params=params, camera=camera, strength=strength, cfg=cfg,
                                 prompts=prompts, direction=d, look_name=look)
        info = "\n".join([
            f"[Difforum Director]  {n} frames  {n / fps:.2f}s  camera {camera_mode}  look {look}",
            *d.summary,
            *([""] + [f"  ! {w}" for w in d.warnings] if d.warnings else []),
            "" if clip is not None else "  (connect a CLIP to get prompt travel on the direction wire)",
        ])
        timed = describe_timed(d.camera_blocks, n, fps)
        return {
            "ui": {"text": [info]},
            "result": (bundle, camera, strength, prompts,
                       d.camera_text + " " + bundle.look_prompt + "\n\n" + timed, info),
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
                "mode": (list(CAMERA_MODES), {"default": "2d"}),
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

    def run(self, params, keys, mode, transition, speed, loop_mode, cycle_frames,
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
        cam = _track(values, _fit(lens, total, 40.0), total, mode, moves)

        depth_moves = sorted({m for m in moves if needs_depth(m)})
        if depth_moves:
            notes.append(f"  {', '.join(depth_moves)}: real parallax needs mode=3d + a depth map; "
                         "otherwise they run as pseudo-3D (dolly->zoom, orbit->pan).")
            notes.append(f"  flat moves: {', '.join(flat_presets()[:10])}...")
        lo, hi = min(cam.fov), max(cam.fov)
        info = "\n".join([f"camera ({mode}, transition {transition:g})",
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


NODE_CLASS_MAPPINGS = {
    "Difforum_Director": DifforumDirector,
    "Difforum_Camera": DifforumCamera,
    "Difforum_CameraExpr": DifforumCameraExpr,
    "Difforum_Storyboard": DifforumStoryboard,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_Director": "Difforum · Director (timeline)",
    "Difforum_Camera": "Difforum · Camera (keys)",
    "Difforum_CameraExpr": "Difforum · Camera (expressions)",
    "Difforum_Storyboard": "Difforum · Storyboard",
}
