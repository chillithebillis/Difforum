"""
Difforum storyboard + keyframed camera nodes.

Difforum · Camera Keys       keyframed camera: moves blend, lens is a channel.
Difforum · Storyboard        the whole clip warped without diffusion, as a sheet.
"""

from __future__ import annotations



import torch  # noqa: E402

from ..core.camera import CAMERA_MODES, build_camera  # noqa: E402
from ..core.camera_keys import (  # noqa: E402
    keys_to_axis_values,
    lens_note,
    parse_camera_keys,
)
from ..core.camera_presets import flat_presets, needs_depth  # noqa: E402
from ..core.loop import LOOP_MODES, close_axis_values, tile_laps  # noqa: E402
from ..core.storyboard import contact_sheet, drift_curve, simulate  # noqa: E402
from ..core.symmetry import SYMMETRY_MODES  # noqa: E402

CATEGORY = "Difforum/camera"

_DEFAULT_KEYS = (
    "0: dolly_in 0.6 0.5 40 ease_in_out\n"
    "30: orbit_right 1.0 0.8 35 ease_in_out\n"
    "60: spiral 1.4 1.1 55 ease_in\n"
    "90: zoom_out 0.5 0.45 40 ease_out"
)


class DifforumCameraKeys:
    """Keyframed camera: each key is a state, and the camera eases between them.

    Syntax, one key per line:

        frame: move [speed] [intensity] [lens] [ease]

    `lens` is the field of view in degrees at that key, interpolated between
    keys - so a push-in with a widening lens gives a dolly-zoom without a
    dedicated preset. `blend` decides how much of each interval is spent
    transitioning: 1.0 is a continuous crane move, 0.0 cuts like a shot list.

    Set `loop_mode` when the clip has to cycle: the motion is projected onto
    components that are periodic over `cycle_frames`, so the path returns to its
    origin with the velocity matched across the wrap - no crossfade needed.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "keys": ("STRING", {"multiline": True, "default": _DEFAULT_KEYS}),
                "blend": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "speed": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 5.0, "step": 0.05}),
                "mode": (list(CAMERA_MODES), {"default": "2d"}),
                "loop_mode": (list(LOOP_MODES), {"default": "off"}),
                "cycle_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "harmonics": ("INT", {"default": 3, "min": 1, "max": 16}),
            },
            "optional": {"audio": ("DIFFORUM_AUDIO",)},
        }

    RETURN_TYPES = ("DIFFORUM_CAMERA", "INT", "STRING")
    RETURN_NAMES = ("camera", "cycle_frames", "info")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, keys, blend, speed, mode, loop_mode, cycle_frames,
            harmonics, audio=None):
        extra = audio.get("curves") if isinstance(audio, dict) else None
        total = int(params["max_frames"])
        fps = float(params["fps"])

        looping = loop_mode != "off"
        cyc = int(cycle_frames) if int(cycle_frames) > 0 else total
        cyc = max(8, min(cyc, total))
        span = cyc if looping else total

        values, lens, summary = keys_to_axis_values(
            keys, max_frames=span, fps=fps, extra_vars=extra, blend=blend
        )

        # global speed rides on top of the per-key rates
        if abs(speed - 1.0) > 1e-6:
            for ax, series in values.items():
                if ax == "zoom":
                    values[ax] = [1.0 + (v - 1.0) * speed for v in series]
                else:
                    values[ax] = [v * speed for v in series]

        notes = []
        laps = 1
        if looping:
            values, residual = close_axis_values(
                values, mode=loop_mode, keep_harmonics=int(harmonics)
            )
            laps = max(1, total // cyc)
            values = tile_laps(values, laps)
            lens = list(lens) * laps
            drift = max(
                (abs(v - (1.0 if a == "zoom" else 0.0)) for a, v in residual.items()),
                default=0.0,
            )
            notes.append(f"  path closure error: {drift:.2e} (0 = exact return)")
            if laps < 2:
                notes.append(
                    f"  ! 1 lap only - the feedback cannot settle. "
                    f"Set max_frames to {cyc * 3} and add Loop Take."
                )

        for ax, series in values.items():
            if len(series) < total:
                series.extend([series[-1]] * (total - len(series)))
            values[ax] = series[:total]
        if len(lens) < total:
            lens.extend([lens[-1]] * (total - len(lens)))
        lens = lens[:total]

        values["fov"] = lens
        cam = build_camera(values, max_frames=total, mode=mode, fov=lens[0])

        parsed = parse_camera_keys(keys)

        # A move that only drives depth axes renders as a frozen frame unless a
        # depth map reaches the Feedback Sampler. Say so loudly - this failure
        # is completely silent otherwise.
        depth_moves = sorted({k["move"] for k in parsed if needs_depth(k["move"])})
        if depth_moves:
            notes.append("")
            notes.append(
                f"  ! {', '.join(depth_moves)} move the camera through space and "
                "do NOTHING without a depth map."
            )
            notes.append(
                "    Connect Depth Anything V2 -> Feedback Sampler.depth and keep "
                "mode=3d,"
            )
            notes.append(
                f"    or swap to a flat move: {', '.join(flat_presets()[:8])}..."
            )
            if mode == "2d":
                notes.append(
                    "    (mode is 2d here, so those axes are discarded before the "
                    "warp even runs)"
                )

        lens_lo, lens_hi = min(lens), max(lens)
        info = "\n".join([
            f"camera keys ({mode}, blend={blend:g}, speed x{speed:g}"
            + (f", loop={loop_mode} h{harmonics}" if looping else "") + ")",
            f"  {len(parsed)} keys over {span}f"
            + (f" x {laps} laps = {total}f" if looping else f" ({total / fps:.1f}s)"),
            f"  lens {lens_lo:.0f}-{lens_hi:.0f} deg  -> {lens_note(lens_lo)}"
            + (f" .. {lens_note(lens_hi)}" if abs(lens_hi - lens_lo) > 5 else ""),
            "",
            summary,
            *notes,
        ])
        return (cam, cyc if looping else total, info)


class DifforumStoryboard:
    """The whole clip, warped but never diffused, as one contact sheet.

    Every frame here is the camera acting on the frame before it - the same
    chain the Feedback Sampler runs, minus the sampler and the VAE. It costs
    well under a second, so the camera, the pacing and the framing can be
    settled before any render time is spent on them.

    `drift` reports how far each frame travels from the last. A flat curve means
    the camera is barely moving and the diffusion will have nothing to grip; a
    spike means the warp is outrunning the sampler and that stretch will smear.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "camera": ("DIFFORUM_CAMERA",),
                "init_image": ("IMAGE",),
                "every_nth": ("INT", {"default": 8, "min": 1, "max": 240}),
                "columns": ("INT", {"default": 6, "min": 1, "max": 16}),
                "cell_width": ("INT", {"default": 192, "min": 48, "max": 512, "step": 16}),
                "preview_scale": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "border": (["reflection", "zeros", "border"], {"default": "reflection"}),
                "symmetry": (list(SYMMETRY_MODES), {"default": "none"}),
                "symmetry_segments": ("INT", {"default": 6, "min": 2, "max": 24}),
            },
            "optional": {
                "depth": ("IMAGE",),
                "translation_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("sheet", "frames", "info")
    FUNCTION = "run"
    CATEGORY = "Difforum/preview"

    def run(self, params, camera, init_image, every_nth, columns, cell_width,
            preview_scale, border, symmetry, symmetry_segments,
            depth=None, translation_scale=1.0):
        n = int(params["max_frames"])
        fps = float(params["fps"])
        w = max(64, int(params["width"] * preview_scale) // 8 * 8)
        h = max(64, int(params["height"] * preview_scale) // 8 * 8)

        frames = simulate(
            init_image, camera, max_frames=n, width=w, height=h,
            depth=depth, border=border, symmetry=symmetry,
            symmetry_segments=int(symmetry_segments),
            translation_scale=float(translation_scale),
        )

        drift = drift_curve(frames)
        step = max(1, int(every_nth))
        picks = list(range(0, len(frames), step))
        if picks and picks[-1] != len(frames) - 1:
            picks.append(len(frames) - 1)

        sheet = contact_sheet(
            [frames[i] for i in picks],
            columns=int(columns),
            cell_width=int(cell_width),
            labels=[f"{i}f {i / fps:.1f}s" for i in picks],
        )
        batch = torch.cat([frames[i] for i in picks], dim=0)

        body = drift[1:] or [0.0]
        avg = sum(body) / len(body)
        peak = max(body)
        peak_at = body.index(peak) + 1
        low = sum(1 for d in body if d < avg * 0.25)

        verdict = []
        if avg < 0.004:
            verdict.append("  camera is nearly still - the loop will stagnate and mush")
            if depth is None:
                verdict.append(
                    "  no depth map here: dolly_in/out, orbit_left/right and sway "
                    "are being discarded"
                )
                verdict.append(
                    "  -> connect Depth Anything V2, or use pan / zoom / roll / "
                    "spiral / rise / shake"
                )
        elif avg > 0.05:
            verdict.append("  camera is very fast - expect smearing unless strength is high")
        if peak > avg * 4:
            verdict.append(
                f"  spike at frame {peak_at} ({peak / avg:.1f}x average) - "
                "that cut will tear; ease it with Camera Keys blend"
            )
        if low > len(body) * 0.4:
            verdict.append(
                f"  {low} of {len(body)} frames barely move - consider fewer, longer moves"
            )
        if not verdict:
            verdict.append("  motion is evenly paced")

        info = "\n".join([
            f"storyboard: {len(frames)} frames simulated at {w}x{h}, "
            f"{len(picks)} shown (every {step})",
            f"  drift  avg {avg:.4f}  peak {peak:.4f} @ frame {peak_at}",
            *verdict,
            "",
            "  no diffusion ran - this is the camera only",
        ])
        return (sheet, batch, info)


NODE_CLASS_MAPPINGS = {
    "DifforumCameraKeys": DifforumCameraKeys,
    "DifforumStoryboard": DifforumStoryboard,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumCameraKeys": "Difforum · Camera Keys (blend + lens)",
    "DifforumStoryboard": "Difforum · Storyboard (no diffusion)",
}
