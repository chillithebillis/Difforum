"""Curves & prompts: expression schedules, audio analysis, prompt travel."""

from __future__ import annotations

import numpy as np

from ..core import EASINGS, build_schedule
from ..core.audio import REACTIVE_MODES, REACTIVE_SOURCES, analyze, reactive_curve
from ..core.prompt import parse_prompt_schedule
from ..core.schedule import Schedule
from ._common import AUDIO, CAT_CURVES, PARAMS, PROMPT, SCHEDULE, audio_vars
from .direction import encode_prompt_track


class DifforumSchedule:
    """A per-frame curve from Deforum keyframe syntax: `0:(0.4), 48:(0.6+0.1*sin(t/8))`.

    Variables: t/f (frame), s (seconds), fps, max_f, and every audio band when
    an Audio Analyzer is connected. Easing applies between keyframes."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": (PARAMS,),
                "schedule": ("STRING", {"multiline": True, "default": "0:(0.45), 60:(0.6), 119:(0.45)"}),
                "easing": (list(EASINGS), {"default": "ease_in_out"}),
            },
            "optional": {"audio": (AUDIO,)},
        }

    RETURN_TYPES = (SCHEDULE, "STRING")
    RETURN_NAMES = ("schedule", "info")
    FUNCTION = "run"
    CATEGORY = CAT_CURVES

    def run(self, params, schedule, easing, audio=None):
        sched = build_schedule(schedule, max_frames=params["max_frames"], fps=params["fps"],
                               easing=easing, extra_vars=audio_vars(audio))
        v = sched.as_list()
        return (sched, f"{len(v)} frames  min {min(v):.4g}  max {max(v):.4g}")


class DifforumSchedulePlot:
    """Draw any schedule as an image (and a one-line summary)."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "schedule": (SCHEDULE,),
            "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
            "height": ("INT", {"default": 200, "min": 64, "max": 4096}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("plot", "info")
    FUNCTION = "run"
    CATEGORY = CAT_CURVES

    def run(self, schedule, width, height):
        import torch

        from ..core.plot import render_curve
        vals = schedule.as_list()
        arr = render_curve(vals, width=int(width), height=int(height))
        info = (f"{len(vals)} frames  min {min(vals):.4g}  max {max(vals):.4g}  "
                f"first {vals[0]:.4g}  last {vals[-1]:.4g}  ({schedule.source})") if vals else "empty"
        return (torch.from_numpy(arr).unsqueeze(0), info)


class DifforumAudioAnalyzer:
    """Audio -> per-frame bands: amp, low, mid, high, onset, beat (0..1).

    Use the bands in any expression (`1 + 0.05*low`), in a Director camera
    block's audio reaction, or turn one into a curve with Audio Curve."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "params": (PARAMS,),
            "audio": ("AUDIO",),
            "smoothing": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.99, "step": 0.01}),
            "beat_sensitivity": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
            "offset_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 36000.0, "step": 0.1,
                               "tooltip": "Start reading the track here (sync to an edit)."}),
        }}

    RETURN_TYPES = (AUDIO, "IMAGE", "STRING")
    RETURN_NAMES = ("audio_curves", "bands_plot", "info")
    FUNCTION = "run"
    CATEGORY = CAT_CURVES

    def run(self, params, audio, smoothing, beat_sensitivity, offset_seconds):
        import torch

        from ..core.plot import render_curve
        sr = int(audio["sample_rate"])
        samples = np.asarray(audio["waveform"].detach().cpu().numpy(), dtype=np.float32)
        start = int(float(offset_seconds) * sr)
        if start:
            samples = samples[..., start:]
        curves = analyze(samples, sample_rate=sr, fps=params["fps"],
                         max_frames=params["max_frames"], smoothing=float(smoothing),
                         normalize=True, beat_sensitivity=float(beat_sensitivity))
        rows = [render_curve(curves[k], width=512, height=64) for k in ("low", "mid", "high", "beat")]
        plot = torch.from_numpy(np.concatenate(rows, axis=0)).unsqueeze(0)
        beats = int(sum(1 for v in curves["beat"] if v > 0.5))
        bundle = {"curves": curves, "fps": params["fps"], "frames": params["max_frames"]}
        return (bundle, plot, f"audio: {params['max_frames']} frames, ~{beats} beats "
                              f"(plot rows: low / mid / high / beat)")


class DifforumAudioCurve:
    """One audio band -> a ready curve: e.g. bass-pumped energy
    (`low`, add, base 0.45, amount 0.2) or a beat-pulsed cfg."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "params": (PARAMS,),
            "audio_curves": (AUDIO,),
            "source": (list(REACTIVE_SOURCES), {"default": "low"}),
            "mode": (list(REACTIVE_MODES), {"default": "add"}),
            "base": ("FLOAT", {"default": 0.45, "min": -100.0, "max": 100.0, "step": 0.01}),
            "amount": ("FLOAT", {"default": 0.2, "min": -100.0, "max": 100.0, "step": 0.01}),
            "smoothing": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.99, "step": 0.01}),
        }}

    RETURN_TYPES = (SCHEDULE,)
    RETURN_NAMES = ("schedule",)
    FUNCTION = "run"
    CATEGORY = CAT_CURVES

    def run(self, params, audio_curves, source, mode, base, amount, smoothing):
        vals = reactive_curve(audio_vars(audio_curves) or {}, source, base, amount, mode=mode,
                              smoothing=float(smoothing), max_frames=params["max_frames"])
        return (Schedule(values=vals, fps=params["fps"],
                         source=f"audio {source} {mode} {base:g}+{amount:g}"),)


class DifforumPromptTravel:
    """Prompt travel from `frame: prompt` lines (or `2.5s: prompt`), blended
    per frame. Encodes each prompt once and blends lazily, so long clips on big
    text encoders stay light. `build_batched` adds one batched conditioning
    (frame i = prompt i) for batch samplers such as AnimateDiff."""

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "params": (PARAMS,),
            "clip": ("CLIP",),
            "prompts": ("STRING", {"multiline": True, "default":
                        "0: a serene misty forest, soft light\n"
                        "2s: a glowing crystal cave, bioluminescent\n"
                        "4s: a vast starry galaxy, swirling nebula"}),
            "easing": (list(EASINGS), {"default": "ease_in_out"}),
        }, "optional": {
            "build_batched": ("BOOLEAN", {"default": False,
                              "tooltip": "Also build the batched output (costs RAM on long clips)."}),
        }}

    RETURN_TYPES = (PROMPT, "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("prompts", "first_frame_cond", "batched", "info")
    FUNCTION = "run"
    CATEGORY = CAT_CURVES

    def run(self, params, clip, prompts, easing, build_batched=False):
        fps = float(params["fps"])
        n = int(params["max_frames"])
        lines = []
        for raw in prompts.splitlines():   # allow "2.5s: ..." timestamps
            head, sep, rest = raw.partition(":")
            h = head.strip().lower()
            if sep and h.endswith("s") and h[:-1].replace(".", "", 1).isdigit():
                lines.append(f"{int(round(float(h[:-1]) * fps))}:{rest}")
            else:
                lines.append(raw)
        kfs = parse_prompt_schedule("\n".join(lines))
        track = encode_prompt_track(clip, kfs, n, easing)
        info = f"{len(kfs)} prompts over {n} frames\n" + "\n".join(
            f"  {f / fps:6.2f}s  {t[:60]}" for f, t in kfs)
        batched = None
        if build_batched:
            from ..core.prompt import batch_conditioning
            batched = batch_conditioning(track)
        return (track, track[0], batched, info)


NODE_CLASS_MAPPINGS = {
    "Difforum_Schedule": DifforumSchedule,
    "Difforum_SchedulePlot": DifforumSchedulePlot,
    "Difforum_AudioAnalyzer": DifforumAudioAnalyzer,
    "Difforum_AudioCurve": DifforumAudioCurve,
    "Difforum_PromptTravel": DifforumPromptTravel,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_Schedule": "Difforum · Schedule",
    "Difforum_SchedulePlot": "Difforum · Schedule Plot",
    "Difforum_AudioAnalyzer": "Difforum · Audio Analyzer",
    "Difforum_AudioCurve": "Difforum · Audio Curve",
    "Difforum_PromptTravel": "Difforum · Prompt Travel",
}
