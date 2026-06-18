"""
Difforum audio-reactive nodes.

DifforumAudioAnalyzer turns a ComfyUI AUDIO input into per-frame reactive
curves (amp/low/mid/high/onset/beat) packaged as DIFFORUM_AUDIO. Feed that into
the optional `audio` socket of Difforum · Schedule and reference the curves by
name in expressions, e.g.  0:(0.2 + 0.8*amp)  or  0:(beat*0.6) .
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.audio import REACTIVE_MODES, REACTIVE_SOURCES, analyze, reactive_curve  # noqa: E402
from core.schedule import Schedule  # noqa: E402

CATEGORY = "Difforum/audio"


class DifforumAudioAnalyzer:
    """Analyze an AUDIO clip into per-frame reactive curves."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "audio": ("AUDIO",),
                "smoothing": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 0.99, "step": 0.01}),
                "normalize": ("BOOLEAN", {"default": True}),
                "beat_sensitivity": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("DIFFORUM_AUDIO", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("audio_curves", "amp", "low", "mid", "high", "onset", "beat")
    OUTPUT_IS_LIST = (False, True, True, True, True, True, True)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, audio, smoothing, normalize, beat_sensitivity):
        waveform = audio["waveform"]
        sample_rate = int(audio["sample_rate"])
        # torch tensor [B,C,N] -> numpy; analyze() collapses to mono
        samples = np.asarray(waveform.detach().cpu().numpy(), dtype=np.float32)
        curves = analyze(
            samples,
            sample_rate=sample_rate,
            fps=params["fps"],
            max_frames=params["max_frames"],
            smoothing=float(smoothing),
            normalize=bool(normalize),
            beat_sensitivity=float(beat_sensitivity),
        )
        bundle = {"curves": curves, "fps": params["fps"], "frames": params["max_frames"]}
        return (
            bundle,
            curves["amp"], curves["low"], curves["mid"],
            curves["high"], curves["onset"], curves["beat"],
        )


class DifforumAudioSchedule:
    """Turn an audio band into a ready-to-use reactive curve (a DIFFORUM_SCHEDULE).

    Pick a band (bass/beat/onset…) and how it modulates a value: e.g. a
    bass-pumped zoom (`low`, multiply, base 1.0, amount 0.08) or a beat-pulsed
    strength (`beat`, add, base 0.4, amount 0.4). Plugs into any schedule input
    (Camera fields via Schedule, Feedback strength/cfg, etc.).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "params": ("DIFFORUM_PARAMS",),
                "audio": ("DIFFORUM_AUDIO",),
                "source": (list(REACTIVE_SOURCES), {"default": "low"}),
                "mode": (list(REACTIVE_MODES), {"default": "add"}),
                "base": ("FLOAT", {"default": 0.4, "min": -100.0, "max": 100.0, "step": 0.01}),
                "amount": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01}),
                "smoothing": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 0.99, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("DIFFORUM_SCHEDULE", "FLOAT")
    RETURN_NAMES = ("schedule", "values")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, params, audio, source, mode, base, amount, smoothing):
        curves = audio.get("curves", {}) if isinstance(audio, dict) else {}
        n = params["max_frames"]
        vals = reactive_curve(curves, source, base, amount, mode=mode,
                              smoothing=float(smoothing), max_frames=n)
        sched = Schedule(values=vals, fps=params["fps"],
                         source=f"audio:{source} {mode} base={base} amt={amount}")
        return (sched, vals)


NODE_CLASS_MAPPINGS = {
    "DifforumAudioAnalyzer": DifforumAudioAnalyzer,
    "DifforumAudioSchedule": DifforumAudioSchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumAudioAnalyzer": "Difforum · Audio Analyzer",
    "DifforumAudioSchedule": "Difforum · Audio Schedule (reactive)",
}
