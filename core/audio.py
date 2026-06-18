"""
Audio analysis for Difforum - turns a waveform into per-frame reactive curves.

Pure numpy (FFT), no librosa, no torch. The ComfyUI node converts the native
AUDIO tensor to a mono numpy array and calls `analyze`. Output curves align
1:1 with animation frames so they can be dropped straight into schedule
expressions, e.g.  0:(0.2 + 0.8*amp)  or  0:(beat*45) .

Curves produced (each a list[float] of length max_frames):
    amp    RMS loudness                 (0..1 if normalized)
    low    bass band energy             (~20-250 Hz)
    mid    mids band energy             (~250-2000 Hz)
    high   highs band energy            (~2000 Hz..nyquist)
    onset  spectral flux (attack/hits)  (0..1)
    beat   gated onset peaks            (0 or 1, with short decay)
"""

from __future__ import annotations

import numpy as np

_BANDS_HZ = {
    "low": (20.0, 250.0),
    "mid": (250.0, 2000.0),
    "high": (2000.0, 20000.0),
}


def _ema(x: np.ndarray, smoothing: float) -> np.ndarray:
    """Exponential moving average. smoothing in [0,1): 0 = none."""
    if smoothing <= 0.0:
        return x
    alpha = 1.0 - float(np.clip(smoothing, 0.0, 0.999))
    out = np.empty_like(x)
    acc = x[0] if len(x) else 0.0
    for i, v in enumerate(x):
        acc = alpha * v + (1.0 - alpha) * acc
        out[i] = acc
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    m = float(x.max()) if x.size else 0.0
    return x / m if m > 1e-9 else x


def to_mono(waveform: np.ndarray) -> np.ndarray:
    """Accepts shapes [N], [C,N] or [B,C,N] and returns float32 mono [N]."""
    w = np.asarray(waveform, dtype=np.float32)
    while w.ndim > 1:
        w = w.mean(axis=0)
    return w


def analyze(
    samples: np.ndarray,
    sample_rate: int,
    fps: float,
    max_frames: int,
    smoothing: float = 0.25,
    normalize: bool = True,
    beat_sensitivity: float = 1.5,
) -> dict[str, list[float]]:
    """Return a dict of per-frame reactive curves of length `max_frames`."""
    x = to_mono(samples)
    sr = int(sample_rate)
    hop = max(1, int(round(sr / float(fps))))
    win = max(hop, 1024)
    window = np.hanning(win).astype(np.float32)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)

    band_masks = {
        name: (freqs >= lo) & (freqs < hi) for name, (lo, hi) in _BANDS_HZ.items()
    }

    amp = np.zeros(max_frames, dtype=np.float32)
    bands = {name: np.zeros(max_frames, dtype=np.float32) for name in _BANDS_HZ}
    onset = np.zeros(max_frames, dtype=np.float32)

    prev_mag = None
    for f in range(max_frames):
        center = f * hop
        start = center - win // 2
        seg = np.zeros(win, dtype=np.float32)
        a = max(0, start)
        b = min(len(x), start + win)
        if b > a:
            seg[(a - start):(b - start)] = x[a:b]

        amp[f] = float(np.sqrt(np.mean(seg * seg)))

        spec = np.abs(np.fft.rfft(seg * window))
        for name, mask in band_masks.items():
            bands[name][f] = float(spec[mask].mean()) if mask.any() else 0.0

        if prev_mag is not None:
            flux = np.maximum(spec - prev_mag, 0.0).sum()
            onset[f] = float(flux)
        prev_mag = spec

    # beat = onset peaks above an adaptive threshold, with a short decay tail
    o = _norm(onset.copy())
    thr = o.mean() + beat_sensitivity * o.std()
    beat = np.zeros(max_frames, dtype=np.float32)
    decay = 0.0
    for f in range(max_frames):
        if o[f] >= thr and o[f] >= (o[f - 1] if f > 0 else 0.0):
            decay = 1.0
        beat[f] = decay
        decay *= 0.6  # ~3-frame tail

    curves = {
        "amp": amp,
        "low": bands["low"],
        "mid": bands["mid"],
        "high": bands["high"],
        "onset": onset,
    }
    if normalize:
        curves = {k: _norm(v) for k, v in curves.items()}
    curves = {k: _ema(v, smoothing) for k, v in curves.items()}
    curves["beat"] = beat  # beat is already 0/1, no smoothing/normalize

    return {k: v.astype(float).tolist() for k, v in curves.items()}


# --- ready-made audio-reactive schemes -------------------------------------

REACTIVE_SOURCES = ("amp", "low", "mid", "high", "onset", "beat")
REACTIVE_MODES = ("add", "subtract", "multiply")


def reactive_curve(
    curves: dict[str, list[float]],
    source: str,
    base: float,
    amount: float,
    mode: str = "add",
    smoothing: float = 0.0,
    max_frames: int | None = None,
) -> list[float]:
    """
    Build a per-frame value curve from an audio band: a ready audio-reactive
    "scheme" (e.g. bass-pump zoom, beat-pulse strength).

      add:      base + amount * curve
      subtract: base - amount * curve        (ducking)
      multiply: base * (1 + amount * curve)

    `curve` is the chosen normalized audio band (0..1). Extra smoothing optional.
    """
    if source not in REACTIVE_SOURCES:
        raise ValueError(f"unknown source {source!r}, pick from {REACTIVE_SOURCES}")
    if mode not in REACTIVE_MODES:
        raise ValueError(f"unknown mode {mode!r}, pick from {REACTIVE_MODES}")
    series = np.asarray(curves.get(source, []), dtype=np.float32)
    n = int(max_frames) if max_frames else series.size
    if n <= 0:
        return []
    if series.size == 0:
        series = np.zeros(n, dtype=np.float32)
    elif series.size < n:
        series = np.pad(series, (0, n - series.size), mode="edge")
    else:
        series = series[:n]
    if smoothing > 0.0:
        series = _ema(series, smoothing)

    b, a = float(base), float(amount)
    if mode == "add":
        out = b + a * series
    elif mode == "subtract":
        out = b - a * series
    else:  # multiply
        out = b * (1.0 + a * series)
    return out.astype(float).tolist()
