"""Standalone tests for the Difforum audio analyzer (numpy only)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.audio import analyze, reactive_curve, to_mono  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


def tone(freq, dur, sr=44100):
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


SR = 44100
FPS = 24
DUR = 2.0
FRAMES = int(FPS * DUR)

print("audio analyzer:")

# mono conversion from [B, C, N]
stereo = np.stack([tone(440, DUR), tone(440, DUR)])[None]  # [1,2,N]
check("to_mono shape", to_mono(stereo).ndim == 1)

# bass tone lands in low band, not high
low_sig = tone(80, DUR)
c = analyze(low_sig, SR, FPS, FRAMES)
check("curves length", all(len(v) == FRAMES for v in c.values()))
check("has all curves", set(c) >= {"amp", "low", "mid", "high", "onset", "beat"})
mid_frame = FRAMES // 2
check("bass -> low>high", c["low"][mid_frame] > c["high"][mid_frame],
      f"low={c['low'][mid_frame]:.3f} high={c['high'][mid_frame]:.3f}")

# treble tone lands in high band, not low
hi_sig = tone(6000, DUR)
c = analyze(hi_sig, SR, FPS, FRAMES)
check("treble -> high>low", c["high"][mid_frame] > c["low"][mid_frame],
      f"low={c['low'][mid_frame]:.3f} high={c['high'][mid_frame]:.3f}")

# amplitude responds to a swell (silence -> loud)
swell = np.concatenate([np.zeros(SR, np.float32), tone(440, 1.0)])
c = analyze(swell, SR, FPS, FRAMES, smoothing=0.0)
check("amp rises", c["amp"][-1] > c["amp"][0],
      f"first={c['amp'][0]:.3f} last={c['amp'][-1]:.3f}")

# onset fires at the attack (silence -> tone transition)
c = analyze(swell, SR, FPS, FRAMES, smoothing=0.0)
attack_frame = int(FPS * 1.0)
near = max(c["onset"][attack_frame - 1: attack_frame + 2])
check("onset at attack", near > 0.3, f"near={near:.3f}")

# normalize keeps values within [0,1]
c = analyze(tone(440, DUR), SR, FPS, FRAMES, normalize=True)
check("normalized range", all(0.0 <= v <= 1.0001 for v in c["amp"]))

print("reactive schemes:")
curves = {"low": [0.0, 1.0, 0.5, 1.0], "beat": [0.0, 1.0, 0.0, 1.0]}
# add: base + amount*curve
add = reactive_curve(curves, "low", base=0.4, amount=0.5, mode="add", smoothing=0.0)
check("add mode", all(abs(a - b) < 1e-5 for a, b in zip(add, [0.4, 0.9, 0.65, 0.9])), str(add))
# multiply: base*(1+amount*curve) -> bass-pump zoom
mul = reactive_curve(curves, "low", base=1.0, amount=0.1, mode="multiply", smoothing=0.0)
check("multiply mode", abs(mul[1] - 1.1) < 1e-6, str(mul))
# subtract = ducking
sub = reactive_curve(curves, "beat", base=1.0, amount=1.0, mode="subtract", smoothing=0.0)
check("subtract mode", all(abs(a - b) < 1e-5 for a, b in zip(sub, [1.0, 0.0, 1.0, 0.0])), str(sub))
# length clamps/pads to max_frames
padded = reactive_curve(curves, "low", 0.0, 1.0, max_frames=6)
check("pads to max_frames", len(padded) == 6 and padded[-1] == padded[3])
# unknown source raises
try:
    reactive_curve(curves, "nope", 0, 1)
    check("reject bad source", False)
except ValueError:
    check("reject bad source", True)

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL AUDIO TESTS PASSED")
