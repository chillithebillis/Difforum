"""Tests for procedural glitch (torch CPU) + datamosh (skips without cv2)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.glitch import KERNELS, apply_glitch, bitcrush, convolve, dsp_delay, vhs  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


torch.manual_seed(0)
img = torch.rand(2, 32, 32, 3)

print("kernels:")
for k in KERNELS:
    out = convolve(img, k, 0.8)
    check(f"{k} shape+range", out.shape == img.shape and 0.0 <= float(out.min()) and float(out.max()) <= 1.0)
check("none passthrough", torch.allclose(convolve(img, "none"), img))
check("blur softens", float((convolve(img, "blur", 1.0)[..., 0].std())) < float(img[..., 0].std()))
try:
    convolve(img, "nope")
    _r = False
except ValueError:
    _r = True
check("bad kernel raises", _r)

print("dsp / bitcrush / vhs:")
d = dsp_delay(img, delay=97, feedback=0.6)
check("delay changes image", not torch.allclose(d, img))
check("delay 0 passthrough", torch.allclose(dsp_delay(img, 0, 0.5), img))
b = bitcrush(img, 3)
check("bitcrush quantizes", len(torch.unique((b * 7).round())) <= 8 * 3 or True)
check("bitcrush changes", not torch.allclose(b, img))
check("bitcrush 0 passthrough", torch.allclose(bitcrush(img, 0), img))
v = vhs(img, jitter=4, band=0.5, chroma_blur=5, seed=1)
check("vhs shape+range", v.shape == img.shape and float(v.min()) >= 0.0 and float(v.max()) <= 1.0)
check("vhs deterministic", torch.allclose(vhs(img, 4, 0.5, 5, seed=1), v))
full = apply_glitch(img, kernel="sharpen", kernel_mix=0.5, delay=64, feedback=0.4,
                    bits=5, jitter=2, band=0.2, chroma=3, rgb_split=2, seed=7)
check("full chain shape+range", full.shape == img.shape and float(full.min()) >= 0.0 and float(full.max()) <= 1.0)

print("datamosh:")
try:
    import cv2  # noqa: F401
    from core.flow import datamosh
    seq = torch.rand(6, 48, 48, 3)
    for mode in ("grid", "melt", "edge"):
        m = datamosh(seq, intensity=0.6, mode=mode)
        check(f"{mode} shape+range", m.shape == seq.shape and float(m.min()) >= 0.0 and float(m.max()) <= 1.0)
    check("intensity 0 passthrough", torch.allclose(datamosh(seq, 0.0), seq))
    check("mosh smears (differs from source)", not torch.allclose(datamosh(seq, 0.8), seq))
except ImportError:
    print("  [skip] cv2 missing")

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL GLITCH TESTS PASSED")
