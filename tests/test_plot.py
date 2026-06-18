"""Tests for the curve rasterizer (numpy only)."""

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.plot import render_curve  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


print("curve plot:")
vals = [math.sin(i / 10.0) for i in range(120)]
img = render_curve(vals, width=400, height=200)
check("shape", img.shape == (200, 400, 3), str(img.shape))
check("dtype float32", img.dtype == np.float32)
check("range 0..1", float(img.min()) >= 0.0 and float(img.max()) <= 1.0)
# the curve color (cyan-ish) must appear somewhere -> blue channel high pixels exist
blue_hi = (img[..., 2] > 0.9).sum()
check("curve drawn", blue_hi > 100, f"blue_hi={blue_hi}")
# not a uniform image
check("not uniform", float(img.std()) > 0.01)

# empty values -> background only, correct shape
empty = render_curve([], width=64, height=64)
check("empty handled", empty.shape == (64, 64, 3))

# constant values don't crash (zero span)
flat = render_curve([0.5] * 30, width=128, height=64)
check("flat curve ok", flat.shape == (64, 128, 3))

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL PLOT TESTS PASSED")
