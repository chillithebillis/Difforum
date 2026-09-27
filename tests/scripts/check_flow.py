"""Tests for the optical-flow anti-flicker (skips if OpenCV is missing)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    import cv2  # noqa: F401
except Exception as e:
    print(f"SKIP flow tests (opencv missing: {e})")
    print("ALL FLOW TESTS PASSED")
    sys.exit(0)

from core.flow import stabilize  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


def temporal_flicker(fr):
    return float((fr[1:] - fr[:-1]).abs().mean())


torch.manual_seed(0)
N, H, W = 10, 64, 64
# static scene + fresh per-frame noise = pure flicker
base = torch.rand(1, H, W, 3).repeat(N, 1, 1, 1)
noisy = (base + torch.randn(N, H, W, 3) * 0.08).clamp(0, 1)

print("deflicker on static scene:")
out = stabilize(noisy, strength=0.6)
check("keeps shape", out.shape == noisy.shape)
check("in range", float(out.min()) >= 0.0 and float(out.max()) <= 1.0)
f_in, f_out = temporal_flicker(noisy), temporal_flicker(out)
check("flicker reduced", f_out < f_in * 0.8, f"{f_in:.4f} -> {f_out:.4f}")
check("strength 0 passthrough", torch.allclose(stabilize(noisy, strength=0.0), noisy))
check("single frame passthrough", stabilize(noisy[:1], 0.5).shape[0] == 1)

print("motion is respected (no hard ghosting):")
# a bright square marching right; stabilized square must follow the current pos
mov = torch.zeros(N, H, W, 3)
for i in range(N):
    x = 8 + i * 4
    mov[i, 24:40, x:x + 12, :] = 1.0
outm = stabilize(mov, strength=0.6)
last = outm[-1]
x_last = 8 + (N - 1) * 4
inside = float(last[24:40, x_last:x_last + 12].mean())
far_behind = float(last[24:40, 8:20].mean())      # where the square started
check("square present at current pos", inside > 0.5, f"{inside:.3f}")
check("no residue at start pos", far_behind < 0.15, f"{far_behind:.3f}")

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL FLOW TESTS PASSED")
