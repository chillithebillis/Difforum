"""Tests for symmetry / kaleidoscope and echo trails (torch CPU)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.effects import echo_trails  # noqa: E402
from core.symmetry import SYMMETRY_MODES, apply_symmetry  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


torch.manual_seed(0)
img = torch.rand(1, 16, 16, 3)

print("symmetry shapes + modes:")
for mode in SYMMETRY_MODES:
    out = apply_symmetry(img, mode=mode, segments=6)
    check(f"{mode} keeps shape", out.shape == img.shape)
check("none passthrough", torch.allclose(apply_symmetry(img, "none"), img))
check("mix 0 passthrough", torch.allclose(apply_symmetry(img, "mirror_h", mix=0.0), img))
check("3d input", apply_symmetry(img[0], "mirror_h").shape == (16, 16, 3))

print("mirror_h is left-right symmetric:")
mh = apply_symmetry(img, "mirror_h")
check("mirror_h equals its flip", torch.allclose(mh, torch.flip(mh, dims=[2]), atol=1e-5))

print("mirror_v is top-bottom symmetric:")
mv = apply_symmetry(img, "mirror_v")
check("mirror_v equals its flip", torch.allclose(mv, torch.flip(mv, dims=[1]), atol=1e-5))

print("mirror_quad is symmetric on both axes:")
mq = apply_symmetry(img, "mirror_quad")
check("quad h-symmetric", torch.allclose(mq, torch.flip(mq, dims=[2]), atol=1e-5))
check("quad v-symmetric", torch.allclose(mq, torch.flip(mq, dims=[1]), atol=1e-5))

print("kaleidoscope:")
k = apply_symmetry(img, "kaleidoscope", segments=8)
check("kaleidoscope in range", float(k.min()) >= 0.0 and float(k.max()) <= 1.0)
check("kaleidoscope changes image", not torch.allclose(k, img, atol=1e-3))

try:
    apply_symmetry(img, "nope")
    _raised = False
except ValueError:
    _raised = True
check("invalid mode raises", _raised)

print("echo trails:")
# a single bright pixel sweeping across frames should leave a trail behind it
frames = torch.zeros(6, 8, 8, 3)
for i in range(6):
    frames[i, 4, i, :] = 1.0
out = echo_trails(frames, decay=0.7, mix=0.6)
check("echo keeps shape", out.shape == frames.shape)
check("echo in range", float(out.min()) >= 0.0 and float(out.max()) <= 1.0)
# at the last frame, an earlier position should now be lit (the trail)
check("echo leaves a trail", float(out[5, 4, 2].mean()) > 0.0,
      f"value {float(out[5,4,2].mean()):.4f}")
check("echo mix 0 passthrough", torch.allclose(echo_trails(frames, mix=0.0), frames))

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL EFFECTS TESTS PASSED")
