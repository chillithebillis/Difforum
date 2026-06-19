"""Tests for the VJ look module (torch CPU): grade, glow, chroma, presets."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.look import (  # noqa: E402
    LOOK_PRESETS,
    apply_look,
    chroma_shift,
    color_grade,
    glow,
    grain,
    vignette,
)

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


torch.manual_seed(0)
img = torch.rand(1, 32, 32, 3)

print("colour grade:")
check("neutral grade is identity", torch.allclose(color_grade(img), img, atol=1e-5))
check("grade keeps shape + range", color_grade(img, exposure=0.5).shape == img.shape)
gray = color_grade(img, saturation=0.0)
check("saturation 0 = grayscale", torch.allclose(gray[..., 0], gray[..., 1], atol=1e-3)
      and torch.allclose(gray[..., 1], gray[..., 2], atol=1e-3))
bright = color_grade(img, exposure=1.0)
check("exposure brightens", bright.mean() > img.mean())
check("grade in range", float(color_grade(img, contrast=2.0).min()) >= 0.0
      and float(color_grade(img, contrast=2.0).max()) <= 1.0)

print("glow:")
spot = torch.zeros(1, 32, 32, 3)
spot[0, 16, 16, :] = 1.0
g = glow(spot, threshold=0.5, radius=6, intensity=1.0)
check("glow keeps shape", g.shape == spot.shape)
check("glow spreads brightness", (g[0, 16, 18] > 0).any())
check("glow intensity 0 passthrough", torch.allclose(glow(img, intensity=0.0), img))
check("glow in range", float(g.min()) >= 0.0 and float(g.max()) <= 1.0)

print("chroma / vignette / grain:")
c = chroma_shift(img, amount=3.0, angle=0.0)
check("chroma keeps shape", c.shape == img.shape)
check("chroma shifts (changes image)", not torch.allclose(c, img))
check("chroma 0 passthrough", torch.allclose(chroma_shift(img, amount=0.0), img))
v = vignette(img, amount=0.5)
check("vignette darkens edges", v[0, 0, 0].mean() < img[0, 0, 0].mean())
check("vignette keeps center", torch.allclose(v[0, 16, 16], img[0, 16, 16], atol=0.05))
check("grain changes image", not torch.allclose(grain(img, amount=0.1), img))
check("grain deterministic w/ seed", torch.allclose(grain(img, 0.1, seed=5), grain(img, 0.1, seed=5)))

print("presets:")
for p in LOOK_PRESETS:
    out = apply_look(img, preset=p, intensity=1.0)
    check(f"{p} keeps shape + range", out.shape == img.shape
          and float(out.min()) >= 0.0 and float(out.max()) <= 1.0)
check("none passthrough", torch.allclose(apply_look(img, "none"), img))
check("intensity 0 passthrough", torch.allclose(apply_look(img, "neon", 0.0), img))
check("preset changes image", not torch.allclose(apply_look(img, "neon", 1.0), img))
check("3d input ok", apply_look(img[0], "neon").shape == (32, 32, 3))
batch = img.expand(4, 32, 32, 3)
check("batch ok", apply_look(batch, "cinematic").shape == (4, 32, 32, 3))
try:
    apply_look(img, "nope")
    _raised = False
except ValueError:
    _raised = True
check("invalid preset raises", _raised)

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL LOOK TESTS PASSED")
