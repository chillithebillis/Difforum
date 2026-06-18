"""Tests for color coherence (torch CPU): RGB + LAB modes."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.color import _lab_to_rgb, _rgb_to_lab, match_color  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


torch.manual_seed(0)
image = (torch.randn(1, 32, 32, 3) * 0.05 + 0.2).clamp(0, 1)
reference = (torch.randn(1, 32, 32, 3) * 0.18 + 0.6).clamp(0, 1)

print("lab round-trip:")
rgb = torch.rand(1, 16, 16, 3)
back = _lab_to_rgb(_rgb_to_lab(rgb))
check("rgb->lab->rgb identity", torch.allclose(rgb, back, atol=2e-3),
      f"max diff {float((rgb-back).abs().max()):.5f}")

print("rgb mode:")
out = match_color(image, reference, strength=1.0, mode="rgb")
check("rgb mean matches", torch.allclose(out.mean(dim=(1, 2)), reference.mean(dim=(1, 2)), atol=0.03))
check("rgb std matches", torch.allclose(out.std(dim=(1, 2)), reference.std(dim=(1, 2)), atol=0.03))

print("lab mode:")
out = match_color(image, reference, strength=1.0, mode="lab")
# in LAB space the matched image's LAB stats should approach the reference's
out_l = _rgb_to_lab(out)
ref_l = _rgb_to_lab(reference)
check("lab mean matches", torch.allclose(out_l.mean(dim=(1, 2)), ref_l.mean(dim=(1, 2)), atol=2.0),
      f"{out_l.mean(dim=(1,2)).tolist()} vs {ref_l.mean(dim=(1,2)).tolist()}")
check("lab output in range", float(out.min()) >= 0.0 and float(out.max()) <= 1.0)
# lab match should brighten the dark image toward the bright reference
check("lab brightens toward ref", out.mean() > image.mean())

print("common:")
check("none mode passthrough", torch.allclose(match_color(image, reference, 1.0, "none"), image))
check("strength 0 unchanged", torch.allclose(match_color(image, reference, 0.0, "lab"), image))
single = match_color(image[0], reference[0], strength=1.0, mode="lab")
check("3d input shape", single.shape == (32, 32, 3))
batch = image.expand(4, 32, 32, 3)
check("batch + single ref", match_color(batch, reference, 1.0, "lab").shape == (4, 32, 32, 3))

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL COLOR TESTS PASSED")
