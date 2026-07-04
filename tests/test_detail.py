"""Tests for detail preservation (torch CPU): sharpen, noise, contrast."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.detail import (  # noqa: E402
    NOISE_MODES,
    add_noise,
    adjust_contrast,
    detail_guard,
    sharpen,
)

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


def edge_energy(x):
    """Mean absolute horizontal gradient (proxy for sharpness)."""
    return float((x[..., 1:, :] - x[..., :-1, :]).abs().mean())


torch.manual_seed(0)
# soft blob: a blurred random image (something sharpening can bite into)
raw = torch.rand(1, 32, 32, 3)
k = torch.ones(1, 1, 5, 5) / 25.0
soft = torch.nn.functional.conv2d(
    raw.permute(0, 3, 1, 2).reshape(3, 1, 32, 32), k, padding=2
).reshape(1, 3, 32, 32).permute(0, 2, 3, 1)

print("sharpen:")
sh = sharpen(soft, 0.8)
check("keeps shape", sh.shape == soft.shape)
check("raises edge energy", edge_energy(sh) > edge_energy(soft))
check("amount 0 passthrough", torch.allclose(sharpen(soft, 0.0), soft))
check("in range", float(sh.min()) >= 0.0 and float(sh.max()) <= 1.0)

print("noise:")
for mode in NOISE_MODES:
    n = add_noise(soft, 0.05, seed=3, mode=mode)
    check(f"{mode} changes image", not torch.allclose(n, soft))
    check(f"{mode} in range", float(n.min()) >= 0.0 and float(n.max()) <= 1.0)
check("deterministic w/ seed", torch.allclose(add_noise(soft, 0.05, seed=7),
                                              add_noise(soft, 0.05, seed=7)))
check("amount 0 passthrough", torch.allclose(add_noise(soft, 0.0), soft))
try:
    add_noise(soft, 0.1, mode="nope")
    _raised = False
except ValueError:
    _raised = True
check("bad mode raises", _raised)

print("contrast:")
c = adjust_contrast(soft, 1.3)
check("raises contrast (std up)", float(c.std()) > float(soft.std()))
check("factor 1 passthrough", torch.allclose(adjust_contrast(soft, 1.0), soft))

print("detail_guard:")
g = detail_guard(soft, sharpen_amount=0.5, noise_amount=0.03, contrast=1.05, seed=1)
check("combo keeps shape + range", g.shape == soft.shape
      and float(g.min()) >= 0.0 and float(g.max()) <= 1.0)
check("combo changes image", not torch.allclose(g, soft))
check("all-off passthrough", torch.allclose(detail_guard(soft), soft))
check("3d input", detail_guard(soft[0], sharpen_amount=0.3).shape == (32, 32, 3))

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL DETAIL TESTS PASSED")
