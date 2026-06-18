"""Tests for the warp engine (torch on CPU, tiny tensors)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.warp import warp_2d, warp_3d, _focal  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


def gradient_img(h, w):
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32),
                            torch.arange(w, dtype=torch.float32), indexing="ij")
    r = xs / (w - 1)
    g = ys / (h - 1)
    b = torch.zeros_like(r)
    return torch.stack([r, g, b], dim=-1)


print("warp_2d:")
img = gradient_img(16, 16)

# identity -> unchanged, fully valid
out, mask = warp_2d(img, 0, 0, 0, 1.0)
check("identity unchanged", torch.allclose(out, img, atol=1e-5))
check("identity full mask", mask.mean().item() > 0.99)

# integer translation shifts content and reveals a hole strip
out, mask = warp_2d(img, translation_x=4, translation_y=0, angle=0, zoom=1.0)
# pixel (8,8) of output should equal input (8, 8-4=4)
check("translate x content", torch.allclose(out[8, 8], img[8, 4], atol=1e-4),
      f"{out[8,8].tolist()} vs {img[8,4].tolist()}")
check("translate reveals hole", mask[8, 0].item() < 0.5)

# shape handling: batched
bimg = img.unsqueeze(0).expand(3, 16, 16, 3)
out, mask = warp_2d(bimg, 2, 2, 0, 1.0)
check("batched shape", out.shape == (3, 16, 16, 3) and mask.shape == (3, 16, 16, 1))

# zoom magnifies (center stays, corners pull in)
out, _ = warp_2d(img, 0, 0, 0, 2.0)
check("zoom runs", out.shape == img.shape)

print("warp_3d:")
img = gradient_img(32, 32)
depth = torch.ones(32, 32)  # flat plane

# focal sanity
check("focal positive", _focal(32, 40) > 0)

# identity transform -> (almost) unchanged where filled, mask mostly valid
I = torch.eye(4)
out, mask = warp_3d(img, depth, I, fov_deg=40, near=10, far=10)
filled = mask.squeeze(-1) > 0.5
check("identity 3d mostly filled", filled.float().mean().item() > 0.9,
      f"filled={filled.float().mean().item():.3f}")
# on filled pixels the color should match the source
diff = (out - img).abs().sum(-1)[filled]
check("identity 3d colors match", diff.mean().item() < 1e-3, f"diff={diff.mean().item():.4f}")

# pure x-translation with flat depth shifts horizontally by f*Tx/Z
Z = 10.0
f = _focal(32, 40)
Tx = Z * 3.0 / f  # aim for a 3px shift
Tm = torch.eye(4)
Tm[0, 3] = Tx
out, mask = warp_3d(img, depth, Tm, fov_deg=40, near=Z, far=Z)
# content moves: output near center should equal a source pixel ~3px to the left
ok_shift = torch.allclose(out[16, 16], img[16, 13], atol=0.05)
check("3d x-translation shifts ~3px", ok_shift, f"{out[16,16].tolist()} vs {img[16,13].tolist()}")

# forward dolly (Tz>0 pushes scene away) shrinks content toward center
Tm = torch.eye(4)
Tm[2, 3] = 20.0
out, mask = warp_3d(img, depth, Tm, fov_deg=40, near=Z, far=Z)
check("dolly produces holes at edges", mask.squeeze(-1)[16, 0].item() < 0.5)

print("hole fill (3d):")
# an expanding dolly leaves speckle holes; fill should clean colour, keep mask
img = gradient_img(48, 48)
depth = torch.ones(48, 48)
Tm = torch.eye(4)
Tm[2, 3] = -8.0  # pull scene closer -> pixels spread -> holes
filled, mask = warp_3d(img, depth, Tm, fov_deg=50, near=10, far=10, fill_holes=True)
raw, mask2 = warp_3d(img, depth, Tm, fov_deg=50, near=10, far=10, fill_holes=False)
m = mask.squeeze(-1) > 0.5
# masks identical (fill must not change occlusion)
check("fill keeps mask honest", torch.equal(mask, mask2))
# filled version has fewer pure-black pixels than the raw splat (holes cleaned)
black_filled = ((filled.sum(-1) == 0)).sum().item()
black_raw = ((raw.sum(-1) == 0)).sum().item()
check("fill reduces black holes", black_filled < black_raw, f"{black_filled} vs {black_raw}")
check("no NaNs after fill", not torch.isnan(filled).any())

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL WARP TESTS PASSED")
