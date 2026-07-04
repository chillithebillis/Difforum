"""
Effect-device helper. ComfyUI IMAGE tensors travel on the CPU, so by default
every per-frame warp / symmetry / sharpen in a feedback loop runs on the CPU
even when a GPU is sitting idle. This picks the best device (cuda > mps > cpu)
for the pixel-effect chain, with a permanent fallback to CPU if an op is not
supported there (older MPS builds lack some grid_sample modes).
"""

from __future__ import annotations

import torch


def fx_device() -> torch.device:
    """Best device for per-frame pixel effects."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FxRunner:
    """Run a per-frame effect function on the fx device, falling back to CPU
    for the rest of the session on the first failure."""

    def __init__(self):
        self.device = fx_device()

    def __call__(self, fn, image: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cpu":
            return fn(image)
        try:
            out = fn(image.to(self.device))
            return out.to("cpu")
        except Exception as e:  # unsupported op on this backend -> stay on CPU
            print(f"[Difforum] effect chain fell back to CPU ({e})")
            self.device = torch.device("cpu")
            return fn(image)
