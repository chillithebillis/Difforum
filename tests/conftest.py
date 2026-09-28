"""Make the pack importable as `difforum` whatever its folder is called, and
provide stub ComfyUI modules so node code runs without ComfyUI or a GPU."""

import sys
import types
from pathlib import Path

import pytest
import torch

PACK = Path(__file__).resolve().parent.parent
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

if "difforum" not in sys.modules:           # load the real package, whatever the folder is called
    import importlib.util

    _spec = importlib.util.spec_from_file_location(
        "difforum", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
    _pkg = importlib.util.module_from_spec(_spec)
    sys.modules["difforum"] = _pkg
    _spec.loader.exec_module(_pkg)


class StubVAE:
    """encode/decode are identity-ish: latent = image, as channels-first."""

    def encode(self, img):
        return img[..., :3].permute(0, 3, 1, 2).clone()

    def decode(self, lat):
        return lat.permute(0, 2, 3, 1).clamp(0, 1)


def _stub_ksampler(model, seed, steps, cfg, sampler, scheduler, pos, neg, latent, denoise=1.0):
    x = latent["samples"]
    g = torch.Generator().manual_seed(int(seed) & 0x7FFFFFFF)
    noise = torch.rand(x.shape, generator=g)
    model.calls.append({"denoise": denoise, "cfg": cfg, "pos": pos, "steps": steps})
    return ({"samples": x * (1 - 0.2 * denoise) + noise * 0.2 * denoise},)


class StubModel:
    def __init__(self):
        self.calls = []


@pytest.fixture(autouse=True, scope="session")
def comfy_stubs():
    nodes_mod = types.ModuleType("nodes")
    nodes_mod.common_ksampler = _stub_ksampler
    nodes_mod.NODE_CLASS_MAPPINGS = {}
    sys.modules.setdefault("nodes", nodes_mod)
    yield


@pytest.fixture
def stub_model():
    return StubModel()


@pytest.fixture
def stub_vae():
    return StubVAE()


def gradient(h=64, w=96):
    ys, xs = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
    return torch.stack([xs, ys, (xs * ys)], dim=-1).unsqueeze(0)
