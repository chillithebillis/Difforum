"""v1 nodes end to end, with stub MODEL / VAE / CLIP."""

import json

import pytest
import torch
from conftest import StubModel, StubVAE, gradient

import difforum
from difforum.core.direction import default_timeline
from difforum.nodes.bridges import (
    DifforumCameraPrompt, DifforumGuideFrames, DifforumH3Shot, DifforumKeyframes,
)
from difforum.nodes.curves import DifforumPromptTravel, DifforumSchedule
from difforum.nodes.direction import DifforumCamera, DifforumDirector, DifforumStoryboard
from difforum.nodes.post import DifforumLoop
from difforum.nodes.render import DifforumFeedbackSampler, DifforumRenderOptions
from difforum.nodes.setup import DifforumSetup


def setup(target="feedback (SD/SDXL/Flux)", seconds=2.0, fps=24.0, edge=128):
    return DifforumSetup().build(target, "seconds", seconds, fps, "16:9 landscape", edge, 0)


class StubClip:
    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        v = float(len(tokens) % 7)
        return [[torch.full((1, 4, 8), v), {"pooled_output": torch.full((1, 8), v)}]]


def director(params, tl=None, mode="2d", clip=None):
    tl = tl or default_timeline(params["max_frames"])
    return DifforumDirector().run(params, json.dumps(tl), mode, "cinematic", 1.0, 1.0, 0.0,
                                  0.0, 0, clip=clip)["result"]


def test_registry():
    v1 = [k for k in difforum.NODE_CLASS_MAPPINGS if k.startswith("Difforum_")]
    assert len(v1) == 30
    legacy = difforum.NODE_CLASS_MAPPINGS["DifforumFeedbackSampler"]
    assert legacy.DEPRECATED and legacy.CATEGORY == "Difforum/legacy"
    for k in v1:
        cls = difforum.NODE_CLASS_MAPPINGS[k]
        cls.INPUT_TYPES()
        assert cls.CATEGORY.startswith("Difforum/")
        assert len(cls.RETURN_TYPES) == len(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES))


@pytest.mark.parametrize("target,rule,mult", [
    ("LTX-2 / 2.5", (8, 1), 32), ("MiniMax H3", (17, 5), 32), ("Wan 2.x", (4, 1), 16),
])
def test_setup_snaps_to_model_grid(target, rule, mult):
    params, w, h, frames, fps, _ = setup(target, seconds=5.0, fps=30.0, edge=1000)
    a, b = rule
    assert (frames - b) % a == 0
    assert w % mult == 0 and h % mult == 0
    if target == "MiniMax H3":
        assert fps == 24.0


def test_director_storyboard_and_sampler():
    params = setup()[0]
    bundle, camera, strength, prompts, text, info = director(params, clip=StubClip())
    assert len(camera.deltas) == params["max_frames"] and len(strength) == params["max_frames"]
    assert text.startswith("The camera")
    assert len(prompts) == params["max_frames"]

    sheet, frames, path, sb_info = DifforumStoryboard().run(gradient(72, 128), 8, 6, 1.0,
                                                            direction=bundle)
    assert frames.shape[1:] == (72, 128, 3) and path.shape[-1] == 3

    model = StubModel()
    out, depth, report = DifforumFeedbackSampler().run(
        model, _c(), _c(), StubVAE(), gradient(72, 128), 4, 5.0, "euler", "normal", 2,
        direction=bundle)
    assert out.shape == (params["max_frames"], 72, 128, 3)
    assert depth.shape[-1] == 3
    assert "keys=" in report
    # the energy curve reached the sampler, and prompt travel conditioning too
    assert len({round(c["denoise"], 3) for c in model.calls}) > 1
    assert all(c["pos"] is not None for c in model.calls)
    # steps scale with the energy (Deforum-style): never more than asked, fewer when denoise < 1
    assert all(1 <= c["steps"] <= 4 for c in model.calls)
    assert any(c["steps"] < 4 for c in model.calls if c["denoise"] < 0.75)
    assert "s/frame" in report


def test_sampler_without_director():
    params = setup(seconds=0.5)[0]
    cam = DifforumCamera().run(params, "0: zoom_in", "2d", 1.0, 1.0, "off", 0, 3)[0]
    sched = DifforumSchedule().run(params, "0:(0.4)", "linear")[0]
    opts = DifforumRenderOptions().run(**{k: s[1]["default"] for k, s in
                                          DifforumRenderOptions.INPUT_TYPES()["required"].items()})[0]
    opts.update(anchor_mode="first", end_frame=6)
    out = DifforumFeedbackSampler().run(StubModel(), _c(), _c(), StubVAE(), gradient(72, 128), 2,
                                        5.0, "euler", "normal", 1, options=opts, params=params,
                                        camera=cam, strength=sched)[0]
    assert out.shape[0] == 6


def test_bridges():
    params = setup("MiniMax H3", seconds=6.0)[0]
    bundle = director(params)[0]
    guides, masks, _ = DifforumGuideFrames().run(gradient(72, 128), "1 = generate (VACE / LTX / H3)",
                                                 "gray", direction=bundle)
    n = params["max_frames"]
    assert guides.shape[0] == n and masks.shape == (n, guides.shape[1], guides.shape[2])
    assert float(masks[0].max()) == 0.0          # frame 0 has nothing to generate

    keys, idx, sparse, first, last, length, _km, _ = DifforumKeyframes().run(
        guides, "LTX-2 / 2.5 (8k+1)", 1.0, 24.0)
    ids = [int(i) for i in idx.split(",")]
    assert (length - 1) % 8 == 0 and ids[0] == 0 and ids[-1] == length - 1
    assert all(i % 8 == 0 for i in ids[:-1])
    assert keys.shape[0] == len(ids) and sparse.shape[0] == length
    assert float(sparse[1].abs().sum()) == 0.0

    first, last, length, w, h, prompt, segments, last_mask, info = DifforumH3Shot().run(
        guides, "124 (~5s)", 0, direction=bundle, shot_description="A forest.")
    assert (length - 5) % 17 == 0 and w % 32 == 0 and h % 32 == 0
    assert segments >= 2 and prompt.startswith("A forest. The camera")

    text = DifforumCameraPrompt().run("prompt suffix", direction=bundle)[0]
    assert text.startswith("Camera:") and "live-action" in text


def test_prompt_travel_seconds_syntax():
    params = setup(seconds=4.0)[0]
    track, first, batched, info = DifforumPromptTravel().run(
        params, StubClip(), "0: a\n2s: bbb\n3.5s: cc", "linear", build_batched=True)
    assert [f for f, _ in track.keyframes] == [0, 48, 84]
    assert batched[0][0].shape[0] == params["max_frames"]


def test_loop_methods():
    frames = torch.rand(24, 16, 16, 3)
    assert DifforumLoop().run(frames, "keep settled lap", 8, 4)[0].shape[0] == 8
    assert DifforumLoop().run(frames, "ping-pong", 0, 4)[0].shape[0] > 24
    assert DifforumLoop().run(frames, "flow crossfade", 0, 4)[0].shape[0] == 20


def _c():
    return [[torch.zeros(1, 4, 8), {}]]


def test_live_sampler_ring_buffer():
    from difforum.nodes.render import DifforumLiveSampler
    params = setup(seconds=0.5)[0]
    bundle = director(params)[0]
    frames, report = DifforumLiveSampler().run(
        StubModel(), _c(), _c(), StubVAE(), gradient(72, 128), 30, 1, 1.0, "euler", "normal", 2,
        0.0, direction=bundle, keep_frames=10, live_preview=False)
    assert frames.shape[0] == 10 and "fps measured" in report


def test_guide_frames_3d_with_depth_and_export(tmp_path, monkeypatch):
    from difforum.nodes.export import DifforumCameraExport, DifforumCameraImport
    params = setup(seconds=1.0)[0]
    tl = default_timeline(params["max_frames"])
    tl["camera"][0]["move"] = "dolly_in"
    bundle = director(params, tl=tl, mode="3d")[0]
    depth = torch.linspace(0, 1, 128).expand(72, 128)[None, ..., None].expand(1, 72, 128, 3)
    guides, masks, info = DifforumGuideFrames().run(gradient(72, 128), "1 = keep", "stretch edge",
                                                    direction=bundle, depth=depth)
    assert "(3d)" in info and float(masks[-1].mean()) < 1.0

    monkeypatch.chdir(tmp_path)
    res = DifforumCameraExport().run("all", "shot", 0.1, direction=bundle)["result"]
    paths = res[0].splitlines()
    assert len(paths) == 3 and all(p.endswith((".json", ".jsx", "_blender.py")) for p in paths)
    cam, _ = DifforumCameraImport().run("", params, "match frames", json_text=res[1])
    assert len(cam.deltas) == params["max_frames"]


def test_2d_director_keeps_depth_moves_alive():
    params = setup(seconds=1.0)[0]
    tl = default_timeline(params["max_frames"])
    tl["camera"] = [{"start": 0, "move": "dolly_in", "speed": 1.0, "intensity": 1.0}]
    bundle, camera = director(params, tl=tl, mode="2d")[:2]
    assert camera.mode == "3d" and camera.flat
    frames = DifforumStoryboard().run(gradient(72, 128), 8, 6, 1.0, direction=bundle)[1]
    assert float((frames[-1] - frames[0]).abs().mean()) > 0.01
    depth = torch.full((1, 72, 128, 3), 0.5)
    info = DifforumStoryboard().run(gradient(72, 128), 8, 6, 1.0, direction=bundle, depth=depth)[3]
    assert "(pseudo3d)" in info          # flat = 2d stays 2d even with a depth map


def test_h3_guides_chain_and_limit(monkeypatch):
    import nodes as stub_nodes

    from difforum.nodes.bridges import DifforumH3Guides

    calls = []

    class FakeAddGuide:
        @classmethod
        def execute(cls, positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None):
            calls.append((frame_idx, audio is not None))
            return (positive + [frame_idx],)

    monkeypatch.setitem(stub_nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3AddGuide", FakeAddGuide)
    keys = torch.rand(6, 32, 32, 3)
    pos, info = DifforumH3Guides().run([], {"samples": None}, object(), keys, "0,17,34,51,68,123",
                                       4, False, audio_vae=object(), audio={"waveform": 1})
    assert [c[0] for c in calls] == [0, 34, 68, 123] and calls[0][1] and not calls[1][1]
    calls.clear()
    DifforumH3Guides().run([], {"samples": None}, object(), keys, "0,17,34,51,68,123", 8, True)
    assert [c[0] for c in calls] == [17, 34, 51, 68, 123]


def test_fill_reveal_keeps_known_pixels(monkeypatch):
    import nodes as stub_nodes

    from difforum.nodes.bridges import DifforumFillReveal

    class FakeInpaint:
        FUNCTION = "encode"

        def encode(self, positive, negative, pixels, vae, mask, noise_mask=True):
            return (positive, negative, {"samples": vae.encode(pixels), "noise_mask": mask})

    monkeypatch.setitem(stub_nodes.NODE_CLASS_MAPPINGS, "InpaintModelConditioning", FakeInpaint)
    imgs = torch.full((3, 32, 48, 3), 0.3)
    masks = torch.zeros(3, 32, 48)
    masks[:, :, 40:] = 1.0                       # right edge revealed
    out, info = DifforumFillReveal().run(imgs, masks, StubModel(), _c(), _c(), StubVAE(), "last",
                                         2, 5.0, "euler", "normal", 4, 3, 0)
    assert torch.equal(out[:2], imgs[:2])                 # only the last frame touched
    assert torch.equal(out[2, :, :30], imgs[2, :, :30])   # known pixels untouched
    assert not torch.allclose(out[2, :, 44:], imgs[2, :, 44:])
    assert "filled" in info


def test_keyframes_animatic_look_mix():
    from difforum.nodes.direction import DifforumAnimatic, DifforumKeyframeImages
    from difforum.nodes.post import DifforumLookMix
    params = setup(seconds=2.0)[0]
    tl = default_timeline(params["max_frames"])
    tl["keys"] = [{"start": 0, "label": "A"}, {"start": 24, "label": "B"}, {"start": 47, "label": "C"}]
    bundle = director(params, tl=tl)[0]
    imgs = torch.rand(3, 50, 60, 3)
    keys, idx, info = DifforumKeyframeImages().run(imgs, direction=bundle)
    assert idx == "0,24,47" and keys.shape[1:] == (72, 128, 3)
    assert DifforumKeyframeImages().run(imgs, direction=bundle, times="0, 1s, 1.5s")[1] == "0,24,36"

    frames, fps, ainfo = DifforumAnimatic().run(bundle, 0.5, True, key_images=keys, key_indices=idx)
    assert frames.shape[0] == params["max_frames"] and fps == 24.0 and "3 key" in ainfo

    out = DifforumFeedbackSampler().run(StubModel(), _c(), _c(), StubVAE(), gradient(72, 128), 2, 5.0,
                                        "euler", "normal", 1, direction=bundle, key_images=keys,
                                        key_indices=idx, key_pull=1.0)[0]
    assert out.shape[0] == params["max_frames"]

    video = torch.rand(48, 32, 32, 3)
    look = torch.rand(24, 16, 16, 3)
    for mode in ("detail transfer", "colour + detail", "flicker cuts", "crossfade", "none"):
        (res,) = DifforumLookMix().run(video, mode, 0.6, 2, 3, 12.0, 24.0, 0, look_pass=look)
        assert res.shape == video.shape
    (stepped,) = DifforumLookMix().run(video, "none", 0.6, 2, 3, 12.0, 24.0, 0)
    assert torch.equal(stepped[0], stepped[1]) and not torch.equal(stepped[1], stepped[2])
