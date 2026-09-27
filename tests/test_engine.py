"""The shared feedback engine: geometry, depth tracking, cadence, anchoring."""

import torch
from conftest import gradient

from difforum.core.camera import build_camera
from difforum.core.engine import EngineConfig, FeedbackEngine, iter_feedback
from difforum.core.prompt import PromptTrack
from difforum.core.warp import affine_2d, pseudo_3d_params, warp_2d, warp_3d, warp_affine


def cam(n=24, mode="2d", **axes):
    vals = {k: [v] * n for k, v in axes.items()}
    return build_camera(vals, n, mode=mode)


def test_warp_3d_is_deterministic_and_tracks_depth():
    img = gradient(32, 48)[0]
    depth = torch.linspace(0, 1, 48).expand(32, 48).clone()
    t = torch.eye(4)
    t[0, 3] = 2.0
    t[2, 3] = -3.0
    a, ma, da = warp_3d(img, depth, t, return_depth=True)
    b, mb, db = warp_3d(img, depth, t, return_depth=True)
    assert torch.equal(a, b) and torch.equal(ma, mb) and torch.equal(da, db)
    assert da.shape == depth.shape
    assert 0.0 <= float(da.min()) and float(da.max()) <= 1.0
    assert not torch.allclose(da, depth)          # depth moved with the image


def test_affine_matches_warp_2d():
    img = gradient(24, 32)
    ref, _ = warp_2d(img, 3.0, -2.0, 7.0, 1.08, padding_mode="border")
    out, _ = warp_affine(img, affine_2d(3.0, -2.0, 7.0, 1.08, 32, 24), padding_mode="border")
    assert torch.allclose(ref, out, atol=1e-4)


def test_pseudo3d_dolly_zooms_and_orbit_pans():
    t = torch.eye(4)
    t[2, 3] = -1.5
    _tx, _ty, _a, zoom = pseudo_3d_params(t, 40.0, 256)
    assert zoom > 1.0
    from difforum.core.camera import euler_to_matrix
    r = torch.eye(4)
    r[:3, :3] = torch.as_tensor(euler_to_matrix(0.0, 1.0, 0.0))
    tx, _ty, _a, _z = pseudo_3d_params(r, 40.0, 256)
    assert abs(tx) > 1.0


def test_3d_without_depth_is_not_frozen():
    c = cam(12, mode="3d", translation_z=-1.5)
    eng = FeedbackEngine(c, EngineConfig(width=48, height=32, sharpen=0, noise=0, hole_noise=0))
    frames = [f for _i, f in iter_feedback(eng, gradient(32, 48), 12)]
    assert eng.report.mode == "pseudo3d"
    assert float((frames[-1] - frames[0]).abs().mean()) > 0.01


def test_3d_with_depth_logs_depth_per_frame():
    c = cam(8, mode="3d", translation_z=-1.0, rotation_3d_y=0.5)
    depth = torch.linspace(0, 1, 48).expand(32, 48).clone()
    eng = FeedbackEngine(c, EngineConfig(width=48, height=32), depth=depth)
    list(iter_feedback(eng, gradient(32, 48), 8))
    assert eng.report.mode == "3d"
    assert sorted(eng.depth_log) == list(range(8))


def test_cadence_crossfade_orders_frames_and_counts():
    c = cam(13, translation_x=1.0)
    eng = FeedbackEngine(c, EngineConfig(width=48, height=32, cadence=4, noise=0, hole_noise=0))
    calls = []

    def diffuse(img, f):
        calls.append(f)
        return (img * 0.9 + 0.05).clamp(0, 1)

    out = list(iter_feedback(eng, gradient(32, 48), 13, diffuse=diffuse))
    assert [f for f, _ in out] == list(range(13))
    assert calls == [4, 8, 12]
    assert eng.report.keys == 3 and eng.report.tweens == 9
    # no pop at the key: the tween before a key is close to the key itself
    d_tween = float((out[7][1] - out[8][1]).abs().mean())
    d_far = float((out[5][1] - out[8][1]).abs().mean())
    assert d_tween <= d_far + 1e-6


def _cond(v):
    return [[torch.full((1, 4, 8), float(v)), {}]]


def test_scene_anchor_releases_during_travel():
    track = PromptTrack([_cond(0), _cond(1)], [(0, "a"), (10, "b")], 20, "linear")
    assert track.scene_index(3) == 0 and track.scene_index(12) == 1
    assert 0.0 < track.transition_weight(5) < 1.0
    assert track.transition_weight(15) == 0.0
    mid = track[5][0][0]
    assert torch.allclose(mid, torch.full_like(mid, 0.5), atol=1e-6)
    from difforum.core.engine import _anchor_strength
    cfg = EngineConfig(width=8, height=8, color_coherence=0.8, anchor_mode="scene")
    assert _anchor_strength(cfg, track, 0) == 0.8
    assert _anchor_strength(cfg, track, 9) < 0.2
