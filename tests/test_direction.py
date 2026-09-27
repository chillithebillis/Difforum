"""Timeline model: parsing, building, words."""

import json

from difforum.core.camera_presets import CAMERA_PRESETS, MOVE_INFO, preset_schedules
from difforum.core.direction import (
    build_direction, default_timeline, describe_blocks, parse_timeline, ui_catalog,
)


def test_every_preset_has_ui_metadata_and_builds():
    assert set(CAMERA_PRESETS) == set(MOVE_INFO)
    for p in CAMERA_PRESETS:
        assert len(preset_schedules(p, 1.0, 1.0)) == 7


def test_default_timeline_is_depth_free_and_fits():
    d = build_direction(default_timeline(120), 120, 24.0, mode="2d")
    assert d.warnings == []
    assert len(d.strength) == 120 and len(d.lens) == 120
    assert all(len(v) == 120 for v in d.axes.values())
    assert d.prompts[0][0] == 0 and len(d.prompts) == 4


def test_legacy_scene_list_is_converted():
    old = [{"frame_start": 0, "mood": "calm", "camera": "dolly_in", "prompt": "a"},
           {"frame_start": 40, "mood": "climax", "camera": "spiral", "prompt": "b"}]
    tl = parse_timeline(json.dumps(old))
    assert [c["move"] for c in tl["camera"]] == ["dolly_in", "spiral"]
    assert [s["start"] for s in tl["scenes"]] == [0, 40]


def test_energy_points_override_moods_and_blocks_past_end_warn():
    tl = default_timeline(120)
    tl["energy"] = [[0, 0.3], [60, 0.7]]
    tl["camera"].append({"start": 500, "move": "shake"})
    d = build_direction(tl, 120, 24.0)
    assert abs(d.strength[0] - 0.3) < 1e-6 and abs(d.strength[-1] - 0.7) < 1e-6
    assert any("after the clip ends" in w for w in d.warnings)


def test_depth_moves_warn_in_2d():
    tl = default_timeline(96)
    tl["camera"][0]["move"] = "orbit_left"
    d = build_direction(tl, 96, 24.0, mode="2d")
    assert any("Orbit left" in w for w in d.warnings)


def test_audio_reaction_changes_the_camera():
    tl = default_timeline(48)
    base = build_direction(tl, 48, 24.0)
    tl["camera"][0]["react"] = "beat_pulse"
    curves = {k: [0.0] * 48 for k in ("amp", "low", "mid", "high", "onset", "beat")}
    curves["beat"][5] = 1.0
    d = build_direction(tl, 48, 24.0, audio_curves=curves)
    assert d.axes["zoom"][5] > base.axes["zoom"][5]
    assert d.axes["zoom"][6] == base.axes["zoom"][6]


def test_variation_zero_is_exact_and_seeded():
    tl = default_timeline(60)
    a = build_direction(tl, 60, 24.0)
    b = build_direction(tl, 60, 24.0, variation=0.0, variation_seed=7)
    assert a.axes == b.axes
    c1 = build_direction(tl, 60, 24.0, variation=0.5, variation_seed=3)
    c2 = build_direction(tl, 60, 24.0, variation=0.5, variation_seed=3)
    assert c1.axes == c2.axes and c1.axes != a.axes


def test_words():
    text = describe_blocks([{"start": 0, "move": "dolly_in", "speed": 0.4, "intensity": 0.8},
                            {"start": 30, "move": "orbit_right", "speed": 1, "intensity": 1},
                            {"start": 60, "move": "crane_up", "speed": 2, "intensity": 1.2}], 90, 24)
    assert text.startswith("The camera pushes in very slowly")
    assert "and finally cranes up" in text


def test_catalog_for_the_ui():
    cat = ui_catalog()
    ids = [m["id"] for m in cat["moves"]]
    assert ids == list(CAMERA_PRESETS)
    assert {m["id"] for m in cat["moods"]} >= {"calm", "climax"}
