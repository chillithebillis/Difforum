"""Shot scripts, external timelines and keyframe assets."""

import json
import sys
import types

import numpy as np
import pytest
from PIL import Image
from test_nodes import director, setup

from difforum.core.script import merge_timelines, parse_time, script_to_timeline
from difforum.nodes.direction import DifforumDirector
from difforum.nodes.orchestrate import DifforumKeyframeAssets, DifforumShotScript

SCRIPT = """# a comment
0s    | calm  | dolly_in slow small | misty forest at dawn, light shafts
4.5s  | build | orbit left fast 35mm | glowing roots and moss
00:09 | key: the light breaks
@12s  the mist lifts
10s | climax | push in x1.3 | explosion of light | energy 0.7"""


@pytest.fixture
def input_dir(tmp_path, monkeypatch):
    fp = types.ModuleType("folder_paths")
    fp.get_input_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    return tmp_path


def test_parse_time():
    assert [parse_time(t, 24) for t in ("0s", "4.5s", "00:09", "1:02.5", "f96", "96", "x")] == \
        [0, 108, 216, 1500, 96, 96, None]


def test_script_lines():
    tl, notes = script_to_timeline(SCRIPT, 24, 360)
    assert [s["start"] for s in tl["scenes"]] == [0, 108, 240, 288]
    assert [s["mood"] for s in tl["scenes"]] == ["calm", "build", "climax", "calm"]
    cam = {c["start"]: c for c in tl["camera"]}
    assert cam[0]["move"] == "dolly_in" and cam[0]["speed"] == 0.6 and cam[0]["intensity"] == 0.5
    assert cam[108]["move"] == "orbit_left" and cam[108]["speed"] == 1.5 and cam[108]["lens"] == 35.0
    assert cam[240]["move"] == "dolly_in" and cam[240]["speed"] == 1.3     # alias push in
    assert tl["keys"] == [{"start": 216, "label": "the light breaks"}]
    assert tl["energy"] == [(240, 0.7)] and not notes


def test_script_csv_untimed_and_json():
    tl, notes = script_to_timeline("a lone tree\nthe storm\nsunrise", 24, 120)
    assert [s["start"] for s in tl["scenes"]] == [0, 40, 80] and "spread evenly" in notes[0]
    csv_text = "time,mood,camera,prompt,key\n0s,calm,zoom_in,forest,\n3s,build,pan right slow,roots,beat"
    tl, _ = script_to_timeline(csv_text, 24, 120)
    assert [c["move"] for c in tl["camera"]] == ["zoom_in", "pan_right"] and tl["keys"][0]["start"] == 72
    tl2, _ = script_to_timeline(json.dumps(tl), 24, 120)
    assert tl2 == tl


def test_merge_modes():
    drawn, _ = script_to_timeline("0s | calm | zoom_in | A\n2s | key: k", 24, 120)
    ext, _ = script_to_timeline("0s | build | B\n1s | tense | C", 24, 120)
    m = merge_timelines(drawn, ext, "text only (keep drawn camera)")
    assert [s["prompt"] for s in m["scenes"]] == ["B", "C"] and m["camera"][0]["move"] == "zoom_in"
    assert m["keys"] == drawn["keys"]
    m = merge_timelines(drawn, ext, "replace")
    assert m["camera"] == [] and m["keys"] == []
    m = merge_timelines(drawn, ext, "add to drawn")
    assert [s["prompt"] for s in m["scenes"]] == ["B", "C"]
    m = merge_timelines(drawn, ext, "camera only (keep drawn text)")
    assert m["scenes"] == drawn["scenes"] and m["camera"] == []


def test_director_timeline_in():
    p = setup(seconds=5.0)[0]
    tl = json.dumps(script_to_timeline("0s | calm | zoom_in | drawn prompt", 24, 120)[0])
    out = DifforumDirector().run(p, tl, "2d", "cinematic", 1.0, 1.0, 0.0, 0.0, 0,
                                 timeline_in="0s | build | outside prompt\n2s | key: hit",
                                 external="text only (keep drawn camera)")
    bundle = out["result"][0]
    baked = json.loads(out["ui"]["difforum_timeline"][0])
    assert baked["scenes"][0]["prompt"] == "outside prompt" and baked["camera"][0]["move"] == "zoom_in"
    assert bundle.direction.keys[0]["start"] == 48
    assert "timeline_in" in out["result"][-1]
    plain = director(p)
    assert plain is not None


def test_shot_script_node(input_dir):
    p = setup(seconds=5.0)[0]
    (input_dir / "shots").mkdir()
    (input_dir / "shots" / "s1.txt").write_text("0s | calm | still | from file", encoding="utf-8")
    node = DifforumShotScript()
    tl, guide, info = node.run("0s | calm | box", "shots/s1.txt", params=p)["result"]
    assert json.loads(tl)["scenes"][0]["prompt"] == "from file" and "TIME | MOOD" in guide
    tl, *_ = node.run("0s | calm | box", "", params=p, script_in="0s | dream | llm text")["result"]
    assert json.loads(tl)["scenes"][0]["prompt"] == "llm text"
    with pytest.raises(ValueError):
        node.run("", "../outside.txt", params=p)


def test_keyframe_assets(input_dir):
    p = setup(seconds=5.0)[0]                       # 120 frames, 128 px long edge
    keys = input_dir / "difforum_keys"
    keys.mkdir()
    for name, color, size in (("4s_mid", 120, (300, 100)), ("0s_open", 30, (64, 64)), ("f119_end", 220, (90, 160))):
        Image.fromarray(np.full((size[1], size[0], 3), color, np.uint8)).save(keys / f"{name}.png")
    node = DifforumKeyframeAssets()
    k, idx, first, last, masks, info = node.run("difforum_keys", "filename", "cover (crop)", params=p)["result"]
    assert idx == "0,96,119" and k.shape[1:] == (p["height"], p["width"], 3)
    assert abs(float(first.mean()) - 30 / 255) < 0.02 and abs(float(last.mean()) - 220 / 255) < 0.02
    assert float(masks.sum()) == 0
    _k, idx, *_r = node.run("difforum_keys", "spread evenly", "contain (pad gray)", params=p)["result"]
    masks = _r[2]
    assert idx == "0,60,119" and float(masks.sum()) > 0
    d = director(p, tl={"scenes": [], "camera": [], "keys": [{"start": 10}, {"start": 50}, {"start": 90}]})
    _k, idx, *_ = node.run("difforum_keys", "Director keys", "stretch", direction=d[0])["result"]
    assert idx == "10,50,90"


def test_shipped_example_scripts(input_dir):
    import os

    from difforum.nodes.orchestrate import EXAMPLES_DIR
    names = sorted(f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".txt"))
    assert len(names) == 10
    p = setup(seconds=10.0)[0]
    for name in names:
        tl, _guide, info = DifforumShotScript().run("", f"examples/{name}", params=None)["result"]
        tl = json.loads(tl)
        assert tl["scenes"] or tl["camera"], name
        assert "nothing recognised" not in info, (name, info)
    with pytest.raises(ValueError):
        DifforumShotScript().run("", "examples/../nodes/render.py", params=p)


def test_resident_models_suspends_flag(monkeypatch):
    from difforum.nodes.render import resident_models
    comfy = types.ModuleType("comfy")
    mm = types.ModuleType("comfy.model_management")
    mm.DISABLE_SMART_MEMORY = True
    comfy.model_management = mm
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    with resident_models() as active:
        assert active and mm.DISABLE_SMART_MEMORY is False
    assert mm.DISABLE_SMART_MEMORY is True
    monkeypatch.setenv("DIFFORUM_RESPECT_MEMORY_FLAGS", "1")
    with resident_models() as active:
        assert not active and mm.DISABLE_SMART_MEMORY is True


def test_upscale_auto_skips_model(monkeypatch):
    import nodes as stub_nodes
    import torch

    from difforum.nodes.finish import DifforumUpscale
    calls = []

    class FakeUp:
        FUNCTION = "upscale"

        def upscale(self, upscale_model, image):
            calls.append(image.shape)
            return (image.repeat_interleave(4, 1).repeat_interleave(4, 2),)

    monkeypatch.setitem(stub_nodes.NODE_CLASS_MAPPINGS, "ImageUpscaleWithModel", FakeUp)
    big, small = torch.rand(2, 72, 128, 3), torch.rand(2, 36, 64, 3)
    out, info = DifforumUpscale().run(big, "x1.5", "lanczos", 0.0, 16, upscale_model=object())
    assert not calls and "model skipped" in info and out.shape[1:3] == (108, 192)
    out, info = DifforumUpscale().run(small, "x4", "lanczos", 0.0, 16, upscale_model=object())
    assert calls and out.shape[1:3] == (144, 256)
    calls.clear()
    DifforumUpscale().run(big, "x1.5", "lanczos", 0.0, 16, upscale_model=object(), model_use="always")
    assert calls
