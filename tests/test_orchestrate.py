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
    assert cam[108]["move"] == "orbit_left" and cam[108]["speed"] == 1.5 and cam[108]["lens"] == 54.4
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
    assert "timeline_in" in out["result"][5]
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


def test_scene_stills(stub_model):
    from conftest import StubVAE
    from test_nodes import StubClip

    from difforum.nodes.orchestrate import DifforumSceneStills
    p = setup(seconds=5.0)[0]
    tl = script_to_timeline("0s | calm | zoom_in | a boat\n2s | build | the storm\n4s | resolve | sunrise", 24, 120)[0]
    d = director(p, tl=tl)[0]
    out = DifforumSceneStills().run(d, stub_model, StubClip(), StubVAE(), "gouache illustration", "blurry",
                                    4, 1.0, "euler", "normal", 7, 0.5, 512)["result"]
    keys, idx, first, last, info = out
    assert idx == "0,48,96" and keys.shape[0] == 3 and keys.shape[-1] == 3
    assert [round(c["denoise"], 3) for c in stub_model.calls] == [1.0, 0.775, 0.775]
    assert first.shape[0] == 1 and "over the previous" in info
    stub_model.calls.clear()
    import torch
    DifforumSceneStills().run(d, stub_model, StubClip(), StubVAE(), "", "blurry", 4, 1.0, "euler", "normal",
                              7, 0.0, 512, first_image=torch.rand(1, 90, 160, 3))
    assert [c["denoise"] for c in stub_model.calls] == [1.0, 1.0]


def test_director_pins_images_and_guides_use_them(monkeypatch):
    import nodes as stub_nodes
    import torch
    from test_nodes import StubClip

    from difforum.nodes.bridges import DifforumH3Guides
    from difforum.nodes.orchestrate import DifforumTravelConditioning
    p = setup(seconds=5.0)[0]
    tl = json.dumps(script_to_timeline(
        "0-47 | calm | zoom_in | a boat | sound: rain\n48-95 | build | the storm | sound: thunder\n"
        "96-119 | resolve | sunrise", 24, 120)[0])
    out = DifforumDirector().run(p, tl, "2d", "cinematic", 1.0, 1.0, 0.0, 0.0, 0, clip=StubClip(),
                                 images=torch.rand(2, 50, 80, 3), image_2=torch.rand(1, 300, 200, 3))
    bundle, keys, idx = out["result"][0], out["result"][6], out["result"][7]
    assert idx == "0,48,96" and keys.shape == (3, p["height"], p["width"], 3)
    assert bundle.key_indices == idx and len(out["ui"]["difforum_thumbs"]) == 3
    assert out["ui"]["difforum_thumbs"][1]["frame"] == 48
    assert bundle.direction.scenes[0]["sound"] == "rain"

    calls = []

    class FakeAddGuide:
        @classmethod
        def execute(cls, positive, latent, frame_idx, vae=None, audio_vae=None, image=None, audio=None):
            calls.append(frame_idx)
            return (positive,)

    monkeypatch.setitem(stub_nodes.NODE_CLASS_MAPPINGS, "MiniMaxH3AddGuide", FakeAddGuide)
    DifforumH3Guides().run([], {"samples": None}, object(), direction=bundle)
    assert calls == [0, 48, 96]
    with pytest.raises(ValueError):
        DifforumH3Guides().run([], {"samples": None}, object())

    cond, n, _info = DifforumTravelConditioning().run(direction=bundle)
    assert n == 120 and cond[0][0].shape[0] == 120 and cond[0][1]["pooled_output"].shape[0] == 120
    cond, n, _ = DifforumTravelConditioning().run(direction=bundle, images=torch.zeros(30, 8, 8, 3))
    assert n == 30 and cond[0][0].shape[0] == 30
    assert float(cond[0][0][0].mean()) != float(cond[0][0][-1].mean())      # the prompt travels


def test_script_ranges_sound_and_roundtrip():
    from difforum.core.h3prompt import scene_sounds
    from difforum.core.script import timeline_to_script
    from difforum.nodes.routes import convert_script
    txt = ("do frame 0 ao frame 35 | calm | zoom_in slow | a boat | sound: rain on glass\n"
           "35-238 | build | pan_right x0.7 amp 0.6 | a stream | sound: trickling water\n"
           "120 | key: the drop\n240 to 299 | resolve | crane_up | a pond")
    tl, notes = script_to_timeline(txt, 24, 300)
    assert not notes and [s["start"] for s in tl["scenes"]] == [0, 35, 240]
    text = timeline_to_script(tl, 300)
    assert "0-34 |" in text and "35-239 |" in text and "sound: rain on glass" in text
    assert script_to_timeline(text, 24, 300)[0] == tl
    assert convert_script({"timeline": tl, "frames": 300})["text"] == text
    assert convert_script({"text": text, "fps": 24, "frames": 300})["timeline"] == tl
    p = setup(seconds=12.5)[0]
    d = director(p, tl=tl)[0]
    assert scene_sounds(d.direction) == "Rain on glass, then trickling water."


def _cams(text, frames=120):
    tl, notes = script_to_timeline(text, 24, frames)
    return tl, [(c["start"], c["move"]) for c in tl["camera"]], notes


def test_script_accepts_what_language_models_write():
    table = ("| Frames | Mood | Camera | Prompt |\n|---|---|---|---|\n"
             "| 0-35 | calm | zoom in slow | a quiet lake |\n| 36-90 | build | pan right | wind rises |")
    tl, cams, notes = _cams(table)
    assert cams == [(0, "zoom_in"), (36, "pan_right")] and not notes
    assert [s["prompt"] for s in tl["scenes"]] == ["a quiet lake", "wind rises"]

    tl, cams, notes = _cams("1. **0-35** | calm | zoom_in | a lake\n2. **36-90** | tense | orbit left, fast | storm")
    assert cams == [(0, "zoom_in"), (36, "orbit_left")] and tl["camera"][1]["speed"] == 1.5 and not notes

    tl, cams, _ = _cams("0-35 | mood: calm | camera: pan left | prompt: a lake | sound: wind")
    assert cams == [(0, "pan_left")] and tl["scenes"][0] == {"start": 0, "mood": "calm", "prompt": "a lake",
                                                             "sound": "wind"}

    tl, cams, notes = _cams("Here you go:\n```\n0-35 | calm | the camera pushes in slowly | a lake\n```\nEnjoy!")
    assert cams == [(0, "dolly_in")] and tl["scenes"][0]["prompt"] == "a lake" and not notes

    tl, cams, notes = _cams("do frame 0 ao frame 35 | calmo | zoom para dentro devagar | um lago\n"
                            "36-90 | tenso | girar para a direita rápido | tempestade")
    assert cams == [(0, "zoom_in"), (36, "roll_cw")] and not notes
    assert [s["mood"] for s in tl["scenes"]] == ["calm", "tense"]


def test_script_warns_instead_of_guessing():
    tl, cams, notes = _cams("0-35 | calm | somersault | a lake")
    assert not cams and tl["scenes"][0]["prompt"] == "a lake"
    assert "somersault" in notes[0] and "not a known move" in notes[0]
    # a prompt is never mistaken for a camera, and long text is never dropped
    tl, cams, notes = _cams("0 | rise of the machines\n40 | calm | a lake at dawn in the mist | birds")
    assert not cams and not notes
    assert [s["prompt"] for s in tl["scenes"]] == ["rise of the machines", "a lake at dawn in the mist, birds"]


def test_script_round_trip_keeps_everything():
    from difforum.core.script import timeline_to_script
    text = ("0-35 | calm | still | zoom lens on a table, red | blue cloth | sound: a clock\n"
            "36-119 | build | free dx -0.2 dy 0.1 zoom 1.5 roll 12 ease_out | the room tilts\n"
            "20 | guidance 5.5\n10 | energy 0.4\n60 | key: the drop")
    tl, notes = script_to_timeline(text, 24, 120)
    assert not notes
    assert tl["camera"][1] == {"start": 36, "move": "free", "speed": 1.0, "intensity": 1.0, "lens": 0.0,
                               "ease": "ease_out", "react": "none", "dx": -0.2, "dy": 0.1, "zoom": 1.5,
                               "roll": 12.0}
    again, notes = script_to_timeline(timeline_to_script(tl, 120), 24, 120)
    assert not notes
    for track in ("scenes", "camera", "keys", "energy", "guidance"):
        assert again[track] == tl[track], track


def test_free_pose_ends_on_its_framing():
    import torch

    from difforum.core.direction import build_direction, pose_matrix
    from difforum.core.engine import EngineConfig, FeedbackEngine
    from difforum.core.h3prompt import camera_clause
    from difforum.nodes.direction import _track
    w, h, n = 640, 360, 60
    tl = {"version": 2, "scenes": [{"start": 0, "mood": "calm", "prompt": "x"}],
          "camera": [{"start": 0, "move": "still"},
                     {"start": 10, "move": "free", "dx": -0.25, "dy": 0.1, "zoom": 1.6, "roll": 20, "ease": "ease_out"},
                     {"start": 40, "move": "still"}]}
    d = build_direction(json.dumps(tl), n, 24.0, width=w, height=h)
    cam = _track(d.axes, d.lens, n, "2d", [b["move"] for b in d.camera_blocks])
    eng = FeedbackEngine(cam, EngineConfig(width=w, height=h))
    acc, at = torch.eye(3, dtype=torch.float64), {}
    for f in range(1, n):
        acc = eng._affine_for(f) @ acc
        at[f] = acc.clone()
    want = torch.tensor(pose_matrix(-0.25, 0.1, 1.6, 20, 1.0, w, h), dtype=torch.float64)
    assert torch.allclose(at[39], want, atol=1e-2)           # the block ends exactly on the stored framing
    assert torch.allclose(at[59], want, atol=1e-2)           # and the still after it holds it
    assert torch.allclose(at[9], torch.eye(3, dtype=torch.float64), atol=1e-6)
    phrase = camera_clause(d.camera_blocks[1])
    assert "pushes in" in phrase and "trucks right" in phrase
