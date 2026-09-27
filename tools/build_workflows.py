"""
Build the example workflows in example_workflows/ from code.

Widget order for Difforum nodes is read from the real INPUT_TYPES, so a
template can never drift out of sync with a node again (the 0.x templates
did: several shipped with camera values in the wrong fields). External nodes
are described in EXTERNAL below.

    python tools/build_workflows.py            # write
    python tools/build_workflows.py --check    # verify committed files are current
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
OUT = PACK / "example_workflows"   # ComfyUI lists these in its Templates browser

if "difforum" not in sys.modules:
    _alias = types.ModuleType("difforum")
    _alias.__path__ = [str(PACK)]
    sys.modules["difforum"] = _alias

from difforum.core.direction import default_timeline, parse_timeline  # noqa: E402
from difforum.nodes import NODE_CLASS_MAPPINGS as DF  # noqa: E402

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN"}

# type -> (link inputs [(name, type)], widget names, outputs [(name, type)])
EXTERNAL = {
    "CheckpointLoaderSimple": ([], ["ckpt_name"], [("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")]),
    "LoraLoader": ([("model", "MODEL"), ("clip", "CLIP")], ["lora_name", "strength_model", "strength_clip"],
                   [("MODEL", "MODEL"), ("CLIP", "CLIP")]),
    "CLIPTextEncode": ([("clip", "CLIP")], ["text"], [("CONDITIONING", "CONDITIONING")]),
    "EmptyLatentImage": ([], ["width", "height", "batch_size"], [("LATENT", "LATENT")]),
    "KSampler": ([("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                  ("latent_image", "LATENT")],
                 ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"],
                 [("LATENT", "LATENT")]),
    "VAEDecode": ([("samples", "LATENT"), ("vae", "VAE")], [], [("IMAGE", "IMAGE")]),
    "LoadImage": ([], ["image", "upload"], [("IMAGE", "IMAGE"), ("MASK", "MASK")]),
    "LoadAudio": ([], ["audio", "audioUI", "upload"], [("AUDIO", "AUDIO")]),
    "PreviewImage": ([("images", "IMAGE")], [], []),
    "SaveImage": ([("images", "IMAGE")], ["filename_prefix"], []),
    "PreviewAny": ([("source", "*")], [], [("STRING", "STRING")]),
    "CreateVideo": ([("images", "IMAGE"), ("audio", "AUDIO")], ["fps"], [("VIDEO", "VIDEO")]),
    "SaveVideo": ([("video", "VIDEO")], ["filename_prefix", "format", "codec"], []),
    "DownloadAndLoadDepthAnythingV2Model": ([], ["model"], [("da_v2_model", "DAMODEL")]),
    "DepthAnything_V2": ([("da_model", "DAMODEL"), ("images", "IMAGE")], [], [("image", "IMAGE")]),
    "EmptyLTXVLatentVideo": ([], ["width", "height", "length", "batch_size"], [("LATENT", "LATENT")]),
    "MarkdownNote": ([], ["text"], []),
}


def difforum_spec(type_):
    cls = DF[type_]
    it = cls.INPUT_TYPES()
    links, widgets = [], []
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            t = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if isinstance(t, (list, tuple)) or (t in WIDGET_TYPES and not opts.get("forceInput")):
                widgets.append(name)
                if t == "INT" and "seed" in name:
                    widgets.append(name + ".control")
            else:
                links.append((name, t))
    outs = list(zip(getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES), cls.RETURN_TYPES))
    defaults = {}
    for section in ("required", "optional"):
        for name, spec in it.get(section, {}).items():
            t = spec[0]
            opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            if "default" in opts:
                defaults[name] = opts["default"]
            elif isinstance(t, (list, tuple)) and t:
                defaults[name] = t[0]
    return links, widgets, outs, defaults


class Graph:
    def __init__(self, title):
        self.title = title
        self.nodes, self.links = [], []

    def add(self, type_, pos, size=(340, 200), title=None, **values):
        if type_ in DF:
            links, widgets, outs, defaults = difforum_spec(type_)
        else:
            links, widgets, outs = EXTERNAL[type_]
            defaults = {}
        wv = []
        for w in widgets:
            if w.endswith(".control") or w == "control_after_generate":
                wv.append("fixed")
            else:
                if w not in values and w not in defaults and type_ not in EXTERNAL:
                    raise KeyError(f"{type_}.{w} has no default")
                wv.append(values.pop(w, defaults.get(w)))
        if values:
            raise KeyError(f"{type_}: unknown widgets {sorted(values)}")
        node = {
            "id": len(self.nodes) + 1, "type": type_, "pos": list(pos), "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in links],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(outs)],
            "properties": {"Node name for S&R": type_},
            "widgets_values": wv,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        return node["id"]

    def note(self, pos, text, size=(420, 260)):
        return self.add("MarkdownNote", pos, size=size, text=text)

    def link(self, src, out, dst, inp):
        s, d = self.nodes[src - 1], self.nodes[dst - 1]
        oi = next(i for i, o in enumerate(s["outputs"]) if o["name"] == out)
        ii = next((i for i, x in enumerate(d["inputs"]) if x["name"] == inp), None)
        if ii is None:                                  # link into a widget (convert-to-input)
            d["inputs"].append({"name": inp, "type": s["outputs"][oi]["type"], "link": None,
                                "widget": {"name": inp}})
            ii = len(d["inputs"]) - 1
        lid = len(self.links) + 1
        self.links.append([lid, src, oi, dst, ii, s["outputs"][oi]["type"]])
        s["outputs"][oi]["links"].append(lid)
        d["inputs"][ii]["link"] = lid

    def json(self):
        return {
            "last_node_id": len(self.nodes), "last_link_id": len(self.links),
            "nodes": self.nodes, "links": self.links, "groups": [], "config": {},
            "extra": {"ds": {"scale": 0.65, "offset": [40, 40]}, "difforum": "1.0",
                      "title": self.title},
            "version": 0.4,
        }


def tl_json(frames, **patch):
    tl = default_timeline(frames)
    for k, v in patch.items():
        tl[k] = v
    return json.dumps(parse_timeline(tl))


# ---------------------------------------------------------------------------
# workflows
# ---------------------------------------------------------------------------

def sd_front(g, x=0, y=0, ckpt="sd_xl_base_1.0.safetensors", pos_text="", steps=24, cfg=6.0,
             sampler="dpmpp_2m", scheduler="karras"):
    """Checkpoint + prompts + first frame. Returns ids."""
    ck = g.add("CheckpointLoaderSimple", (x, y), size=(320, 100), ckpt_name=ckpt)
    pos = g.add("CLIPTextEncode", (x + 360, y), size=(380, 140), title="Look / style (positive)",
                text=pos_text or "cinematic film still, rich detail, volumetric light, 35mm")
    neg = g.add("CLIPTextEncode", (x + 360, y + 180), size=(380, 120), title="Negative",
                text="blurry, low quality, watermark, text, frame, border")
    g.link(ck, "CLIP", pos, "clip")
    g.link(ck, "CLIP", neg, "clip")
    return ck, pos, neg


def first_frame(g, ck, pos, neg, setup, x, y, steps=24, cfg=6.0, sampler="dpmpp_2m", scheduler="karras"):
    lat = g.add("EmptyLatentImage", (x, y), size=(260, 110), width=1024, height=576, batch_size=1)
    g.link(setup, "width", lat, "width")
    g.link(setup, "height", lat, "height")
    ks = g.add("KSampler", (x + 290, y), size=(300, 260), title="First frame", seed=7, steps=steps,
               cfg=cfg, sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    g.link(ck, "MODEL", ks, "model")
    g.link(pos, "CONDITIONING", ks, "positive")
    g.link(neg, "CONDITIONING", ks, "negative")
    g.link(lat, "LATENT", ks, "latent_image")
    dec = g.add("VAEDecode", (x + 620, y), size=(200, 60))
    g.link(ks, "LATENT", dec, "samples")
    g.link(ck, "VAE", dec, "vae")
    return dec


def wf_storyboard():
    g = Graph("01 · Storyboard - direct a shot, no model needed")
    g.note((0, -300), "## Start here - no model needed\n\n"
           "1. Load any still in **Load Image**.\n2. Edit the **Director** timeline: drag blocks, "
           "pick camera moves, press ▶ to preview.\n3. Queue: the **Storyboard** renders the whole "
           "camera move in about a second, and **Camera → Prompt** writes it in words for "
           "prompt-driven video models.\n\nWhen the motion feels right, move to 02 (render) or "
           "07/08 (LTX / H3).", size=(520, 260))
    s = g.add("Difforum_Setup", (0, 0), size=(320, 300), duration=5.0)
    d = g.add("Difforum_Director", (360, 0), size=(800, 880), timeline=tl_json(120))
    g.link(s, "params", d, "params")
    img = g.add("LoadImage", (0, 360), size=(320, 360), image="example.png")
    sb = g.add("Difforum_Storyboard", (1200, 0), size=(320, 220))
    g.link(img, "IMAGE", sb, "init_image")
    g.link(d, "direction", sb, "direction")
    p1 = g.add("PreviewImage", (1560, 0), size=(620, 380), title="Contact sheet")
    g.link(sb, "sheet", p1, "images")
    p2 = g.add("PreviewImage", (1560, 420), size=(300, 300), title="Camera path (top view)")
    g.link(sb, "camera_path", p2, "images")
    cp = g.add("Difforum_CameraPrompt", (1200, 280), size=(320, 160))
    g.link(d, "direction", cp, "direction")
    t = g.add("PreviewAny", (1200, 480), size=(320, 200), title="Camera in words")
    g.link(cp, "text", t, "source")
    return g


def wf_feedback(depth=False, title="02 · Feedback render (SDXL)", camera_mode="2d", tl=None,
                duration=5.0):
    g = Graph(title)
    ck, pos, neg = sd_front(g, 0, 0)
    s = g.add("Difforum_Setup", (0, 360), size=(320, 300), duration=duration, long_edge=1024)
    d = g.add("Difforum_Director", (420, 360), size=(800, 880), camera_mode=camera_mode,
              timeline=tl or tl_json(120))
    g.link(s, "params", d, "params")
    g.link(ck, "CLIP", d, "clip")
    dec = first_frame(g, ck, pos, neg, s, 820, 0)
    fb = g.add("Difforum_FeedbackSampler", (1260, 360), size=(360, 460), steps=20, cfg=6.0,
               sampler_name="dpmpp_2m", scheduler="karras", cadence=2)
    for a, b in (("MODEL", "model"), ("VAE", "vae")):
        g.link(ck, a, fb, b)
    g.link(pos, "CONDITIONING", fb, "positive")
    g.link(neg, "CONDITIONING", fb, "negative")
    g.link(dec, "IMAGE", fb, "init_image")
    g.link(d, "direction", fb, "direction")
    if depth:
        dm = g.add("DownloadAndLoadDepthAnythingV2Model", (1260, 0), size=(360, 80),
                   model="depth_anything_v2_vitl_fp32.safetensors")
        da = g.add("DepthAnything_V2", (1260, 120), size=(360, 80))
        g.link(dm, "da_v2_model", da, "da_model")
        g.link(dec, "IMAGE", da, "images")
        g.link(da, "image", fb, "depth")
    st = g.add("Difforum_FlowStabilize", (1660, 360), size=(300, 130), strength=0.45)
    g.link(fb, "frames", st, "frames")
    cv = g.add("CreateVideo", (2000, 360), size=(260, 100), fps=24.0)
    g.link(st, "frames", cv, "images")
    g.link(s, "fps", cv, "fps")
    sv = g.add("SaveVideo", (2000, 500), size=(300, 320), filename_prefix="video/difforum",
               format="auto", codec="auto")
    g.link(cv, "VIDEO", sv, "video")
    rp = g.add("PreviewAny", (1660, 540), size=(300, 200), title="Run report")
    g.link(fb, "report", rp, "source")
    return g, s, d, fb, cv


def wf_feedback_2d():
    g, *_ = wf_feedback()
    g.note((0, -320), "## Feedback render (SDXL)\n\nThe classic Deforum look on a modern model. "
           "**Director** drives camera, energy and prompt travel through one `direction` wire.\n\n"
           "- Any SD1.5 / SDXL / Flux checkpoint works (plain MODEL/VAE/CONDITIONING).\n"
           "- `cadence 2` diffuses every other frame and crossfades the rest (~2x faster).\n"
           "- Faster still: a DMD2 / Lightning / Turbo LoRA at 4-8 steps, cfg 1-2.\n"
           "- Director prompts replace the positive prompt once CLIP is connected; the positive "
           "box still sets the first frame.", size=(560, 260))
    return g


def wf_parallax():
    tl = tl_json(120, camera=[
        {"start": 0, "move": "dolly_in", "speed": 0.9, "intensity": 0.8, "ease": "ease_in_out"},
        {"start": 40, "move": "orbit_right", "speed": 1.0, "intensity": 0.9, "ease": "ease_in_out"},
        {"start": 80, "move": "crane_up", "speed": 0.8, "intensity": 0.8, "ease": "ease_out"},
    ])
    g, *_ = wf_feedback(depth=True, title="03 · Real parallax (3D + depth)", camera_mode="3d", tl=tl)
    g.note((0, -320), "## Real parallax\n\nDepth Anything V2 reads the first frame, the Director "
           "is in **3d**, and the sampler re-projects the depth with the image every frame - "
           "near things move faster than far ones for the whole clip, not just the start.\n\n"
           "Needs **ComfyUI-DepthAnythingV2** (Kijai). Too strong? Lower the sampler's "
           "`translation_scale`. The `depth` output is the tracked depth per frame, ready for "
           "compositing.", size=(560, 240))
    return g


def wf_audio():
    g, s, d, fb, cv = wf_feedback(title="04 · Audio reactive", duration=10.0, tl=tl_json(240, camera=[
        {"start": 0, "move": "zoom_in", "speed": 0.7, "intensity": 0.8, "react": "beat_pulse"},
        {"start": 96, "move": "spiral", "speed": 1.0, "intensity": 1.0, "react": "bass_speed"},
        {"start": 180, "move": "handheld", "speed": 1.0, "intensity": 1.2, "react": "onset_shake"},
    ]))
    au = g.add("LoadAudio", (0, 720), size=(320, 140), audio="track.mp3")
    an = g.add("Difforum_AudioAnalyzer", (0, 900), size=(320, 200))
    g.link(s, "params", an, "params")
    g.link(au, "AUDIO", an, "audio")
    g.link(an, "audio_curves", d, "audio")
    g.link(au, "AUDIO", cv, "audio")
    pv = g.add("PreviewImage", (0, 1140), size=(320, 260), title="Bands: low / mid / high / beat")
    g.link(an, "bands_plot", pv, "images")
    g.note((0, -320), "## Audio reactive, no expressions\n\nEach camera block in the Director has "
           "an **Audio** reaction: pulse on beat, shake on hits, bass drives speed, mids rock the "
           "roll. The analyzer feeds them; **Create Video** muxes the track back in.\n\nFor custom "
           "curves use **Audio Curve** or type bands (`low`, `beat`...) in a Schedule.", size=(560, 220))
    return g


def wf_live():
    g = Graph("05 · Live (turbo models)")
    ck, pos, neg = sd_front(g, 0, 0, ckpt="sd_xl_turbo_1.0_fp16.safetensors",
                            pos_text="liquid chrome sculpture, iridescent, studio light")
    s = g.add("Difforum_Setup", (0, 360), size=(320, 300), duration=10.0, long_edge=512)
    d = g.add("Difforum_Director", (420, 360), size=(800, 880), timeline=tl_json(240))
    g.link(s, "params", d, "params")
    dec = first_frame(g, ck, pos, neg, s, 820, 0, steps=2, cfg=1.0, sampler="euler_ancestral",
                      scheduler="sgm_uniform")
    lv = g.add("Difforum_LiveSampler", (1260, 360), size=(360, 560), steps=2, cfg=1.0,
               sampler_name="euler_ancestral", scheduler="sgm_uniform", run_frames=1200)
    op = g.add("Difforum_RenderOptions", (1260, 960), size=(340, 560), title="Render Options (kaleidoscope)",
               symmetry="kaleidoscope", color_coherence=0.5)
    g.link(op, "options", lv, "options")
    for a, b in (("MODEL", "model"), ("VAE", "vae")):
        g.link(ck, a, lv, b)
    g.link(pos, "CONDITIONING", lv, "positive")
    g.link(neg, "CONDITIONING", lv, "negative")
    g.link(dec, "IMAGE", lv, "init_image")
    g.link(d, "direction", lv, "direction")
    pv = g.add("PreviewImage", (1660, 360), size=(400, 300))
    g.link(lv, "frames", pv, "images")
    g.note((0, -320), "## Live\n\nQueue once: the node plays in place. `live_source` = `0` for a "
           "webcam magic mirror; `stream_dir` / `spout_name` feed OBS, Resolume or TouchDesigner. "
           "Use a 1-4 step model at ~512 px.", size=(560, 160))
    return g


def wf_loop():
    g = Graph("06 · Seamless loop (installations)")
    ck, pos, neg = sd_front(g, 0, 0)
    s = g.add("Difforum_Setup", (0, 360), size=(320, 300), duration_mode="frames", duration=360.0)
    cam = g.add("Difforum_Camera", (420, 360), size=(380, 360), keys="0: orbit_right 1.0 0.8\n60: spiral 1.2 1.0",
                loop_mode="harmonic", cycle_frames=120)
    g.link(s, "params", cam, "params")
    sch = g.add("Difforum_Schedule", (420, 760), size=(380, 160), schedule="0:(0.5)")
    g.link(s, "params", sch, "params")
    dec = first_frame(g, ck, pos, neg, s, 820, 0)
    fb = g.add("Difforum_FeedbackSampler", (860, 360), size=(360, 460))
    for a, b in (("MODEL", "model"), ("VAE", "vae")):
        g.link(ck, a, fb, b)
    g.link(pos, "CONDITIONING", fb, "positive")
    g.link(neg, "CONDITIONING", fb, "negative")
    g.link(dec, "IMAGE", fb, "init_image")
    g.link(s, "params", fb, "params")
    g.link(cam, "camera", fb, "camera")
    g.link(sch, "schedule", fb, "strength")
    lp = g.add("Difforum_Loop", (1260, 360), size=(300, 160), method="keep settled lap")
    g.link(fb, "frames", lp, "frames")
    g.link(cam, "cycle_frames", lp, "cycle_frames")
    cv = g.add("CreateVideo", (1600, 360), size=(260, 100), fps=24.0)
    g.link(lp, "frames", cv, "images")
    sv = g.add("SaveVideo", (1600, 500), size=(300, 320), filename_prefix="video/difforum_loop",
               format="auto", codec="auto")
    g.link(cv, "VIDEO", sv, "video")
    g.note((0, -320), "## Loop without a crossfade\n\nThe Camera path is made periodic over 120 frames "
           "and rendered for 3 laps; the feedback settles onto its cycle and **Loop** keeps the last "
           "lap. For projections that run for hours.", size=(560, 180))
    return g


def wf_ltx():
    g = Graph("07 · LTX-2 / 2.5 guides from the Director")
    s = g.add("Difforum_Setup", (0, 0), size=(320, 300), target="LTX-2 / 2.5", duration=5.0,
              long_edge=1216)
    d = g.add("Difforum_Director", (360, 0), size=(800, 880), timeline=tl_json(121))
    g.link(s, "params", d, "params")
    img = g.add("LoadImage", (0, 360), size=(320, 360), image="example.png", title="Anchor / first frame")
    gf = g.add("Difforum_GuideFrames", (1200, 0), size=(340, 200))
    g.link(img, "IMAGE", gf, "anchor_image")
    g.link(d, "direction", gf, "direction")
    kf = g.add("Difforum_Keyframes", (1200, 240), size=(340, 220), every_seconds=1.0)
    g.link(gf, "guide_frames", kf, "frames")
    g.link(s, "fps", kf, "fps")
    lat = g.add("EmptyLTXVLatentVideo", (1200, 500), size=(300, 160), width=1216, height=704,
                length=121, batch_size=1)
    g.link(s, "width", lat, "width")
    g.link(s, "height", lat, "height")
    g.link(kf, "length", lat, "length")
    lg = g.add("Difforum_LTXGuides", (1580, 240), size=(340, 260))
    g.link(kf, "keyframes", lg, "keyframes")
    g.link(kf, "indices", lg, "indices")
    g.link(lat, "LATENT", lg, "latent")
    cp = g.add("Difforum_CameraPrompt", (1580, 0), size=(340, 160), format="prompt suffix")
    g.link(d, "direction", cp, "direction")
    pv = g.add("PreviewImage", (1580, 540), size=(340, 260), title="Keyframes")
    g.link(kf, "keyframes", pv, "images")
    g.note((1960, 0), "## Wire into your LTX graph\n\nConnect your LTX **positive / negative** "
           "(text encoded with the Camera → Prompt text appended), and the LTX **VAE** into "
           "**LTX Guides**. Its outputs replace the conditioning and latent going into your LTX "
           "sampler.\n\n- Setup target = LTX keeps length on 8k+1 and size on 32 px.\n"
           "- Guides every 1 s at strength 0.7; frame 0 at 1.0 locks the look.\n"
           "- Swap Guide Frames for a Feedback render or a Storyboard to guide with other "
           "images.", size=(520, 300))
    return g


def wf_h3():
    g = Graph("08 · MiniMax H3 first/last frame from the Director")
    s = g.add("Difforum_Setup", (0, 0), size=(320, 300), target="MiniMax H3", duration=10.0,
              long_edge=1344)
    d = g.add("Difforum_Director", (360, 0), size=(800, 880), timeline=tl_json(243))
    g.link(s, "params", d, "params")
    img = g.add("LoadImage", (0, 360), size=(320, 360), image="example.png", title="First frame")
    gf = g.add("Difforum_GuideFrames", (1200, 0), size=(340, 200))
    g.link(img, "IMAGE", gf, "anchor_image")
    g.link(d, "direction", gf, "direction")
    h3 = g.add("Difforum_H3Shot", (1200, 240), size=(340, 300), segment_length="243 (~10s)",
               shot_description="A misty ancient forest at dawn; light shafts cut through the canopy.")
    g.link(gf, "guide_frames", h3, "frames")
    g.link(d, "direction", h3, "direction")
    p1 = g.add("PreviewImage", (1580, 0), size=(300, 220), title="first_frame")
    g.link(h3, "first_frame", p1, "images")
    p2 = g.add("PreviewImage", (1580, 260), size=(300, 220), title="last_frame")
    g.link(h3, "last_frame", p2, "images")
    t = g.add("PreviewAny", (1580, 520), size=(300, 200), title="prompt")
    g.link(h3, "prompt", t, "source")
    g.note((1920, 0), "## Wire into MiniMax H3\n\n**MiniMax H3 Image to Video** (core): "
           "`first_frame`, `last_frame`, `width`, `height`, `length` from **H3 Shot**; put its "
           "`prompt` in your H3 prompt (e.g. MiniMax H3 Prompt Format, mode fl2va).\n\n"
           "The last frame is the anchor carried to where the camera ends, so H3 performs "
           "the move you drew. Revealed edges are gray - for pans, render the path with the "
           "Feedback Sampler first and feed those frames instead.\n\nLonger than 20 s: raise "
           "`segment`, and use each segment's last frame as the next first frame.", size=(520, 340))
    return g


def wf_export():
    g = Graph("09 · Camera to After Effects / Blender (and back)")
    s = g.add("Difforum_Setup", (0, 0), size=(320, 300))
    d = g.add("Difforum_Director", (360, 0), size=(800, 880), camera_mode="3d", timeline=tl_json(120))
    g.link(s, "params", d, "params")
    ex = g.add("Difforum_CameraExport", (1200, 0), size=(360, 200))
    g.link(d, "direction", ex, "direction")
    t = g.add("PreviewAny", (1600, 0), size=(360, 160), title="Written files")
    g.link(ex, "paths", t, "source")
    im = g.add("Difforum_CameraImport", (1200, 260), size=(360, 200))
    g.link(s, "params", im, "params")
    img = g.add("LoadImage", (0, 360), size=(320, 360), image="example.png")
    sb = g.add("Difforum_Storyboard", (1600, 260), size=(320, 220))
    g.link(img, "IMAGE", sb, "init_image")
    g.link(s, "params", sb, "params")
    g.link(im, "camera", sb, "camera")
    pv = g.add("PreviewImage", (1960, 260), size=(520, 320))
    g.link(sb, "sheet", pv, "images")
    g.note((0, -320), "## Camera interchange\n\n**Camera Export** writes to `output/difforum/`: a "
           "`.jsx` for After Effects (File > Scripts > Run Script File), a `.py` for Blender "
           "(Text Editor > Run) and a `.json`. Composite titles or 3D on top of the render with "
           "the same camera.\n\n**Camera Import** reads a `.json` from `input/` - from Camera "
           "Export, or from `tools/blender/difforum_export_camera.py` / `tools/aftereffects/"
           "Difforum Export Camera.jsx` - so a move blocked in Blender or AE drives the AI shot. "
           "(The import branch errors until a file is in `input/`: bypass it with Ctrl+B.)",
           size=(620, 260))
    return g


WORKFLOWS = {
    "01_storyboard_no_model.json": wf_storyboard,
    "02_feedback_sdxl.json": wf_feedback_2d,
    "03_parallax_3d_depth.json": wf_parallax,
    "04_audio_reactive.json": wf_audio,
    "05_live_turbo.json": wf_live,
    "06_seamless_loop.json": wf_loop,
    "07_ltx_guides.json": wf_ltx,
    "08_h3_first_last.json": wf_h3,
    "09_camera_to_ae_blender.json": wf_export,
}


def main(check=False):
    OUT.mkdir(parents=True, exist_ok=True)
    stale = []
    for name, fn in WORKFLOWS.items():
        text = json.dumps(fn().json(), indent=1, ensure_ascii=False) + "\n"
        path = OUT / name
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.write_text(text, encoding="utf-8")
            print("wrote", path.relative_to(PACK))
    if check and stale:
        print("stale workflows:", ", ".join(stale))
        sys.exit(1)


if __name__ == "__main__":
    main(check="--check" in sys.argv)
