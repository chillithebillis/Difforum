"""
Generate Difforum example workflows as ComfyUI litegraph JSON.

Run with any python:  python examples/_build_examples.py
Produces (next to this script):
    difforum_schedule_basic.json     - self-contained, no assets, runs anywhere
    difforum_audio_reactive.json     - needs an audio file via Load Audio

Also validates that every link references an existing node/slot before writing.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


class WF:
    """Tiny builder for litegraph workflow JSON."""

    def __init__(self):
        self.nodes = []
        self.links = []
        self._nid = 0
        self._lid = 0

    def node(self, type_, pos, inputs, outputs, widgets=None, is_output=False):
        self._nid += 1
        node = {
            "id": self._nid,
            "type": type_,
            "pos": pos,
            "size": [320, 200],
            "flags": {},
            "order": self._nid - 1,
            "mode": 0,
            "inputs": [
                {"name": n, "type": t, "link": None} for (n, t) in inputs
            ],
            "outputs": [
                {"name": n, "type": t, "links": [], "slot_index": i}
                for i, (n, t) in enumerate(outputs)
            ],
            "properties": {"Node name for S&R": type_},
            "widgets_values": widgets or [],
        }
        self.nodes.append(node)
        return self._nid

    def link(self, src_id, src_slot, dst_id, dst_slot, type_):
        self._lid += 1
        self.links.append([self._lid, src_id, src_slot, dst_id, dst_slot, type_])
        src = next(n for n in self.nodes if n["id"] == src_id)
        dst = next(n for n in self.nodes if n["id"] == dst_id)
        src["outputs"][src_slot]["links"].append(self._lid)
        dst["inputs"][dst_slot]["link"] = self._lid
        return self._lid

    def validate(self):
        ids = {n["id"] for n in self.nodes}
        for lid, s, ss, d, ds, _ in self.links:
            assert s in ids and d in ids, f"link {lid} dangling node"
            src = next(n for n in self.nodes if n["id"] == s)
            dst = next(n for n in self.nodes if n["id"] == d)
            assert ss < len(src["outputs"]), f"link {lid} bad src slot"
            assert ds < len(dst["inputs"]), f"link {lid} bad dst slot"
        return True

    def dump(self):
        self.validate()
        return {
            "last_node_id": self._nid,
            "last_link_id": self._lid,
            "nodes": self.nodes,
            "links": self.links,
            "groups": [],
            "config": {},
            "extra": {},
            "version": 0.4,
        }


def build_basic():
    w = WF()
    setup = w.node(
        "DifforumAnimSetup", [80, 200],
        inputs=[],
        outputs=[("params", "DIFFORUM_PARAMS")],
        widgets=[768, 768, 24, 120, 0, "fixed"],
    )
    sched = w.node(
        "DifforumSchedule", [460, 200],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
        widgets=["0:(0), 60:(0.5*sin(2*pi*t/30)), 120:(1.0)", "linear"],
    )
    info = w.node(
        "DifforumScheduleInfo", [860, 120],
        inputs=[("schedule", "DIFFORUM_SCHEDULE")],
        outputs=[("info", "STRING")],
        widgets=[],
        is_output=True,
    )
    plot = w.node(
        "DifforumSchedulePlot", [860, 320],
        inputs=[("schedule", "DIFFORUM_SCHEDULE")],
        outputs=[("plot", "IMAGE")],
        widgets=[512, 256],
    )
    preview = w.node(
        "PreviewImage", [1200, 320], inputs=[("images", "IMAGE")],
        outputs=[], widgets=[], is_output=True,
    )
    w.link(setup, 0, sched, 0, "DIFFORUM_PARAMS")
    w.link(sched, 0, info, 0, "DIFFORUM_SCHEDULE")
    w.link(sched, 0, plot, 0, "DIFFORUM_SCHEDULE")
    w.link(plot, 0, preview, 0, "IMAGE")
    return w.dump()


def build_audio():
    w = WF()
    setup = w.node(
        "DifforumAnimSetup", [80, 120],
        inputs=[],
        outputs=[("params", "DIFFORUM_PARAMS")],
        widgets=[768, 768, 24, 240, 0, "fixed"],
    )
    load = w.node(
        "LoadAudio", [80, 420],
        inputs=[],
        outputs=[("AUDIO", "AUDIO")],
        widgets=["your_track.mp3"],
    )
    analyzer = w.node(
        "DifforumAudioAnalyzer", [460, 360],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "AUDIO")],
        outputs=[
            ("audio_curves", "DIFFORUM_AUDIO"), ("amp", "FLOAT"), ("low", "FLOAT"),
            ("mid", "FLOAT"), ("high", "FLOAT"), ("onset", "FLOAT"), ("beat", "FLOAT"),
        ],
        widgets=[0.25, True, 1.5],
    )
    zoom = w.node(
        "DifforumSchedule", [880, 120],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
        widgets=["0:(1.0 + 0.6*low)", "linear"],  # zoom pumps on the bass
    )
    spin = w.node(
        "DifforumSchedule", [880, 380],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
        widgets=["0:(beat*30)", "linear"],  # kick a spin on every beat
    )
    info_zoom = w.node(
        "DifforumScheduleInfo", [1280, 120],
        inputs=[("schedule", "DIFFORUM_SCHEDULE")],
        outputs=[("info", "STRING")], widgets=[], is_output=True,
    )
    info_spin = w.node(
        "DifforumScheduleInfo", [1280, 380],
        inputs=[("schedule", "DIFFORUM_SCHEDULE")],
        outputs=[("info", "STRING")], widgets=[], is_output=True,
    )
    w.link(setup, 0, analyzer, 0, "DIFFORUM_PARAMS")
    w.link(load, 0, analyzer, 1, "AUDIO")
    w.link(setup, 0, zoom, 0, "DIFFORUM_PARAMS")
    w.link(analyzer, 0, zoom, 1, "DIFFORUM_AUDIO")
    w.link(setup, 0, spin, 0, "DIFFORUM_PARAMS")
    w.link(analyzer, 0, spin, 1, "DIFFORUM_AUDIO")
    w.link(zoom, 0, info_zoom, 0, "DIFFORUM_SCHEDULE")
    w.link(spin, 0, info_spin, 0, "DIFFORUM_SCHEDULE")
    return w.dump()


def build_feedback():
    """Full Classic+ pipeline: txt2img frame 0 -> Deforum feedback -> save."""
    w = WF()
    ckpt = w.node(
        "CheckpointLoaderSimple", [40, 40], inputs=[],
        outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
        widgets=["dreamshaper_8.safetensors"],
    )
    pos = w.node(
        "CLIPTextEncode", [340, 20], inputs=[("clip", "CLIP")],
        outputs=[("CONDITIONING", "CONDITIONING")],
        widgets=["psychedelic cosmic tunnel, vivid colors, highly detailed"],
    )
    neg = w.node(
        "CLIPTextEncode", [340, 200], inputs=[("clip", "CLIP")],
        outputs=[("CONDITIONING", "CONDITIONING")],
        widgets=["blurry, low quality, watermark, text"],
    )
    lat = w.node(
        "EmptyLatentImage", [340, 380], inputs=[],
        outputs=[("LATENT", "LATENT")], widgets=[768, 768, 1],
    )
    ks = w.node(
        "KSampler", [660, 120],
        inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
        outputs=[("LATENT", "LATENT")],
        widgets=[0, "fixed", 20, 7.0, "euler", "normal", 1.0],
    )
    dec = w.node(
        "VAEDecode", [980, 120],
        inputs=[("samples", "LATENT"), ("vae", "VAE")],
        outputs=[("IMAGE", "IMAGE")], widgets=[],
    )
    setup = w.node(
        "DifforumAnimSetup", [40, 520], inputs=[],
        outputs=[("params", "DIFFORUM_PARAMS")],
        widgets=[768, 768, 24, 48, 0, "fixed"],
    )
    cam = w.node(
        "DifforumCamera", [340, 560],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
        widgets=["3d", 40.0, "0:(0)", "0:(0)", "0:(1.5)", "0:(0)",
                 "0:(0.5*sin(2*pi*t/120))", "0:(0)", "0:(1.0)"],
    )
    strength = w.node(
        "DifforumSchedule", [340, 820],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
        widgets=["0:(0.45)", "linear"],
    )
    prompt = w.node(
        "DifforumPromptSchedule", [340, 1060],
        inputs=[("params", "DIFFORUM_PARAMS"), ("clip", "CLIP")],
        outputs=[("positive_schedule", "DIFFORUM_PROMPT"),
                 ("first_frame_cond", "CONDITIONING"), ("info", "STRING")],
        widgets=["0: psychedelic cosmic tunnel, vivid colors\n"
                 "24: deep space nebula, glowing stars\n"
                 "47: intricate fractal mandala, vivid", "ease_in_out"],
    )
    fb = w.node(
        "DifforumFeedbackSampler", [1000, 480],
        inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"), ("vae", "VAE"),
                ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                ("init_image", "IMAGE"), ("strength_schedule", "DIFFORUM_SCHEDULE"),
                ("depth", "IMAGE"), ("cfg_schedule", "DIFFORUM_SCHEDULE"),
                ("positive_schedule", "DIFFORUM_PROMPT")],
        outputs=[("frames", "IMAGE")],
        widgets=[20, 7.0, "euler", "normal", 0.8, "lab", 1.0, 100.0, False, 1.0],
    )
    save = w.node(
        "SaveImage", [1360, 480], inputs=[("images", "IMAGE")],
        outputs=[], widgets=["Difforum"], is_output=True,
    )
    # frame 0 txt2img
    w.link(ckpt, 0, ks, 0, "MODEL")
    w.link(ckpt, 1, pos, 0, "CLIP")
    w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(pos, 0, ks, 1, "CONDITIONING")
    w.link(neg, 0, ks, 2, "CONDITIONING")
    w.link(lat, 0, ks, 3, "LATENT")
    w.link(ks, 0, dec, 0, "LATENT")
    w.link(ckpt, 2, dec, 1, "VAE")
    # difforum control plane
    w.link(setup, 0, cam, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, strength, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, prompt, 0, "DIFFORUM_PARAMS")
    w.link(ckpt, 1, prompt, 1, "CLIP")
    # feedback sampler
    w.link(ckpt, 0, fb, 0, "MODEL")
    w.link(pos, 0, fb, 1, "CONDITIONING")
    w.link(neg, 0, fb, 2, "CONDITIONING")
    w.link(ckpt, 2, fb, 3, "VAE")
    w.link(setup, 0, fb, 4, "DIFFORUM_PARAMS")
    w.link(cam, 0, fb, 5, "DIFFORUM_CAMERA")
    w.link(dec, 0, fb, 6, "IMAGE")
    w.link(strength, 0, fb, 7, "DIFFORUM_SCHEDULE")
    w.link(prompt, 0, fb, 10, "DIFFORUM_PROMPT")
    w.link(fb, 0, save, 0, "IMAGE")
    return w.dump()


def build_camera_warp():
    """Show the warp: Load an image, warp one frame by the camera, preview it."""
    w = WF()
    load = w.node(
        "LoadImage", [40, 60], inputs=[],
        outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["example.png"],
    )
    setup = w.node(
        "DifforumAnimSetup", [40, 360], inputs=[],
        outputs=[("params", "DIFFORUM_PARAMS")], widgets=[768, 768, 24, 48, 0, "fixed"],
    )
    cam = w.node(
        "DifforumCameraMove", [360, 360],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
        widgets=["spiral", 1.0, 1.0, "2d", 40.0],
    )
    warp = w.node(
        "DifforumWarp", [760, 120],
        inputs=[("image", "IMAGE"), ("camera", "DIFFORUM_CAMERA"), ("depth", "IMAGE")],
        outputs=[("warped", "IMAGE"), ("occlusion_mask", "MASK")],
        widgets=[12, "follow_camera", 1.0, 100.0, False],
    )
    prev = w.node(
        "PreviewImage", [1120, 120], inputs=[("images", "IMAGE")],
        outputs=[], widgets=[], is_output=True,
    )
    w.link(setup, 0, cam, 0, "DIFFORUM_PARAMS")
    w.link(load, 0, warp, 0, "IMAGE")
    w.link(cam, 0, warp, 1, "DIFFORUM_CAMERA")
    w.link(warp, 0, prev, 0, "IMAGE")
    return w.dump()


def build_hybrid_guides():
    """Hybrid bridge: warp an anchor along the camera path into a Wan guide batch."""
    w = WF()
    load = w.node(
        "LoadImage", [40, 60], inputs=[],
        outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["anchor.png"],
    )
    setup = w.node(
        "DifforumAnimSetup", [40, 360], inputs=[],
        outputs=[("params", "DIFFORUM_PARAMS")], widgets=[832, 480, 16, 81, 0, "fixed"],
    )
    cam = w.node(
        "DifforumCamera", [360, 360],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
        widgets=["3d", 40.0, "0:(0)", "0:(0)", "0:(1.2)", "0:(0)",
                 "0:(0.3*sin(2*pi*t/160))", "0:(0)", "0:(1.0)"],
    )
    guides = w.node(
        "DifforumGuideBuilder", [760, 120],
        inputs=[("anchor_image", "IMAGE"), ("camera", "DIFFORUM_CAMERA"),
                ("params", "DIFFORUM_PARAMS"), ("depth", "IMAGE")],
        outputs=[("guide_frames", "IMAGE"), ("occlusion_masks", "MASK"), ("info", "STRING")],
        widgets=["force_2d", 1.0, 100.0, False],
    )
    prev = w.node(
        "PreviewImage", [1140, 120], inputs=[("images", "IMAGE")],
        outputs=[], widgets=[], is_output=True,
    )
    w.link(setup, 0, cam, 0, "DIFFORUM_PARAMS")
    w.link(load, 0, guides, 0, "IMAGE")
    w.link(cam, 0, guides, 1, "DIFFORUM_CAMERA")
    w.link(setup, 0, guides, 2, "DIFFORUM_PARAMS")
    w.link(guides, 0, prev, 0, "IMAGE")
    return w.dump()


def build_models_info():
    """Reference card: GPU profile + model recipe + a sampled schedule value."""
    w = WF()
    profile = w.node(
        "DifforumModelProfile", [40, 60], inputs=[],
        outputs=[("profile", "DIFFORUM_PROFILE"), ("summary", "STRING"),
                 ("width", "INT"), ("height", "INT"), ("segment_frames", "INT")],
        widgets=["wan22", "balanced", True, 12, "sdpa"],
        is_output=True,
    )
    catalog = w.node(
        "DifforumModelCatalog", [40, 360], inputs=[],
        outputs=[("recipe", "DIFFORUM_RECIPE"), ("guide", "STRING")],
        widgets=["wan22_12gb"], is_output=True,
    )
    setup = w.node(
        "DifforumAnimSetup", [440, 360], inputs=[],
        outputs=[("params", "DIFFORUM_PARAMS")], widgets=[768, 768, 24, 60, 0, "fixed"],
    )
    sched = w.node(
        "DifforumSchedule", [440, 560],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
        widgets=["0:(0.2), 30:(0.8), 59:(0.2)", "ease_in_out"],
    )
    sample = w.node(
        "DifforumSampleSchedule", [840, 560],
        inputs=[("schedule", "DIFFORUM_SCHEDULE")],
        outputs=[("value", "FLOAT")], widgets=[30],
    )
    info = w.node(
        "DifforumScheduleInfo", [840, 360],
        inputs=[("schedule", "DIFFORUM_SCHEDULE")],
        outputs=[("info", "STRING")], widgets=[], is_output=True,
    )
    w.link(setup, 0, sched, 0, "DIFFORUM_PARAMS")
    w.link(sched, 0, sample, 0, "DIFFORUM_SCHEDULE")
    w.link(sched, 0, info, 0, "DIFFORUM_SCHEDULE")
    return w.dump()


def build_deluxe():
    """Everything: feedback + prompt travel + ControlNet + RIFE interp -> MP4."""
    w = WF()
    ckpt = w.node("CheckpointLoaderSimple", [40, 40], inputs=[],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=["dreamshaper_8.safetensors"])
    pos = w.node("CLIPTextEncode", [340, 20], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["psychedelic mandala, vivid colors, intricate, highly detailed"])
    neg = w.node("CLIPTextEncode", [340, 200], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["blurry, low quality, watermark, text"])
    lat = w.node("EmptyLatentImage", [340, 380], inputs=[],
                 outputs=[("LATENT", "LATENT")], widgets=[768, 768, 1])
    ks = w.node("KSampler", [660, 120],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                outputs=[("LATENT", "LATENT")],
                widgets=[0, "fixed", 20, 7.0, "euler", "normal", 1.0])
    dec = w.node("VAEDecode", [980, 120],
                 inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], widgets=[])
    cn = w.node("ControlNetLoader", [40, 280], inputs=[],
                outputs=[("CONTROL_NET", "CONTROL_NET")],
                widgets=["control_v11f1e_sd15_tile.pth"])
    setup = w.node("DifforumAnimSetup", [40, 520], inputs=[],
                   outputs=[("params", "DIFFORUM_PARAMS")],
                   widgets=[768, 768, 24, 48, 0, "fixed"])
    cam = w.node("DifforumCamera", [340, 560],
                 inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                 outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
                 widgets=["2d", 40.0, "0:(2)", "0:(0)", "0:(0)", "0:(0)",
                          "0:(0.6)", "0:(0)", "0:(1.02)"])
    strength = w.node("DifforumSchedule", [340, 820],
                      inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                      outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
                      widgets=["0:(0.5)", "linear"])
    prompt = w.node("DifforumPromptSchedule", [340, 1060],
                    inputs=[("params", "DIFFORUM_PARAMS"), ("clip", "CLIP")],
                    outputs=[("positive_schedule", "DIFFORUM_PROMPT"),
                             ("first_frame_cond", "CONDITIONING"), ("info", "STRING")],
                    widgets=["0: psychedelic mandala, vivid colors\n"
                             "24: cosmic fractal temple, golden light\n"
                             "47: deep space nebula, glowing stars", "ease_in_out"])
    fb = w.node("DifforumFeedbackSampler", [1000, 480],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("vae", "VAE"),
                        ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                        ("init_image", "IMAGE"), ("strength_schedule", "DIFFORUM_SCHEDULE"),
                        ("depth", "IMAGE"), ("cfg_schedule", "DIFFORUM_SCHEDULE"),
                        ("positive_schedule", "DIFFORUM_PROMPT"),
                        ("control_net", "CONTROL_NET"), ("control_image", "IMAGE")],
                outputs=[("frames", "IMAGE")],
                widgets=[20, 7.0, "euler", "normal", 0.85, "lab",
                         1.0, 100.0, False, 1.0, 0.6])
    rife = w.node("RIFE VFI", [1360, 360], inputs=[("frames", "IMAGE")],
                  outputs=[("IMAGE", "IMAGE")],
                  widgets=["rife47.pth", 10, 2, True, True, 1.0])
    vhs = w.node("VHS_VideoCombine", [1700, 360], inputs=[("images", "IMAGE")],
                 outputs=[], is_output=True,
                 widgets=[16, 0, "Difforum", "video/h264-mp4", False, True])
    save = w.node("SaveImage", [1360, 620], inputs=[("images", "IMAGE")],
                  outputs=[], widgets=["Difforum"], is_output=True)
    # frame 0 txt2img
    w.link(ckpt, 0, ks, 0, "MODEL")
    w.link(ckpt, 1, pos, 0, "CLIP")
    w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(pos, 0, ks, 1, "CONDITIONING")
    w.link(neg, 0, ks, 2, "CONDITIONING")
    w.link(lat, 0, ks, 3, "LATENT")
    w.link(ks, 0, dec, 0, "LATENT")
    w.link(ckpt, 2, dec, 1, "VAE")
    # control plane
    w.link(setup, 0, cam, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, strength, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, prompt, 0, "DIFFORUM_PARAMS")
    w.link(ckpt, 1, prompt, 1, "CLIP")
    # feedback sampler (+ controlnet + prompt travel)
    w.link(ckpt, 0, fb, 0, "MODEL")
    w.link(pos, 0, fb, 1, "CONDITIONING")
    w.link(neg, 0, fb, 2, "CONDITIONING")
    w.link(ckpt, 2, fb, 3, "VAE")
    w.link(setup, 0, fb, 4, "DIFFORUM_PARAMS")
    w.link(cam, 0, fb, 5, "DIFFORUM_CAMERA")
    w.link(dec, 0, fb, 6, "IMAGE")
    w.link(strength, 0, fb, 7, "DIFFORUM_SCHEDULE")
    w.link(prompt, 0, fb, 10, "DIFFORUM_PROMPT")
    w.link(cn, 0, fb, 11, "CONTROL_NET")
    # output: frames -> interpolate -> video, and also save the PNG sequence
    w.link(fb, 0, rife, 0, "IMAGE")
    w.link(rife, 0, vhs, 0, "IMAGE")
    w.link(fb, 0, save, 0, "IMAGE")
    return w.dump()


def build_intuitive():
    """Intuitive controls: Camera Move (preset) + Prompt Scenes -> feedback video."""
    w = WF()
    ckpt = w.node("CheckpointLoaderSimple", [40, 40], inputs=[],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=["dreamshaper_8.safetensors"])
    pos = w.node("CLIPTextEncode", [340, 20], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["a serene misty forest, soft volumetric light"])
    neg = w.node("CLIPTextEncode", [340, 200], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["blurry, low quality, watermark, text"])
    lat = w.node("EmptyLatentImage", [340, 380], inputs=[],
                 outputs=[("LATENT", "LATENT")], widgets=[768, 768, 1])
    ks = w.node("KSampler", [660, 120],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                outputs=[("LATENT", "LATENT")],
                widgets=[0, "fixed", 20, 7.0, "euler", "normal", 1.0])
    dec = w.node("VAEDecode", [980, 120], inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], widgets=[])
    setup = w.node("DifforumAnimSetup", [40, 460], inputs=[],
                   outputs=[("params", "DIFFORUM_PARAMS")], widgets=[768, 768, 24, 48, 0, "fixed"])
    move = w.node("DifforumCameraMove", [340, 500],
                  inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                  outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
                  widgets=["spiral", 1.0, 1.0, "2d", 40.0])
    strength = w.node("DifforumSchedule", [340, 720],
                      inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                      outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
                      widgets=["0:(0.5)", "linear"])
    scenes = w.node("DifforumPromptScenes", [340, 920],
                    inputs=[("params", "DIFFORUM_PARAMS"), ("clip", "CLIP")],
                    outputs=[("positive_schedule", "DIFFORUM_PROMPT"),
                             ("first_frame_cond", "CONDITIONING"), ("info", "STRING")],
                    widgets=["a serene misty forest, soft light",
                             "a glowing crystal cave, bioluminescent",
                             "a vast starry galaxy, swirling nebula", "", "ease_in_out"])
    fb = w.node("DifforumFeedbackSampler", [1000, 460],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("vae", "VAE"),
                        ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                        ("init_image", "IMAGE"), ("strength_schedule", "DIFFORUM_SCHEDULE"),
                        ("depth", "IMAGE"), ("cfg_schedule", "DIFFORUM_SCHEDULE"),
                        ("positive_schedule", "DIFFORUM_PROMPT")],
                outputs=[("frames", "IMAGE")],
                widgets=[20, 7.0, "euler", "normal", 0.85, "lab", 1.0, 100.0, False, 1.0, 0.6])
    save = w.node("SaveImage", [1360, 460], inputs=[("images", "IMAGE")],
                  outputs=[], widgets=["Difforum"], is_output=True)
    w.link(ckpt, 0, ks, 0, "MODEL"); w.link(ckpt, 1, pos, 0, "CLIP"); w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(pos, 0, ks, 1, "CONDITIONING"); w.link(neg, 0, ks, 2, "CONDITIONING")
    w.link(lat, 0, ks, 3, "LATENT"); w.link(ks, 0, dec, 0, "LATENT"); w.link(ckpt, 2, dec, 1, "VAE")
    w.link(setup, 0, move, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, strength, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, scenes, 0, "DIFFORUM_PARAMS"); w.link(ckpt, 1, scenes, 1, "CLIP")
    w.link(ckpt, 0, fb, 0, "MODEL"); w.link(pos, 0, fb, 1, "CONDITIONING"); w.link(neg, 0, fb, 2, "CONDITIONING")
    w.link(ckpt, 2, fb, 3, "VAE"); w.link(setup, 0, fb, 4, "DIFFORUM_PARAMS")
    w.link(move, 0, fb, 5, "DIFFORUM_CAMERA"); w.link(dec, 0, fb, 6, "IMAGE")
    w.link(strength, 0, fb, 7, "DIFFORUM_SCHEDULE"); w.link(scenes, 0, fb, 10, "DIFFORUM_PROMPT")
    w.link(fb, 0, save, 0, "IMAGE")
    return w.dump()


def build_turbo_live():
    """SD-Turbo / LCM tuned for speed: low steps, cfg 1.0, no ControlNet, 512px.

    Use an SD-Turbo checkpoint (sd_turbo) OR SD1.5 + LCM-LoRA. NOTE: don't mix an
    SD1.5 ControlNet with an SD2.1-based sd_turbo (dim mismatch) - left out here.
    """
    w = WF()
    ckpt = w.node("CheckpointLoaderSimple", [40, 40], inputs=[],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=["sd_turbo.safetensors"])
    pos = w.node("CLIPTextEncode", [340, 20], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["psychedelic mandala, vivid colors, intricate"])
    neg = w.node("CLIPTextEncode", [340, 200], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=["blurry, low quality"])
    lat = w.node("EmptyLatentImage", [340, 380], inputs=[],
                 outputs=[("LATENT", "LATENT")], widgets=[512, 512, 1])
    ks = w.node("KSampler", [660, 120],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                outputs=[("LATENT", "LATENT")],
                widgets=[0, "fixed", 4, 1.0, "euler", "sgm_uniform", 1.0])
    dec = w.node("VAEDecode", [980, 120], inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], widgets=[])
    setup = w.node("DifforumAnimSetup", [40, 460], inputs=[],
                   outputs=[("params", "DIFFORUM_PARAMS")], widgets=[512, 512, 24, 64, 0, "fixed"])
    move = w.node("DifforumCameraMove", [340, 500],
                  inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                  outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
                  widgets=["spiral", 1.0, 1.0, "2d", 40.0])
    strength = w.node("DifforumSchedule", [340, 720],
                      inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                      outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
                      widgets=["0:(0.5)", "linear"])
    scenes = w.node("DifforumPromptScenes", [340, 920],
                    inputs=[("params", "DIFFORUM_PARAMS"), ("clip", "CLIP")],
                    outputs=[("positive_schedule", "DIFFORUM_PROMPT"),
                             ("first_frame_cond", "CONDITIONING"), ("info", "STRING")],
                    widgets=["psychedelic mandala, vivid colors",
                             "cosmic fractal temple, golden light",
                             "deep space nebula, glowing stars", "", "ease_in_out"])
    fb = w.node("DifforumFeedbackSampler", [1000, 460],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("vae", "VAE"),
                        ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                        ("init_image", "IMAGE"), ("strength_schedule", "DIFFORUM_SCHEDULE"),
                        ("depth", "IMAGE"), ("cfg_schedule", "DIFFORUM_SCHEDULE"),
                        ("positive_schedule", "DIFFORUM_PROMPT")],
                outputs=[("frames", "IMAGE")],
                # steps=4, cfg=1.0 -> Turbo/LCM settings
                widgets=[4, 1.0, "euler", "sgm_uniform", 0.7, "lab", 1.0, 100.0, False, 1.0, 0.6])
    save = w.node("SaveImage", [1360, 460], inputs=[("images", "IMAGE")],
                  outputs=[], widgets=["Difforum_turbo"], is_output=True)
    w.link(ckpt, 0, ks, 0, "MODEL"); w.link(ckpt, 1, pos, 0, "CLIP"); w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(pos, 0, ks, 1, "CONDITIONING"); w.link(neg, 0, ks, 2, "CONDITIONING")
    w.link(lat, 0, ks, 3, "LATENT"); w.link(ks, 0, dec, 0, "LATENT"); w.link(ckpt, 2, dec, 1, "VAE")
    w.link(setup, 0, move, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, strength, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, scenes, 0, "DIFFORUM_PARAMS"); w.link(ckpt, 1, scenes, 1, "CLIP")
    w.link(ckpt, 0, fb, 0, "MODEL"); w.link(pos, 0, fb, 1, "CONDITIONING"); w.link(neg, 0, fb, 2, "CONDITIONING")
    w.link(ckpt, 2, fb, 3, "VAE"); w.link(setup, 0, fb, 4, "DIFFORUM_PARAMS")
    w.link(move, 0, fb, 5, "DIFFORUM_CAMERA"); w.link(dec, 0, fb, 6, "IMAGE")
    w.link(strength, 0, fb, 7, "DIFFORUM_SCHEDULE"); w.link(scenes, 0, fb, 10, "DIFFORUM_PROMPT")
    w.link(fb, 0, save, 0, "IMAGE")
    return w.dump()


def build_animatediff():
    """AnimateDiff look + Difforum control: native motion via AnimateDiff-Evolved,
    prompt travel batched by Difforum. Needs comfyui-animatediff-evolved installed.
    batch_size MUST equal max_frames so prompt-batch lines up with the latents.
    """
    N = 16
    w = WF()
    ckpt = w.node("CheckpointLoaderSimple", [40, 40], inputs=[],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=["dreamshaper_8.safetensors"])
    neg = w.node("CLIPTextEncode", [360, 220], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["blurry, low quality, watermark, text"])
    setup = w.node("DifforumAnimSetup", [40, 360], inputs=[],
                   outputs=[("params", "DIFFORUM_PARAMS")], widgets=[512, 512, 12, N, 0, "fixed"])
    scenes = w.node("DifforumPromptScenes", [360, 380],
                    inputs=[("params", "DIFFORUM_PARAMS"), ("clip", "CLIP")],
                    outputs=[("positive_schedule", "DIFFORUM_PROMPT"),
                             ("first_frame_cond", "CONDITIONING"), ("info", "STRING")],
                    widgets=["a serene koi pond, lily pads",
                             "an autumn forest path, falling leaves",
                             "a snowy mountain peak at dawn", "", "ease_in_out"])
    pbatch = w.node("DifforumPromptBatch", [720, 380],
                    inputs=[("positive_schedule", "DIFFORUM_PROMPT")],
                    outputs=[("conditioning", "CONDITIONING")], widgets=[])
    ctx = w.node("ADE_AnimateDiffUniformContextOptions", [40, 560], inputs=[],
                 outputs=[("CONTEXT_OPTIONS", "CONTEXT_OPTIONS")],
                 widgets=[16, 1, 4, False, "flat"])
    ade = w.node("ADE_AnimateDiffLoaderGen1", [360, 560],
                 inputs=[("model", "MODEL"), ("context_options", "CONTEXT_OPTIONS"),
                         ("motion_lora", "MOTION_LORA")],
                 outputs=[("MODEL", "MODEL")],
                 widgets=["v3_sd15_mm.ckpt", "autoselect"])
    lat = w.node("EmptyLatentImage", [360, 760], inputs=[],
                 outputs=[("LATENT", "LATENT")], widgets=[512, 512, N])
    ks = w.node("KSampler", [760, 560],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                outputs=[("LATENT", "LATENT")],
                widgets=[0, "fixed", 20, 7.5, "euler", "normal", 1.0])
    dec = w.node("VAEDecode", [1120, 560],
                 inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], widgets=[])
    vhs = w.node("VHS_VideoCombine", [1400, 560], inputs=[("images", "IMAGE")],
                 outputs=[], is_output=True,
                 widgets=[12, 0, "Difforum_ADE", "video/h264-mp4", False, True])
    w.link(setup, 0, scenes, 0, "DIFFORUM_PARAMS")
    w.link(ckpt, 1, scenes, 1, "CLIP")
    w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(scenes, 0, pbatch, 0, "DIFFORUM_PROMPT")
    w.link(ckpt, 0, ade, 0, "MODEL")
    w.link(ctx, 0, ade, 1, "CONTEXT_OPTIONS")
    w.link(ade, 0, ks, 0, "MODEL")
    w.link(pbatch, 0, ks, 1, "CONDITIONING")
    w.link(neg, 0, ks, 2, "CONDITIONING")
    w.link(lat, 0, ks, 3, "LATENT")
    w.link(ks, 0, dec, 0, "LATENT")
    w.link(ckpt, 2, dec, 1, "VAE")
    w.link(dec, 0, vhs, 0, "IMAGE")
    return w.dump()


def build_ipadapter_coherent():
    """16:9 Classic+ with IPAdapter style-lock for strong frame-to-frame coherence.
    Needs comfyui_ipadapter_plus + an SD1.5 IPAdapter model. Style image stays
    consistent across the whole animation -> much less drift."""
    W, H = 768, 432  # 16:9
    w = WF()
    ckpt = w.node("CheckpointLoaderSimple", [40, 40], inputs=[],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=["dreamshaper_8.safetensors"])
    style = w.node("LoadImage", [40, 260], inputs=[],
                   outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["style.png"])
    ipload = w.node("IPAdapterUnifiedLoader", [360, 40],
                    inputs=[("model", "MODEL"), ("ipadapter", "IPADAPTER")],
                    outputs=[("model", "MODEL"), ("ipadapter", "IPADAPTER")],
                    widgets=["PLUS (high strength)"])
    ipapply = w.node("IPAdapter", [660, 40],
                     inputs=[("model", "MODEL"), ("ipadapter", "IPADAPTER"), ("image", "IMAGE")],
                     outputs=[("MODEL", "MODEL")],
                     widgets=[0.7, "standard", 0.0, 1.0])
    pos = w.node("CLIPTextEncode", [360, 260], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["a dreamlike landscape, painterly, vivid"])
    neg = w.node("CLIPTextEncode", [360, 420], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=["blurry, low quality"])
    lat = w.node("EmptyLatentImage", [360, 580], inputs=[],
                 outputs=[("LATENT", "LATENT")], widgets=[W, H, 1])
    ks = w.node("KSampler", [960, 40],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                outputs=[("LATENT", "LATENT")], widgets=[0, "fixed", 20, 7.0, "euler", "normal", 1.0])
    dec = w.node("VAEDecode", [1260, 40], inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], widgets=[])
    setup = w.node("DifforumAnimSetup", [40, 600], inputs=[],
                   outputs=[("params", "DIFFORUM_PARAMS")], widgets=[W, H, 24, 48, 0, "fixed"])
    cam = w.node("DifforumCameraMove", [360, 760],
                 inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                 outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
                 widgets=["dolly_in", 1.0, 1.0, "2d", 40.0])
    strength = w.node("DifforumSchedule", [360, 980],
                      inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                      outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
                      widgets=["0:(0.45)", "linear"])
    fb = w.node("DifforumFeedbackSampler", [1260, 300],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("vae", "VAE"),
                        ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                        ("init_image", "IMAGE"), ("strength_schedule", "DIFFORUM_SCHEDULE"),
                        ("depth", "IMAGE"), ("cfg_schedule", "DIFFORUM_SCHEDULE"),
                        ("positive_schedule", "DIFFORUM_PROMPT"),
                        ("control_net", "CONTROL_NET"), ("control_image", "IMAGE")],
                outputs=[("frames", "IMAGE")],
                widgets=[20, 7.0, "euler", "normal", 0.85, "lab", 1.0, 100.0, False, 1.0, 0.6])
    save = w.node("SaveImage", [1620, 300], inputs=[("images", "IMAGE")],
                  outputs=[], widgets=["Difforum_ipa"], is_output=True)
    w.link(ckpt, 0, ipload, 0, "MODEL")
    w.link(ipload, 0, ipapply, 0, "MODEL"); w.link(ipload, 1, ipapply, 1, "IPADAPTER")
    w.link(style, 0, ipapply, 2, "IMAGE")
    w.link(ckpt, 1, pos, 0, "CLIP"); w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(ipapply, 0, ks, 0, "MODEL"); w.link(pos, 0, ks, 1, "CONDITIONING")
    w.link(neg, 0, ks, 2, "CONDITIONING"); w.link(lat, 0, ks, 3, "LATENT")
    w.link(ks, 0, dec, 0, "LATENT"); w.link(ckpt, 2, dec, 1, "VAE")
    w.link(setup, 0, cam, 0, "DIFFORUM_PARAMS"); w.link(setup, 0, strength, 0, "DIFFORUM_PARAMS")
    w.link(ipapply, 0, fb, 0, "MODEL"); w.link(pos, 0, fb, 1, "CONDITIONING")
    w.link(neg, 0, fb, 2, "CONDITIONING"); w.link(ckpt, 2, fb, 3, "VAE")
    w.link(setup, 0, fb, 4, "DIFFORUM_PARAMS"); w.link(cam, 0, fb, 5, "DIFFORUM_CAMERA")
    w.link(dec, 0, fb, 6, "IMAGE"); w.link(strength, 0, fb, 7, "DIFFORUM_SCHEDULE")
    w.link(fb, 0, save, 0, "IMAGE")
    return w.dump()


def build_audio_reactive_video():
    """16:9 audio-reactive: bass pumps the zoom, beats pulse the denoise strength."""
    W, H = 768, 432
    w = WF()
    ckpt = w.node("CheckpointLoaderSimple", [40, 40], inputs=[],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=["dreamshaper_8.safetensors"])
    pos = w.node("CLIPTextEncode", [340, 20], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["neon synthwave city, glowing grid, vivid"])
    neg = w.node("CLIPTextEncode", [340, 200], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=["blurry, low quality"])
    lat = w.node("EmptyLatentImage", [340, 360], inputs=[],
                 outputs=[("LATENT", "LATENT")], widgets=[W, H, 1])
    ks = w.node("KSampler", [660, 120],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                outputs=[("LATENT", "LATENT")], widgets=[0, "fixed", 20, 7.0, "euler", "normal", 1.0])
    dec = w.node("VAEDecode", [980, 120], inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], widgets=[])
    setup = w.node("DifforumAnimSetup", [40, 420], inputs=[],
                   outputs=[("params", "DIFFORUM_PARAMS")], widgets=[W, H, 24, 96, 0, "fixed"])
    audio = w.node("LoadAudio", [40, 640], inputs=[],
                   outputs=[("AUDIO", "AUDIO")], widgets=["your_track.mp3"])
    analyzer = w.node("DifforumAudioAnalyzer", [340, 640],
                      inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "AUDIO")],
                      outputs=[("audio_curves", "DIFFORUM_AUDIO"), ("amp", "FLOAT"), ("low", "FLOAT"),
                               ("mid", "FLOAT"), ("high", "FLOAT"), ("onset", "FLOAT"), ("beat", "FLOAT")],
                      widgets=[0.25, True, 1.5])
    # beat -> denoise strength pulse
    asched = w.node("DifforumAudioSchedule", [700, 640],
                    inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                    outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
                    widgets=["beat", "add", 0.4, 0.35, 0.2])
    # bass -> zoom (camera reads audio var `low`)
    cam = w.node("DifforumCamera", [340, 860],
                 inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                 outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
                 widgets=["2d", 40.0, "0:(0)", "0:(0)", "0:(0)", "0:(0)", "0:(0.3)", "0:(1.0 + 0.06*low)"])
    fb = w.node("DifforumFeedbackSampler", [1280, 360],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("vae", "VAE"),
                        ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                        ("init_image", "IMAGE"), ("strength_schedule", "DIFFORUM_SCHEDULE")],
                outputs=[("frames", "IMAGE")],
                widgets=[20, 7.0, "euler", "normal", 0.8, "lab", 1.0, 100.0, False, 1.0, 0.6])
    vhs = w.node("VHS_VideoCombine", [1640, 360], inputs=[("images", "IMAGE")],
                 outputs=[], is_output=True,
                 widgets=[24, 0, "Difforum_audio", "video/h264-mp4", False, True])
    w.link(ckpt, 0, ks, 0, "MODEL"); w.link(ckpt, 1, pos, 0, "CLIP"); w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(pos, 0, ks, 1, "CONDITIONING"); w.link(neg, 0, ks, 2, "CONDITIONING")
    w.link(lat, 0, ks, 3, "LATENT"); w.link(ks, 0, dec, 0, "LATENT"); w.link(ckpt, 2, dec, 1, "VAE")
    w.link(setup, 0, analyzer, 0, "DIFFORUM_PARAMS"); w.link(audio, 0, analyzer, 1, "AUDIO")
    w.link(setup, 0, asched, 0, "DIFFORUM_PARAMS"); w.link(analyzer, 0, asched, 1, "DIFFORUM_AUDIO")
    w.link(setup, 0, cam, 0, "DIFFORUM_PARAMS"); w.link(analyzer, 0, cam, 1, "DIFFORUM_AUDIO")
    w.link(ckpt, 0, fb, 0, "MODEL"); w.link(pos, 0, fb, 1, "CONDITIONING")
    w.link(neg, 0, fb, 2, "CONDITIONING"); w.link(ckpt, 2, fb, 3, "VAE")
    w.link(setup, 0, fb, 4, "DIFFORUM_PARAMS"); w.link(cam, 0, fb, 5, "DIFFORUM_CAMERA")
    w.link(dec, 0, fb, 6, "IMAGE"); w.link(asched, 0, fb, 7, "DIFFORUM_SCHEDULE")
    w.link(fb, 0, vhs, 0, "IMAGE")
    return w.dump()


def build_qrcode_illusion():
    """16:9 hidden-pattern illusion: QR Code Monster ControlNet locks a fixed
    grayscale pattern (spiral/logo/mask) while the feedback loop morphs the scene.
    Needs an SD1.5 checkpoint + control_v1p_sd15_qrcode_monster + a pattern image."""
    W, H = 768, 432
    w = WF()
    ckpt = w.node("CheckpointLoaderSimple", [40, 40], inputs=[],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=["dreamshaper_8.safetensors"])
    pattern = w.node("LoadImage", [40, 260], inputs=[],
                     outputs=[("IMAGE", "IMAGE"), ("MASK", "MASK")], widgets=["pattern.png"])
    cn = w.node("ControlNetLoader", [40, 480], inputs=[],
                outputs=[("CONTROL_NET", "CONTROL_NET")],
                widgets=["control_v1p_sd15_qrcode_monster.safetensors"])
    pos = w.node("CLIPTextEncode", [360, 20], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["ornate fractal mandala, molten gold and emerald, intricate"])
    neg = w.node("CLIPTextEncode", [360, 200], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=["blurry, low quality, flat"])
    lat = w.node("EmptyLatentImage", [360, 360], inputs=[],
                 outputs=[("LATENT", "LATENT")], widgets=[W, H, 1])
    ks = w.node("KSampler", [660, 120],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                outputs=[("LATENT", "LATENT")], widgets=[0, "fixed", 20, 7.0, "euler", "normal", 1.0])
    dec = w.node("VAEDecode", [980, 120], inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], widgets=[])
    setup = w.node("DifforumAnimSetup", [40, 660], inputs=[],
                   outputs=[("params", "DIFFORUM_PARAMS")], widgets=[W, H, 24, 48, 0, "fixed"])
    cam = w.node("DifforumCameraMove", [360, 560],
                 inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                 outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
                 widgets=["zoom_in", 0.5, 0.6, "2d", 40.0])
    strength = w.node("DifforumSchedule", [360, 780],
                      inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                      outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
                      widgets=["0:(0.4)", "linear"])  # lower denoise -> pattern persists
    fb = w.node("DifforumFeedbackSampler", [1280, 300],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("vae", "VAE"),
                        ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                        ("init_image", "IMAGE"), ("strength_schedule", "DIFFORUM_SCHEDULE"),
                        ("depth", "IMAGE"), ("cfg_schedule", "DIFFORUM_SCHEDULE"),
                        ("positive_schedule", "DIFFORUM_PROMPT"),
                        ("control_net", "CONTROL_NET"), ("control_image", "IMAGE")],
                outputs=[("frames", "IMAGE")],
                # control_strength 1.2 -> strong illusion
                widgets=[20, 7.0, "euler", "normal", 0.85, "lab", 1.0, 100.0, False, 1.0, 1.2])
    save = w.node("SaveImage", [1640, 300], inputs=[("images", "IMAGE")],
                  outputs=[], widgets=["Difforum_illusion"], is_output=True)
    w.link(ckpt, 1, pos, 0, "CLIP"); w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(ckpt, 0, ks, 0, "MODEL"); w.link(pos, 0, ks, 1, "CONDITIONING")
    w.link(neg, 0, ks, 2, "CONDITIONING"); w.link(lat, 0, ks, 3, "LATENT")
    w.link(ks, 0, dec, 0, "LATENT"); w.link(ckpt, 2, dec, 1, "VAE")
    w.link(setup, 0, cam, 0, "DIFFORUM_PARAMS"); w.link(setup, 0, strength, 0, "DIFFORUM_PARAMS")
    w.link(ckpt, 0, fb, 0, "MODEL"); w.link(pos, 0, fb, 1, "CONDITIONING")
    w.link(neg, 0, fb, 2, "CONDITIONING"); w.link(ckpt, 2, fb, 3, "VAE")
    w.link(setup, 0, fb, 4, "DIFFORUM_PARAMS"); w.link(cam, 0, fb, 5, "DIFFORUM_CAMERA")
    w.link(dec, 0, fb, 6, "IMAGE"); w.link(strength, 0, fb, 7, "DIFFORUM_SCHEDULE")
    w.link(cn, 0, fb, 11, "CONTROL_NET"); w.link(pattern, 0, fb, 12, "IMAGE")
    w.link(fb, 0, save, 0, "IMAGE")
    return w.dump()


def build_realtime():
    """Native realtime: Difforum Live Sampler runs an internal resident-model loop
    (warp + kaleidoscope + 1-step Turbo/LCM re-diffuse) and streams a LIVE preview
    into the node. Queue once and watch it generate. No external runtime."""
    w = WF()
    ckpt = w.node("CheckpointLoaderSimple", [40, 40], inputs=[],
                  outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
                  widgets=["sd_turbo.safetensors"])
    pos = w.node("CLIPTextEncode", [360, 20], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")],
                 widgets=["psychedelic stained glass mandala, vivid colors, intricate, glowing"])
    neg = w.node("CLIPTextEncode", [360, 160], inputs=[("clip", "CLIP")],
                 outputs=[("CONDITIONING", "CONDITIONING")], widgets=["blurry, low quality"])
    lat = w.node("EmptyLatentImage", [360, 300], inputs=[],
                 outputs=[("LATENT", "LATENT")], widgets=[512, 512, 1])
    ks = w.node("KSampler", [680, 60],
                inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                        ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
                outputs=[("LATENT", "LATENT")],
                widgets=[0, "fixed", 4, 1.0, "euler", "sgm_uniform", 1.0])
    dec = w.node("VAEDecode", [1000, 60], inputs=[("samples", "LATENT"), ("vae", "VAE")],
                 outputs=[("IMAGE", "IMAGE")], widgets=[])
    setup = w.node("DifforumAnimSetup", [40, 360], inputs=[],
                   outputs=[("params", "DIFFORUM_PARAMS")], widgets=[512, 512, 24, 120, 0, "fixed"])
    move = w.node("DifforumCameraMove", [360, 460],
                  inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
                  outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
                  widgets=["spiral", 1.0, 1.0, "2d", 40.0])
    live = w.node("DifforumLiveSampler", [1000, 320],
                  inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                          ("negative", "CONDITIONING"), ("vae", "VAE"),
                          ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                          ("init_image", "IMAGE")],
                  outputs=[("frames", "IMAGE")],
                  # duration, strength, steps, cfg, sampler, scheduler, color_coh,
                  # color_mode, symmetry, segments, target_fps
                  widgets=[240, 0.5, 1, 1.0, "lcm", "sgm_uniform", 0.5, "lab",
                           "kaleidoscope", 6, 0.0])
    save = w.node("SaveImage", [1360, 320], inputs=[("images", "IMAGE")],
                  outputs=[], widgets=["Difforum_live"], is_output=True)
    w.link(ckpt, 0, ks, 0, "MODEL"); w.link(ckpt, 1, pos, 0, "CLIP"); w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(pos, 0, ks, 1, "CONDITIONING"); w.link(neg, 0, ks, 2, "CONDITIONING")
    w.link(lat, 0, ks, 3, "LATENT"); w.link(ks, 0, dec, 0, "LATENT"); w.link(ckpt, 2, dec, 1, "VAE")
    w.link(setup, 0, move, 0, "DIFFORUM_PARAMS")
    w.link(ckpt, 0, live, 0, "MODEL"); w.link(pos, 0, live, 1, "CONDITIONING")
    w.link(neg, 0, live, 2, "CONDITIONING"); w.link(ckpt, 2, live, 3, "VAE")
    w.link(setup, 0, live, 4, "DIFFORUM_PARAMS"); w.link(move, 0, live, 5, "DIFFORUM_CAMERA")
    w.link(dec, 0, live, 6, "IMAGE")
    w.link(live, 0, save, 0, "IMAGE")
    return w.dump()


def build_mesmerize():
    """Living kaleidoscope: feedback loop with in-loop symmetry (the diffusion
    heals the seams each frame) + Echo Trails for smooth hypnotic motion."""
    w = WF()
    ckpt = w.node(
        "CheckpointLoaderSimple", [40, 40], inputs=[],
        outputs=[("MODEL", "MODEL"), ("CLIP", "CLIP"), ("VAE", "VAE")],
        widgets=["dreamshaper_8.safetensors"],
    )
    pos = w.node(
        "CLIPTextEncode", [340, 20], inputs=[("clip", "CLIP")],
        outputs=[("CONDITIONING", "CONDITIONING")],
        widgets=["ornate symmetric mandala, stained glass, iridescent, "
                 "intricate fractal detail, glowing"],
    )
    neg = w.node(
        "CLIPTextEncode", [340, 200], inputs=[("clip", "CLIP")],
        outputs=[("CONDITIONING", "CONDITIONING")],
        widgets=["blurry, low quality, watermark, text, asymmetric"],
    )
    lat = w.node(
        "EmptyLatentImage", [340, 380], inputs=[],
        outputs=[("LATENT", "LATENT")], widgets=[768, 768, 1],
    )
    ks = w.node(
        "KSampler", [660, 120],
        inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"), ("latent_image", "LATENT")],
        outputs=[("LATENT", "LATENT")],
        widgets=[0, "fixed", 20, 7.0, "euler", "normal", 1.0],
    )
    dec = w.node(
        "VAEDecode", [980, 120],
        inputs=[("samples", "LATENT"), ("vae", "VAE")],
        outputs=[("IMAGE", "IMAGE")], widgets=[],
    )
    setup = w.node(
        "DifforumAnimSetup", [40, 520], inputs=[],
        outputs=[("params", "DIFFORUM_PARAMS")],
        widgets=[768, 768, 24, 72, 0, "fixed"],
    )
    # slow rotation + gentle zoom feeds the kaleidoscope new material
    cam = w.node(
        "DifforumCamera", [340, 560],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("camera", "DIFFORUM_CAMERA"), ("info", "STRING")],
        widgets=["2d", 40.0, "0:(0)", "0:(0)", "0:(1.0)", "0:(0.6)",
                 "0:(0)", "0:(0)", "0:(1.01)"],
    )
    strength = w.node(
        "DifforumSchedule", [340, 820],
        inputs=[("params", "DIFFORUM_PARAMS"), ("audio", "DIFFORUM_AUDIO")],
        outputs=[("schedule", "DIFFORUM_SCHEDULE"), ("values", "FLOAT")],
        widgets=["0:(0.4)", "linear"],
    )
    fb = w.node(
        "DifforumFeedbackSampler", [1000, 480],
        inputs=[("model", "MODEL"), ("positive", "CONDITIONING"),
                ("negative", "CONDITIONING"), ("vae", "VAE"),
                ("params", "DIFFORUM_PARAMS"), ("camera", "DIFFORUM_CAMERA"),
                ("init_image", "IMAGE"), ("strength_schedule", "DIFFORUM_SCHEDULE")],
        outputs=[("frames", "IMAGE")],
        # ...,control_strength, symmetry, symmetry_segments
        widgets=[20, 7.0, "euler", "normal", 0.85, "lab", 1.0, 100.0, False, 1.0,
                 0.6, "kaleidoscope", 6],
    )
    echo = w.node(
        "DifforumEchoTrails", [1360, 360],
        inputs=[("frames", "IMAGE")],
        outputs=[("frames", "IMAGE")], widgets=[0.6, 0.4],
    )
    save = w.node(
        "SaveImage", [1680, 360], inputs=[("images", "IMAGE")],
        outputs=[], widgets=["Difforum_mesmerize"], is_output=True,
    )
    # frame 0 txt2img
    w.link(ckpt, 0, ks, 0, "MODEL")
    w.link(ckpt, 1, pos, 0, "CLIP")
    w.link(ckpt, 1, neg, 0, "CLIP")
    w.link(pos, 0, ks, 1, "CONDITIONING")
    w.link(neg, 0, ks, 2, "CONDITIONING")
    w.link(lat, 0, ks, 3, "LATENT")
    w.link(ks, 0, dec, 0, "LATENT")
    w.link(ckpt, 2, dec, 1, "VAE")
    # difforum control plane
    w.link(setup, 0, cam, 0, "DIFFORUM_PARAMS")
    w.link(setup, 0, strength, 0, "DIFFORUM_PARAMS")
    # feedback sampler with in-loop kaleidoscope
    w.link(ckpt, 0, fb, 0, "MODEL")
    w.link(pos, 0, fb, 1, "CONDITIONING")
    w.link(neg, 0, fb, 2, "CONDITIONING")
    w.link(ckpt, 2, fb, 3, "VAE")
    w.link(setup, 0, fb, 4, "DIFFORUM_PARAMS")
    w.link(cam, 0, fb, 5, "DIFFORUM_CAMERA")
    w.link(dec, 0, fb, 6, "IMAGE")
    w.link(strength, 0, fb, 7, "DIFFORUM_SCHEDULE")
    w.link(fb, 0, echo, 0, "IMAGE")
    w.link(echo, 0, save, 0, "IMAGE")
    return w.dump()


def add_note(data, text, pos=(40, -210)):
    """Append a frontend Note node (yellow sticky) with a model checklist."""
    nid = data["last_node_id"] + 1
    data["last_node_id"] = nid
    data["nodes"].append({
        "id": nid, "type": "Note", "pos": list(pos), "size": [460, 300],
        "flags": {}, "order": len(data["nodes"]), "mode": 0,
        "inputs": [], "outputs": [], "title": "READ ME - models needed",
        "properties": {"text": ""}, "widgets_values": [text],
        "color": "#432", "bgcolor": "#653",
    })


# concise, real download pointers per template (folder shown after ->)
NOTES = {
    "difforum_schedule_basic.json":
        "Difforum: Schedule basics.\n\nNo models needed - this runs anywhere "
        "(schedules + curve plot only).",
    "difforum_models_info.json":
        "Difforum: Reference card.\n\nNo models needed. Model Profile auto-detects "
        "your VRAM; Model Catalog lists recipes + download links.",
    "difforum_audio_reactive.json":
        "Difforum: Audio analysis (no render).\n\nNeeds:\n"
        "- an audio file (.mp3/.wav) -> dropped in the Load Audio node\n"
        "No checkpoint required.",
    "difforum_camera_warp.json":
        "Difforum: Warp preview.\n\nNeeds:\n- an input image -> Load Image\n"
        "No checkpoint (this only previews the 2D/3D warp + occlusion mask).\n"
        "Tip: uses the intuitive Camera Move presets.",
    "difforum_hybrid_wan_guides.json":
        "Difforum: Hybrid bridge -> Wan 2.2 VACE.\n\nThis builds camera-warped GUIDE "
        "frames. To actually render, wire `guide_frames` into a Wan 2.2 VACE graph.\n\n"
        "Models:\n- anchor image -> Load Image\n"
        "- Wan 2.2 5B GGUF -> models/unet  (hf: QuantStack/Wan2.2-TI2V-5B-GGUF)\n"
        "- umt5 text encoder -> models/text_encoders  (hf: Comfy-Org/Wan_2.1_ComfyUI_repackaged)\n"
        "- wan_2.1_vae.safetensors -> models/vae\n"
        "- nodes: ComfyUI-WanVideoWrapper + ComfyUI-GGUF",
    "difforum_feedback_classic.json":
        "Difforum: Classic+ feedback video.\n\nModels:\n"
        "- SD1.5 checkpoint -> models/checkpoints  (DreamShaper: civitai.com/models/4384)\n"
        "- (optional) better VAE vae-ft-mse-840000 -> models/vae  (hf: stabilityai/sd-vae-ft-mse)\n\n"
        "Quality upgrade: swap the checkpoint for SDXL or Flux (model-agnostic).\n"
        "Tip: try Camera Move + Prompt Scenes for an easier setup.",
    "difforum_intuitive_controls.json":
        "Difforum: Intuitive controls (Camera Move presets + Prompt Scenes).\n\nModels:\n"
        "- SD1.5 checkpoint -> models/checkpoints  (DreamShaper: civitai.com/models/4384)\n"
        "Swap to SDXL/Flux for higher quality.",
    "difforum_turbo_live.json":
        "Difforum: Turbo/LCM fast feedback (4 steps, cfg 1).\n\nModels:\n"
        "- SD-Turbo -> models/checkpoints  (hf: stabilityai/sd-turbo)\n"
        "  or any SD1.5 + LCM-LoRA (hf: latent-consistency/lcm-lora-sdv1-5 -> models/loras)\n"
        "This is the path toward realtime.",
    "difforum_deluxe_travel_controlnet_video.json":
        "Difforum: Deluxe (travel + ControlNet + interpolation + MP4).\n\nModels:\n"
        "- SD1.5 checkpoint -> models/checkpoints  (DreamShaper: civitai.com/models/4384)\n"
        "- tile ControlNet control_v11f1e_sd15_tile -> models/controlnet\n"
        "  (hf: comfyanonymous/ControlNet-v1-1_fp16_safetensors)\n"
        "- nodes: ComfyUI-Frame-Interpolation (RIFE auto-downloads) + ComfyUI-VideoHelperSuite\n"
        "ControlNet must match the base family (SD1.5 here).",
    "difforum_ipadapter_coherent.json":
        "Difforum: IPAdapter style-lock (coherence).\n\nModels:\n"
        "- SD1.5 checkpoint -> models/checkpoints\n"
        "- IP-Adapter ip-adapter-plus_sd15.safetensors -> models/ipadapter  (hf: h94/IP-Adapter)\n"
        "- CLIP-Vision (ViT-H) -> models/clip_vision  (hf: h94/IP-Adapter, image_encoder)\n"
        "- a style/reference image -> Load Image\n"
        "- node: ComfyUI_IPAdapter_plus",
    "difforum_audio_reactive_video.json":
        "Difforum: Audio-reactive video (bass->zoom, beat->strength).\n\nModels:\n"
        "- SD1.5 checkpoint -> models/checkpoints\n"
        "- an audio file (.mp3/.wav) -> Load Audio\n"
        "- node: ComfyUI-VideoHelperSuite (for the MP4)",
    "difforum_animatediff_sd15.json":
        "Difforum: AnimateDiff bridge (native motion).\n\nModels:\n"
        "- SD1.5 checkpoint -> models/checkpoints\n"
        "- motion module mm_sd15_v3.ckpt -> models/animatediff_models  (hf: guoyww/animatediff)\n"
        "- node: ComfyUI-AnimateDiff-Evolved",
    "difforum_qrcode_illusion.json":
        "Difforum: Hidden-pattern illusion (QR Monster).\n\nModels:\n"
        "- SD1.5 checkpoint -> models/checkpoints\n"
        "- control_v1p_sd15_qrcode_monster -> models/controlnet  (hf: monster-labs/control_v1p_sd15_qrcode_monster)\n"
        "- a grayscale pattern image (spiral/logo/mask) -> Load Image\n"
        "control_strength ~1.2 = strong illusion; lower for subtle.",
    "difforum_realtime_live.json":
        "Difforum: Native realtime (Live Sampler internal loop).\n\n"
        "Queue ONCE and watch the node stream a live preview as it generates - the\n"
        "Live Sampler keeps the model resident and loops warp + kaleidoscope +\n"
        "1-step re-diffuse internally. No external runtime.\n\nModels:\n"
        "- SD-Turbo -> models/checkpoints  (hf: stabilityai/sd-turbo)\n"
        "  or any SD1.5 + LCM-LoRA (hf: latent-consistency/lcm-lora-sdv1-5)\n"
        "Use 512x512, 1-2 steps, cfg 1. Add TensorRT + TAESD for max FPS.\n\n"
        "Magic mirror: set the Live Sampler's live_source to 0 (webcam) or a video\n"
        "path to stylize a live feed (kaleidoscope your camera). Needs opencv.\n"
        "Live output: set stream_dir (folder for OBS/Resolume) or spout_name\n"
        "(needs SpoutGL). See REALTIME.md.",
    "difforum_mesmerize_kaleidoscope.json":
        "Difforum: Living kaleidoscope (in-loop symmetry + Echo Trails).\n\nModels:\n"
        "- SD1.5 checkpoint -> models/checkpoints  (DreamShaper: civitai.com/models/4384)\n"
        "Swap to SDXL/Flux for higher quality.\n\n"
        "How it works: the Feedback Sampler's `symmetry` = kaleidoscope folds each\n"
        "warped frame, then the diffusion heals the seams, so the pattern grows and\n"
        "stays symmetric. Echo Trails adds smooth motion-blur trails.\n"
        "Try symmetry mirror_h/mirror_v/mirror_quad, or raise symmetry_segments.",
}


def main():
    out = {
        "difforum_qrcode_illusion.json": build_qrcode_illusion(),
        "difforum_ipadapter_coherent.json": build_ipadapter_coherent(),
        "difforum_audio_reactive_video.json": build_audio_reactive_video(),
        "difforum_animatediff_sd15.json": build_animatediff(),
        "difforum_turbo_live.json": build_turbo_live(),
        "difforum_intuitive_controls.json": build_intuitive(),
        "difforum_schedule_basic.json": build_basic(),
        "difforum_audio_reactive.json": build_audio(),
        "difforum_feedback_classic.json": build_feedback(),
        "difforum_camera_warp.json": build_camera_warp(),
        "difforum_hybrid_wan_guides.json": build_hybrid_guides(),
        "difforum_models_info.json": build_models_info(),
        "difforum_deluxe_travel_controlnet_video.json": build_deluxe(),
        "difforum_mesmerize_kaleidoscope.json": build_mesmerize(),
        "difforum_realtime_live.json": build_realtime(),
    }
    for name, data in out.items():
        if name in NOTES:
            add_note(data, NOTES[name])
        path = HERE / name
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"wrote {path.name}  ({len(data['nodes'])} nodes, {len(data['links'])} links)")
    print("OK")


if __name__ == "__main__":
    main()
