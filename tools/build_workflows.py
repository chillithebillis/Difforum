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
    # core Depth Anything 3 (models/geometry_estimation). DynamicCombo sub-values are left to
    # their defaults (mono, v2_style = near white, no sky clip).
    "LoadDA3Model": ([], ["model_name", "weight_dtype"], [("DA3_MODEL", "DA3_MODEL")]),
    "DA3Inference": ([("da3_model", "DA3_MODEL"), ("image", "IMAGE")], ["resolution", "resize_method", "mode"],
                     [("da3_geometry", "DA3_GEOMETRY")]),
    "DA3Render": ([("da3_geometry", "DA3_GEOMETRY")], ["output"], [("IMAGE", "IMAGE")]),
    "ControlNetLoader": ([], ["control_net_name"], [("CONTROL_NET", "CONTROL_NET")]),
    "SetUnionControlNetType": ([("control_net", "CONTROL_NET")], ["type"], [("CONTROL_NET", "CONTROL_NET")]),
    "ControlNetApplyAdvanced": ([("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
                                 ("control_net", "CONTROL_NET"), ("image", "IMAGE"), ("vae", "VAE")],
                                ["strength", "start_percent", "end_percent"],
                                [("positive", "CONDITIONING"), ("negative", "CONDITIONING")]),
    "Canny": ([("image", "IMAGE")], ["low_threshold", "high_threshold"], [("IMAGE", "IMAGE")]),
    "ImageScaleToTotalPixels": ([("image", "IMAGE")], ["upscale_method", "megapixels", "resolution_steps"],
                                [("IMAGE", "IMAGE")]),
    "VAEEncode": ([("pixels", "IMAGE"), ("vae", "VAE")], [], [("LATENT", "LATENT")]),
    # AnimateDiff-Evolved (Kosinkadink)
    "ADE_StandardUniformContextOptions": ([("prev_context", "CONTEXT_OPTIONS"), ("view_opts", "VIEW_OPTS")],
                                          ["context_length", "context_stride", "context_overlap", "fuse_method",
                                           "use_on_equal_length", "start_percent", "guarantee_steps"],
                                          [("CONTEXT_OPTS", "CONTEXT_OPTIONS")]),
    "ADE_AnimateDiffLoaderGen1": ([("model", "MODEL"), ("context_options", "CONTEXT_OPTIONS"),
                                   ("motion_lora", "MOTION_LORA"), ("ad_settings", "AD_SETTINGS"),
                                   ("ad_keyframes", "AD_KEYFRAMES"), ("sample_settings", "SAMPLE_SETTINGS"),
                                   ("scale_multival", "MULTIVAL"), ("effect_multival", "MULTIVAL"),
                                   ("per_block", "PER_BLOCK")], ["model_name", "beta_schedule"],
                                  [("MODEL", "MODEL")]),
    "EmptyLTXVLatentVideo": ([], ["width", "height", "length", "batch_size"], [("LATENT", "LATENT")]),
    "MarkdownNote": ([], ["text"], []),
    "BatchImagesNode": ([("images.image0", "IMAGE"), ("images.image1", "IMAGE"), ("images.image2", "IMAGE")],
                        [], [("IMAGE", "IMAGE")]),
    # MiniMax H3 (ComfyUI core) - names match the official templates
    "UNETLoader": ([], ["unet_name", "weight_dtype"], [("MODEL", "MODEL")]),
    "LoraLoaderModelOnly": ([("model", "MODEL")], ["lora_name", "strength_model"], [("MODEL", "MODEL")]),
    "CLIPLoader": ([], ["clip_name", "type", "device"], [("CLIP", "CLIP")]),
    "VAELoader": ([], ["vae_name"], [("VAE", "VAE")]),
    "MiniMaxH3ImageToVideo": ([("clip", "CLIP"), ("vae", "VAE"), ("first_frame", "IMAGE"), ("last_frame", "IMAGE")],
                              ["prompt", "width", "height", "length"],
                              [("positive", "CONDITIONING"), ("LATENT", "LATENT")]),
    "MiniMaxH3ReferenceToVideo": ([("clip", "CLIP"), ("vae", "VAE"), ("audio_vae", "VAE"),
                                   ("ref_images.ref_image_0", "IMAGE")],
                                  ["prompt", "width", "height", "length", "ref_image_size"],
                                  [("positive", "CONDITIONING"), ("LATENT", "LATENT")]),
    "RandomNoise": ([], ["noise_seed", "control_after_generate"], [("NOISE", "NOISE")]),
    "KSamplerSelect": ([], ["sampler_name"], [("SAMPLER", "SAMPLER")]),
    "BasicScheduler": ([("model", "MODEL")], ["scheduler", "steps", "denoise"], [("SIGMAS", "SIGMAS")]),
    "BasicGuider": ([("model", "MODEL"), ("conditioning", "CONDITIONING")], [], [("GUIDER", "GUIDER")]),
    "SamplerCustomAdvanced": ([("noise", "NOISE"), ("guider", "GUIDER"), ("sampler", "SAMPLER"),
                               ("sigmas", "SIGMAS"), ("latent_image", "LATENT")], [],
                              [("output", "LATENT"), ("denoised_output", "LATENT")]),
    "VAEDecodeAudio": ([("samples", "LATENT"), ("vae", "VAE")], [], [("AUDIO", "AUDIO")]),
    # KJNodes (optional): live previews while sampling
    "ModelPreviewOverrideKJ": ([("model", "MODEL"), ("vae", "VAE"), ("audio_vae", "VAE")],
                               ["max_resolution", "jpeg_quality", "suppress_default_preview", "preview_frames",
                                "preview_fps", "tiny_vae"], [("MODEL", "MODEL")]),
    "UpscaleModelLoader": ([], ["model_name"], [("UPSCALE_MODEL", "UPSCALE_MODEL")]),
    # MiniMax H3 latent upscale (two-stage): core AV split/concat + LBH-123-AI upscaler.
    # Its `mode` is a DynamicCombo: the frontend restores the sub-value (scale 2.0) from its
    # default, so it is left out of widgets_values or every later widget shifts by one.
    "LTXVSeparateAVLatent": ([("av_latent", "LATENT")], [], [("video_latent", "LATENT"), ("audio_latent", "LATENT")]),
    "LTXVConcatAVLatent": ([("video_latent", "LATENT"), ("audio_latent", "LATENT")], [], [("latent", "LATENT")]),
    "ManualSigmas": ([], ["sigmas"], [("SIGMAS", "SIGMAS")]),
    "MinimaxH3LatentUpscaler3D": ([("latent", "*")], ["model_name", "mode", "align",
                                                     "enable_temporal_chunking", "force_unload", "device",
                                                     "precision"], [("latent", "*")]),
    "LoadVideo": ([], ["file", "upload"], [("VIDEO", "VIDEO")]),
    "GetVideoComponents": ([("video", "VIDEO")], [], [("images", "IMAGE"), ("audio", "AUDIO"), ("fps", "FLOAT"),
                                                     ("bit_depth", "COMBO"), ("color_space", "COMBO")]),
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
    """Template builder. Nodes are added inside named blocks (ComfyUI groups);
    json() lays every block out left to right along the data flow, so no
    coordinates are hand-placed and groups never overlap."""

    def __init__(self, title):
        self.title = title
        self.nodes, self.links = [], []
        self.blocks = []            # [{title, color, col, row, ids}]
        self._cur = None

    # -- blocks ------------------------------------------------------------
    def block(self, title, color, col=None, row=None):
        """col/row are ignored: blocks are placed by role (see _place)."""
        g = self

        class _Ctx:
            def __enter__(self_):
                blk = next((b for b in g.blocks if b["title"] == title), None)
                if blk is None:
                    r, under = _place(title)
                    blk = {"title": title, "color": color, "row": r, "under": under, "ids": []}
                    g.blocks.append(blk)
                self_.prev, g._cur = g._cur, blk
                return blk

            def __exit__(self_, *exc):
                g._cur = self_.prev
        return _Ctx()

    # -- nodes -------------------------------------------------------------
    def add(self, type_, size=(340, 200), title=None, mode=0, props=None, **values):
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
        if self._cur is None:
            raise RuntimeError(f"{type_} added outside a block")
        # the frontend grows a node to fit its slots and widgets; reserve that height so
        # stacked nodes never overlap
        n_w = sum(1 for w in widgets if not w.endswith(".control"))
        est = 12 + 21 * max(len(links), len(outs)) + 24 * n_w
        size = (size[0], max(size[1], est))
        node = {
            "id": len(self.nodes) + 1, "type": type_, "pos": [0, 0], "size": list(size),
            "flags": {}, "order": len(self.nodes), "mode": mode,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in links],
            "outputs": [{"name": n, "type": t, "links": [], "slot_index": i}
                        for i, (n, t) in enumerate(outs)],
            "properties": {"Node name for S&R": type_, **(props or {})},
            "widgets_values": wv,
        }
        if title:
            node["title"] = title
        self.nodes.append(node)
        self._cur["ids"].append(node["id"])
        return node["id"]

    def note(self, text, size=(420, 260)):
        return self.add("MarkdownNote", size=size, text=text)

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

    # -- layout ------------------------------------------------------------
    TITLE, GAP_X, GAP_Y, PAD, HEAD, BLOCK_GAP = 30, 40, 26, 24, 56, 70

    def _depths(self):
        preds = {n["id"]: [] for n in self.nodes}
        for _lid, s, _o, d, _i, _t in self.links:
            preds[d].append(s)
        memo = {}

        def depth(i):
            if i not in memo:
                memo[i] = 0
                memo[i] = max((depth(p) + 1 for p in preds[i]), default=0)
            return memo[i]
        return {i: depth(i) for i in preds}

    def _layout(self):
        dep = self._depths()
        by_id = {n["id"]: n for n in self.nodes}
        sizes = {}
        for b in self.blocks:
            cols = sorted({dep[i] for i in b["ids"]})
            col_of = {d: k for k, d in enumerate(cols)}
            columns = [[] for _ in cols]
            for i in b["ids"]:
                columns[col_of[dep[i]]].append(i)
            x, local, bw, bh = 0, {}, 0, 0
            for ids in columns:
                w = max(by_id[i]["size"][0] for i in ids)
                y = 0
                for i in ids:
                    local[i] = (x, y + self.TITLE)
                    y += self.TITLE + by_id[i]["size"][1] + self.GAP_Y
                bh = max(bh, y - self.GAP_Y)
                x += w + self.GAP_X
                bw = x - self.GAP_X
            b["local"], b["w"], b["h"] = local, bw + 2 * self.PAD, bh + self.HEAD + self.PAD
            sizes[b["title"]] = (b["w"], b["h"])
        # rows top to bottom; in a row, slots left to right in creation order; a block
        # with `under` stacks below the block whose title starts with it
        slots = {}                      # row -> [[blocks]]
        for b in self.blocks:
            row = slots.setdefault(b["row"], [])
            host = next((sl for r in slots.values() for sl in r
                         if b["under"] and sl[0]["title"].startswith(b["under"])), None)
            if host is not None:
                host.append(b)
            else:
                row.append([b])
        pos, gy = {}, 0
        for r in sorted(slots):
            gx, rh = 0, 0
            for sl in slots[r]:
                sy, sw = gy, 0
                for b in sl:
                    pos[b["title"]] = (gx, sy)
                    sy += b["h"] + self.BLOCK_GAP
                    sw = max(sw, b["w"])
                rh = max(rh, sy - gy - self.BLOCK_GAP)
                gx += sw + self.BLOCK_GAP
            gy += rh + self.BLOCK_GAP
        groups = []
        for gi, b in enumerate(self.blocks, 1):
            gx, gy = pos[b["title"]]
            for i, (lx, ly) in b["local"].items():
                by_id[i]["pos"] = [gx + self.PAD + lx, gy + self.HEAD + ly - self.TITLE + 10]
            groups.append({"id": gi, "title": b["title"], "bounding": [gx, gy, b["w"], b["h"]],
                           "color": b["color"], "font_size": 22, "flags": {}})
        return groups

    def _switch_props(self):
        """Tell Workflow Switches how each group turns off: mute (skip it and what needs it),
        bypass (data passes through), or locked (always on, locate only)."""
        locked, modes = [], {}
        for b in self.blocks:
            t = b["title"]
            if t in BYPASS_BLOCKS:
                modes[t] = "bypass"
            elif t in LOCKED_BLOCKS or t.startswith("1 · "):
                locked.append(t)
            else:
                modes[t] = "mute"
        for n in self.nodes:
            if n["type"] == "Difforum_Switches":
                n["properties"].update(locked=locked, modes=modes)
                n["size"] = [340, 24 + 34 * len(self.blocks)]

    def json(self):
        self._switch_props()
        groups = self._layout()
        return {
            "last_node_id": len(self.nodes), "last_link_id": len(self.links),
            "nodes": self.nodes, "links": self.links, "groups": groups, "config": {},
            "extra": {"ds": {"scale": 0.55, "offset": [60, 80]}, "difforum": "1.0",
                      "title": self.title},
            "version": 0.4,
        }


def tl_json(frames, **patch):
    tl = default_timeline(frames)
    for k, v in patch.items():
        tl[k] = v
    return json.dumps(parse_timeline(tl))


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------

C_CONTROL, C_DIRECT, C_PREVIZ, C_MODELS = "#444", "#3f789e", "#8AA", "#88A"
C_RENDER, C_LIVE, C_FILL, C_STYLE, C_UP, C_OUT, C_MISC = (
    "#8A8", "#b58b2a", "#b06634", "#a1309b", "#b58b2a", "#8A8", "#A88")

B_CONTROL = "0 · Control"
B_DIRECT = "1 · Direction"
B_PREVIZ = "2 · Previz"
B_MODELS = "Models"
B_FIRST = "First frame"
B_LIVE = "Live Preview (TAEH3)"
B_FILL = "Fill Reveal (AI)"
B_RESTYLE = "Restyle (look pass)"
B_LOOKMIX = "Look Mix"
B_UPSCALE = "Upscale 2K"
B_H3UP = "H3 Latent Upscale (x2)"
B_POLISH = "Keyframe Polish"
B_DEPTH = "Depth (Depth Anything 3)"
B_STRUCT = "Structure (ControlNet)"
B_AD = "AnimateDiff (motion module)"
B_OUTPUT = "Output"
B_LOOKPASS = "3 · Look pass (Feedback)"
BYPASS_BLOCKS = {B_LIVE, B_FILL, B_RESTYLE, B_LOOKMIX, B_UPSCALE, B_H3UP, B_POLISH, B_DEPTH, B_STRUCT, B_AD, "Audio"}
LOCKED_BLOCKS = {B_CONTROL, B_MODELS, B_FIRST, B_LOOKPASS, "3 · H3 guides", "3 · H3 shot"}
UNDER = {B_LIVE: "3 · Render", B_LOOKMIX: B_RESTYLE, "4 · Import": "3 · Export"}
ROW0_PREFIX = ("0 · ", "1 · Direction", "1 · Source", "2 · ", "3 · H3", "3 · Look pass", "3 · LTX",
               "3 · Storyboard", "3 · Export", "Audio", B_FILL, B_POLISH, B_DEPTH)


def _place(title):
    """Row 0 = plan (control, direction, previz, bridges); row 1 = the render pipeline."""
    if title in UNDER:
        return 1, UNDER[title]
    return (0 if title.startswith(ROW0_PREFIX) else 1), None


def control(g, readme, size=(460, 300)):
    """Switches + read-me, top left of every template."""
    with g.block(B_CONTROL, C_CONTROL, col=0, row=0):
        g.add("Difforum_Switches", size=(320, 260), title="Workflow Switches")
        g.note(readme, size=size)


def previz(g, d, init=None, keys=None, depth=None, col=2, row=0):
    """Animatic -> video: previz the whole shot before rendering anything heavy."""
    with g.block(B_PREVIZ, C_PREVIZ, col=col, row=row):
        g.note("## Previz first\n\nThe **Animatic** plays the Director's camera, scenes, energy and keys "
               "over the first frame in seconds.\n\nSwitch **Render** off (or press **Previz only** on "
               "the Director) and Queue: only this runs.", size=(360, 200))
        an = g.add("Difforum_Animatic", size=(360, 180), title="Animatic (previz)")
        g.link(d, "direction", an, "direction")
        if init:
            g.link(init[0], init[1], an, "init_image")
        if depth:
            g.link(depth[0], depth[1], an, "depth")
        if keys:
            g.link(keys[0], "keyframes", an, "key_images")
            g.link(keys[0], "indices", an, "key_indices")
        cv = g.add("CreateVideo", size=(260, 100), fps=24.0)
        g.link(an, "frames", cv, "images")
        g.link(an, "fps", cv, "fps")
        sv = g.add("SaveVideo", size=(360, 320), title="Previz video",
                   filename_prefix="video/difforum_previz", format="auto", codec="auto")
        g.link(cv, "VIDEO", sv, "video")
    return an


def upscale(g, src, col, row=0, frames_out="frames"):
    """Model upscale + exact 2K resize; switch the group off to deliver at render size."""
    with g.block(B_UPSCALE, C_UP, col=col, row=row):
        um = g.add("UpscaleModelLoader", size=(320, 80), title="Upscale model",
                   model_name="RealESRGAN_x4plus.safetensors")
        up = g.add("Difforum_Upscale", size=(320, 200), target="2K (2048 long edge)")
        g.link(src[0], src[1], up, "frames")
        g.link(um, "UPSCALE_MODEL", up, "upscale_model")
    return up, "frames"


def output(g, src, col, row=0, prefix="video/difforum", audio=None, fps_src=None):
    with g.block(B_OUTPUT, C_OUT, col=col, row=row):
        cv = g.add("CreateVideo", size=(260, 100), fps=24.0)
        g.link(src[0], src[1], cv, "images")
        if audio:
            g.link(audio[0], audio[1], cv, "audio")
        if fps_src:
            g.link(fps_src[0], fps_src[1], cv, "fps")
        sv = g.add("SaveVideo", size=(480, 420), filename_prefix=prefix, format="auto", codec="auto")
        g.link(cv, "VIDEO", sv, "video")
    return sv


def sd_models(g, ckpt="sd_xl_base_1.0.safetensors", pos_text="", neg_text="", title="Checkpoint",
              pos_title="Look / style (positive)"):
    ck = g.add("CheckpointLoaderSimple", size=(320, 100), title=title, ckpt_name=ckpt)
    pos = g.add("CLIPTextEncode", size=(380, 140), title=pos_title,
                text=pos_text or "cinematic film still, rich detail, volumetric light, 35mm")
    neg = g.add("CLIPTextEncode", size=(380, 110), title="Negative",
                text=neg_text or "blurry, low quality, watermark, text, frame, border")
    g.link(ck, "CLIP", pos, "clip")
    g.link(ck, "CLIP", neg, "clip")
    return ck, pos, neg


def first_frame(g, ck, pos, neg, setup, steps=24, cfg=6.0, sampler="dpmpp_2m", scheduler="karras"):
    lat = g.add("EmptyLatentImage", size=(260, 110), width=1024, height=576, batch_size=1)
    g.link(setup, "width", lat, "width")
    g.link(setup, "height", lat, "height")
    ks = g.add("KSampler", size=(300, 260), title="First frame", seed=7, steps=steps,
               cfg=cfg, sampler_name=sampler, scheduler=scheduler, denoise=1.0)
    g.link(ck, "MODEL", ks, "model")
    g.link(pos, "CONDITIONING", ks, "positive")
    g.link(neg, "CONDITIONING", ks, "negative")
    g.link(lat, "LATENT", ks, "latent_image")
    dec = g.add("VAEDecode", size=(200, 60))
    g.link(ks, "LATENT", dec, "samples")
    g.link(ck, "VAE", dec, "vae")
    return dec


def direction_block(g, frames, seconds=5.0, target=None, long_edge=None, image_title="First frame",
                    **director):
    with g.block(B_DIRECT, C_DIRECT, col=1, row=0):
        kw = {"duration": seconds}
        if target:
            kw["target"] = target
        if long_edge:
            kw["long_edge"] = long_edge
        s = g.add("Difforum_Setup", size=(320, 300), **kw)
        img = g.add("LoadImage", size=(320, 360), image="example.png", title=image_title) if image_title else None
        d = g.add("Difforum_Director", size=(800, 880), timeline=director.pop("timeline", tl_json(frames)),
                  **director)
        g.link(s, "params", d, "params")
    return s, img, d


# ---------------------------------------------------------------------------
# 01 - 06: storyboard and feedback
# ---------------------------------------------------------------------------

def wf_storyboard():
    g = Graph("01 · Storyboard - direct a shot, no model needed")
    control(g, "## Start here - no model needed\n\n1. Load any still in **Load Image**.\n2. Edit the "
            "**Director** timeline: scenes, camera moves, keys; press ▶ to preview.\n3. Queue: the "
            "**Animatic** renders a previz video, the **Storyboard** a contact sheet, and **Camera → "
            "Prompt** writes the move in words for prompt-driven video models.\n\nThen move to 02 "
            "(feedback render) or 08 / 10 (MiniMax H3).")
    s, img, d = direction_block(g, 120, image_title="Image")
    previz(g, d, init=(img, "IMAGE"))
    with g.block("3 · Storyboard", C_RENDER, col=3):
        sb = g.add("Difforum_Storyboard", size=(320, 220))
        g.link(img, "IMAGE", sb, "init_image")
        g.link(d, "direction", sb, "direction")
        p1 = g.add("PreviewImage", size=(620, 380), title="Contact sheet")
        g.link(sb, "sheet", p1, "images")
        p2 = g.add("PreviewImage", size=(300, 300), title="Camera path (top view)")
        g.link(sb, "camera_path", p2, "images")
        cp = g.add("Difforum_CameraPrompt", size=(320, 160))
        g.link(d, "direction", cp, "direction")
        t = g.add("PreviewAny", size=(320, 200), title="Camera in words")
        g.link(cp, "text", t, "source")
    return g


def wf_feedback(depth=False, title="02 · Feedback render (SDXL)", camera_mode="2d", tl=None,
                duration=5.0, readme=""):
    g = Graph(title)
    control(g, readme)
    with g.block(B_DIRECT, C_DIRECT, col=1, row=0):
        s = g.add("Difforum_Setup", size=(320, 300), duration=duration, long_edge=1024)
        d = g.add("Difforum_Director", size=(800, 880), camera_mode=camera_mode, timeline=tl or tl_json(120))
        g.link(s, "params", d, "params")
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        ck, pos, neg = sd_models(g)
        g.link(ck, "CLIP", d, "clip")
    with g.block(B_FIRST, C_MODELS, col=1, row=1):
        dec = first_frame(g, ck, pos, neg, s)
    dp = depth_block(g, (dec, "IMAGE")) if depth else None
    previz(g, d, init=(dec, "IMAGE"), depth=dp)
    with g.block("3 · Render · Feedback Sampler", C_RENDER, col=3):
        fb = g.add("Difforum_FeedbackSampler", size=(360, 460), steps=20, cfg=6.0,
                   sampler_name="dpmpp_2m", scheduler="karras", cadence=2)
        for a, b in (("MODEL", "model"), ("VAE", "vae")):
            g.link(ck, a, fb, b)
        g.link(pos, "CONDITIONING", fb, "positive")
        g.link(neg, "CONDITIONING", fb, "negative")
        g.link(dec, "IMAGE", fb, "init_image")
        g.link(d, "direction", fb, "direction")
        if depth:
            g.link(dp[0], dp[1], fb, "depth")
        st = g.add("Difforum_FlowStabilize", size=(300, 130), strength=0.45)
        g.link(fb, "frames", st, "frames")
        rp = g.add("PreviewAny", size=(300, 200), title="Run report")
        g.link(fb, "report", rp, "source")
    up = upscale(g, (st, "frames"), col=4)
    sv = output(g, up, col=5, prefix="video/difforum", fps_src=(s, "fps"))
    return g, s, d, fb, sv


FEEDBACK_README = ("## Feedback render (SDXL)\n\nThe classic Deforum look on a modern model: the "
                   "**Director** drives camera, energy and prompt travel through one `direction` wire.\n\n"
                   "- Any SD1.5 / SDXL / Flux checkpoint (plain MODEL / VAE / CONDITIONING).\n"
                   "- `cadence 2` diffuses every other frame; steps scale with the energy.\n"
                   "- Faster: a DMD2 / Lightning / Turbo LoRA at 4-8 steps, cfg 1-2.\n"
                   "- **Workflow Switches** turn Previz, Render and Upscale on and off; ⌖ jumps to a group.")


def wf_feedback_2d():
    g, *_ = wf_feedback(readme=FEEDBACK_README)
    return g


def wf_parallax():
    tl = tl_json(120, camera=[
        {"start": 0, "move": "dolly_in", "speed": 0.9, "intensity": 0.8, "ease": "ease_in_out"},
        {"start": 40, "move": "orbit_right", "speed": 1.0, "intensity": 0.9, "ease": "ease_in_out"},
        {"start": 80, "move": "crane_up", "speed": 0.8, "intensity": 0.8, "ease": "ease_out"},
    ])
    g, *_ = wf_feedback(depth=True, title="03 · Real parallax (3D + depth)", camera_mode="3d", tl=tl,
                        readme="## Real parallax\n\nDepth Anything 3 (core) reads the first frame, the Director "
                        "is in **3d**, and the sampler re-projects the depth with the image every frame - "
                        "near things move faster than far ones for the whole clip.\n\nUses the core **Depth Anything 3** nodes. Too strong? Lower `translation_scale` on "
                        "Render Options. The sampler's `depth` output is the tracked depth per frame.")
    return g


def wf_audio():
    g, s, d, fb, sv = wf_feedback(title="04 · Audio reactive", duration=10.0, tl=tl_json(240, camera=[
        {"start": 0, "move": "zoom_in", "speed": 0.8, "intensity": 0.8, "ease": "ease_in_out",
         "react": "bass_speed"},
        {"start": 96, "move": "spiral", "speed": 1.0, "intensity": 1.0, "ease": "ease_in_out",
         "react": "beat_pulse"},
        {"start": 168, "move": "sway", "speed": 1.0, "intensity": 0.9, "ease": "ease_in_out",
         "react": "onset_shake"},
    ]), readme="## Audio reactive\n\nEach camera block on the Director picks an **Audio** reaction "
        "(beat pulse, onset shake, bass speed, mid sway): no expressions. The Audio Analyzer feeds the "
        "Director; Create Video puts the track back on the clip.\n\n`offset_seconds` on the analyzer "
        "syncs with your edit. The bands plot shows what the camera is listening to.")
    with g.block("Audio", C_MISC, col=0, row=2):
        au = g.add("LoadAudio", size=(320, 140), audio="track.mp3")
        an = g.add("Difforum_AudioAnalyzer", size=(320, 200))
        g.link(s, "params", an, "params")
        g.link(au, "AUDIO", an, "audio")
        g.link(an, "audio_curves", d, "audio")
        pv = g.add("PreviewImage", size=(320, 260), title="Bands: low / mid / high / beat")
        g.link(an, "bands_plot", pv, "images")
    cv = next(n for n in g.nodes if n["type"] == "CreateVideo" and any(
        link[3] == sv and link[1] == n["id"] for link in g.links))
    g.link(au, "AUDIO", cv["id"], "audio")
    return g


def wf_live():
    g = Graph("05 · Live (turbo models)")
    control(g, "## Live\n\nRealtime feedback with a turbo model (SDXL-Turbo, SD-Turbo, LCM, DMD2 at 1-4 "
            "steps). The Live Sampler plays inside the node, loops the camera, and can mirror a webcam "
            "(`live_source 0`) or a video, stream PNGs to a folder or Spout for Resolume / TouchDesigner / "
            "OBS.\n\nRender Options here sets a kaleidoscope symmetry in the loop.")
    with g.block(B_DIRECT, C_DIRECT, col=1, row=0):
        s = g.add("Difforum_Setup", size=(320, 300), duration=10.0, long_edge=512)
        d = g.add("Difforum_Director", size=(800, 880), timeline=tl_json(240))
        g.link(s, "params", d, "params")
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        ck, pos, neg = sd_models(g, ckpt="sd_xl_turbo_1.0_fp16.safetensors")
    with g.block(B_FIRST, C_MODELS, col=1, row=1):
        dec = first_frame(g, ck, pos, neg, s, steps=2, cfg=1.0, sampler="euler_ancestral",
                          scheduler="sgm_uniform")
    previz(g, d, init=(dec, "IMAGE"))
    with g.block("3 · Render · Live Sampler", C_RENDER, col=3):
        op = g.add("Difforum_RenderOptions", size=(340, 560), title="Render Options (kaleidoscope)",
                   symmetry="kaleidoscope", symmetry_segments=8)
        lv = g.add("Difforum_LiveSampler", size=(360, 560), steps=2, cfg=1.0,
                   sampler_name="euler_ancestral", scheduler="sgm_uniform")
        g.link(op, "options", lv, "options")
        for a, b in (("MODEL", "model"), ("VAE", "vae")):
            g.link(ck, a, lv, b)
        g.link(pos, "CONDITIONING", lv, "positive")
        g.link(neg, "CONDITIONING", lv, "negative")
        g.link(dec, "IMAGE", lv, "init_image")
        g.link(d, "direction", lv, "direction")
        pv = g.add("PreviewImage", size=(400, 300))
        g.link(lv, "frames", pv, "images")
        rp = g.add("PreviewAny", size=(300, 160), title="Report")
        g.link(lv, "report", rp, "source")
    return g


def wf_loop():
    g = Graph("06 · Seamless loop (installations)")
    control(g, "## Loop without a crossfade\n\nThe Camera path is made periodic over 120 frames and "
            "rendered for 3 laps; the feedback settles onto its cycle and **Loop** keeps the last lap. "
            "For projections that run for hours.\n\n**Speed:** cadence 2 and steps x energy (10 of 20 at "
            "0.5). Faster: a DMD2 / Lightning LoRA (steps 4-6, cfg 1-2) or `long_edge` 512. Low-VRAM "
            "launch flags (`--lowvram`, `--disable-smart-memory`) reload the model every frame.")
    with g.block(B_DIRECT, C_DIRECT, col=1, row=0):
        s = g.add("Difforum_Setup", size=(320, 300), duration_mode="frames", duration=360.0)
        cam = g.add("Difforum_Camera", size=(380, 360), keys="0: orbit_right 1.0 0.8\n60: spiral 1.2 1.0",
                    loop_mode="harmonic", cycle_frames=120)
        g.link(s, "params", cam, "params")
        sch = g.add("Difforum_Schedule", size=(380, 160), schedule="0:(0.5)")
        g.link(s, "params", sch, "params")
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        ck, pos, neg = sd_models(g)
    with g.block(B_FIRST, C_MODELS, col=1, row=1):
        dec = first_frame(g, ck, pos, neg, s)
    with g.block(B_PREVIZ, C_PREVIZ, col=2):
        sb = g.add("Difforum_Storyboard", size=(360, 220), title="Storyboard (previz the loop)")
        g.link(dec, "IMAGE", sb, "init_image")
        g.link(s, "params", sb, "params")
        g.link(cam, "camera", sb, "camera")
        sbp = g.add("PreviewImage", size=(360, 300), title="Contact sheet")
        g.link(sb, "sheet", sbp, "images")
    with g.block("3 · Render · Feedback loop", C_RENDER, col=3):
        fb = g.add("Difforum_FeedbackSampler", size=(360, 460), cadence=2)
        for a, b in (("MODEL", "model"), ("VAE", "vae")):
            g.link(ck, a, fb, b)
        g.link(pos, "CONDITIONING", fb, "positive")
        g.link(neg, "CONDITIONING", fb, "negative")
        g.link(dec, "IMAGE", fb, "init_image")
        g.link(s, "params", fb, "params")
        g.link(cam, "camera", fb, "camera")
        g.link(sch, "schedule", fb, "strength")
        lp = g.add("Difforum_Loop", size=(300, 160), method="keep settled lap")
        g.link(fb, "frames", lp, "frames")
        g.link(cam, "cycle_frames", lp, "cycle_frames")
    up = upscale(g, (lp, "frames"), col=4)
    output(g, up, col=5, prefix="video/difforum_loop")
    return g


# ---------------------------------------------------------------------------
# 07: LTX guides
# ---------------------------------------------------------------------------

def wf_ltx():
    g = Graph("07 · LTX-2 / 2.5 guides from the Director")
    control(g, "## Wire into your LTX graph\n\nConnect your LTX **positive / negative** (text encoded with "
            "the Camera → Prompt text appended) and the LTX **VAE** into **LTX Guides**. Its outputs "
            "replace the conditioning and latent going into your LTX sampler.\n\n- Setup target = LTX "
            "keeps length on 8k+1 and size on 32 px.\n- Guides every 1 s at strength 0.7; frame 0 at 1.0 "
            "locks the look.\n- Swap Guide Frames for a Feedback render or a Storyboard to guide with "
            "other images.")
    s, img, d = direction_block(g, 121, target="LTX-2 / 2.5", long_edge=1216, image_title="Anchor / first frame")
    previz(g, d, init=(img, "IMAGE"))
    with g.block("3 · LTX guides", C_RENDER, col=3):
        gf = g.add("Difforum_GuideFrames", size=(340, 200))
        g.link(img, "IMAGE", gf, "anchor_image")
        g.link(d, "direction", gf, "direction")
        kf = g.add("Difforum_Keyframes", size=(340, 220), every_seconds=1.0)
        g.link(gf, "guide_frames", kf, "frames")
        g.link(s, "fps", kf, "fps")
        lat = g.add("EmptyLTXVLatentVideo", size=(300, 160), width=1216, height=704, length=121, batch_size=1)
        g.link(s, "width", lat, "width")
        g.link(s, "height", lat, "height")
        g.link(kf, "length", lat, "length")
        lg = g.add("Difforum_LTXGuides", size=(340, 260))
        g.link(kf, "keyframes", lg, "keyframes")
        g.link(kf, "indices", lg, "indices")
        g.link(lat, "LATENT", lg, "latent")
        cp = g.add("Difforum_CameraPrompt", size=(340, 160), format="prompt suffix")
        g.link(d, "direction", cp, "direction")
        pv = g.add("PreviewImage", size=(340, 260), title="Keyframes")
        g.link(kf, "keyframes", pv, "images")
    return g


# ---------------------------------------------------------------------------
# 08, 10, 11: MiniMax H3
# ---------------------------------------------------------------------------

def h3_loaders(g, unet):
    un = g.add("UNETLoader", size=(420, 90), unet_name=unet, weight_dtype="default")
    cl = g.add("CLIPLoader", size=(420, 110), clip_name="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
               type="minimax", device="default")
    vv = g.add("VAELoader", size=(420, 60), title="Video VAE",
               vae_name="minimax_h3_video_vae_int8_convrot.safetensors")
    va = g.add("VAELoader", size=(420, 60), title="Audio VAE", vae_name="minimax_h3_audio_vae_fp32.safetensors")
    return un, cl, vv, va


def h3_render(g, conditioning, latent, un, vv, va, lora, col, steps=20):
    """Official MiniMax H3 sampling tail, with a switchable TAEH3 live preview."""
    with g.block("3 · Render · MiniMax H3", C_RENDER, col=col):
        lo = g.add("LoraLoaderModelOnly", size=(360, 90), title="Turbo LoRA (Ctrl+B to enable, then steps 4-8)",
                   mode=4, lora_name=lora, strength_model=1.0)
        g.link(un, "MODEL", lo, "model")
    with g.block(B_LIVE, C_LIVE, col=col, row=1):
        g.note("## Live preview while H3 samples\n\n**Model Preview Override** (KJNodes) shows the video "
               "forming at every step, decoded by **taeh3** (put `taeh3.safetensors` in `models/vae_approx`). "
               "`preview_frames` > 1 plays it as a clip. Switch the group off to skip it.", size=(360, 200))
        mp = g.add("ModelPreviewOverrideKJ", size=(380, 640), max_resolution=768, jpeg_quality=80,
                   suppress_default_preview=True, preview_frames=16, preview_fps=12,
                   tiny_vae="taeh3.safetensors")
        g.link(lo, "MODEL", mp, "model")
    with g.block("3 · Render · MiniMax H3", C_RENDER, col=col):
        gd = g.add("BasicGuider", size=(220, 50))
        g.link(mp, "MODEL", gd, "model")
        g.link(conditioning[0], conditioning[1], gd, "conditioning")
        nz = g.add("RandomNoise", size=(260, 90), noise_seed=7)
        ks = g.add("KSamplerSelect", size=(260, 60), sampler_name="res_multistep")
        sc = g.add("BasicScheduler", size=(260, 110), scheduler="simple", steps=steps, denoise=1.0)
        g.link(mp, "MODEL", sc, "model")
        sa = g.add("SamplerCustomAdvanced", size=(260, 120))
        g.link(nz, "NOISE", sa, "noise")
        g.link(gd, "GUIDER", sa, "guider")
        g.link(ks, "SAMPLER", sa, "sampler")
        g.link(sc, "SIGMAS", sa, "sigmas")
        g.link(latent[0], latent[1], sa, "latent_image")
    # two-stage: H3 renders small and fast, a learned latent upscaler lifts it to ~1 MP and H3
    # re-samples a few steps at full size - real, temporally coherent detail, no VAE round trip
    with g.block(B_H3UP, C_UP, col=col):
        g.note("## H3 Latent Upscale (two-stage)\n\nH3 renders at the Setup size (640 px, fast), the "
               "**Minimax H3 Latent Upscaler (3D)** lifts the latent to ~1 MP, and H3 re-samples it for a few "
               "steps at full size: new detail that holds over time, no decode / re-encode.\n\n"
               "- Needs [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler) "
               "and `minimax_h3_latent_upscaler_3d_fp16.safetensors` in `models/latent_upscale_models`.\n"
               "- Refine sigmas: 4 steps from 0.63 for the base model; with the Turbo LoRA on use "
               "`0.6316, 0.3158, 0.0000`. Start higher (0.8-0.9) for more new detail, lower (0.5) to keep "
               "the motion exactly.\n- Saves time, not VRAM: the refine runs at the 2x size. Switch off to decode "
               "the small render directly.", size=(380, 360))
        sep = g.add("LTXVSeparateAVLatent", size=(240, 70))
        g.link(sa, "output", sep, "av_latent")
        up3 = g.add("MinimaxH3LatentUpscaler3D", size=(340, 260), title="Minimax H3 Latent Upscaler (3D)",
                    model_name="minimax_h3_latent_upscaler_3d_fp16.safetensors", mode="scale by multiplier",
                    align=32, enable_temporal_chunking=False, force_unload=True,
                    device="cuda", precision="fp16")
        g.link(sep, "video_latent", up3, "latent")
        cat = g.add("LTXVConcatAVLatent", size=(240, 70))
        g.link(up3, "latent", cat, "video_latent")
        g.link(sep, "audio_latent", cat, "audio_latent")
        gd2 = g.add("BasicGuider", size=(220, 50), title="Refine guider")
        g.link(lo, "MODEL", gd2, "model")
        g.link(conditioning[0], conditioning[1], gd2, "conditioning")
        ks2 = g.add("KSamplerSelect", size=(260, 60), sampler_name="euler")
        sg2 = g.add("ManualSigmas", size=(300, 60), title="Refine sigmas", sigmas="0.6316, 0.4737, 0.3158, 0.1579, 0.0000")
        sa2 = g.add("SamplerCustomAdvanced", size=(260, 120), title="Refine at 2x")
        g.link(nz, "NOISE", sa2, "noise")
        g.link(gd2, "GUIDER", sa2, "guider")
        g.link(ks2, "SAMPLER", sa2, "sampler")
        g.link(sg2, "SIGMAS", sa2, "sigmas")
        g.link(cat, "latent", sa2, "latent_image")
    with g.block("3 · Render · MiniMax H3", C_RENDER, col=col):
        dv = g.add("VAEDecode", size=(200, 60))
        g.link(sa2, "output", dv, "samples")
        g.link(vv, "VAE", dv, "vae")
        da = g.add("VAEDecodeAudio", size=(200, 60))
        g.link(sa2, "output", da, "samples")
        g.link(va, "VAE", da, "vae")
    return (dv, "IMAGE"), (da, "AUDIO")


H3_TIMELINE = dict(scenes=[
    {"start": 0, "mood": "calm", "prompt": "misty ancient forest at dawn, light shafts through the canopy"},
    {"start": 62, "mood": "build", "prompt": "the mist lifts, glowing moss and roots in the foreground"},
], camera=[
    {"start": 0, "move": "dolly_in", "speed": 0.8, "intensity": 0.8, "ease": "ease_in_out"},
    {"start": 62, "move": "orbit_right", "speed": 0.9, "intensity": 0.8, "ease": "ease_in_out"},
])

SCENE = "misty ancient forest at dawn, light shafts through the canopy, mossy trees, volumetric light"


def fill_models(g, scene):
    ck = g.add("CheckpointLoaderSimple", size=(360, 100), title="Fill model (an inpaint model is best)",
               ckpt_name="sd_xl_base_1.0.safetensors")
    pos = g.add("CLIPTextEncode", size=(400, 120), title="Fill prompt (describe the scene)", text=scene)
    neg = g.add("CLIPTextEncode", size=(400, 90), title="Fill negative",
                text="border, frame, seam, blur, text, watermark")
    g.link(ck, "CLIP", pos, "clip")
    g.link(ck, "CLIP", neg, "clip")
    return ck, pos, neg


def fill_node(g, fm, frames, images_src, masks_src):
    ck, pos, neg = fm
    fr = g.add("Difforum_FillReveal", size=(360, 380), frames=frames, steps=24, cfg=5.0,
               sampler_name="dpmpp_2m", scheduler="karras")
    g.link(images_src[0], images_src[1], fr, "images")
    g.link(masks_src[0], masks_src[1], fr, "masks")
    g.link(ck, "MODEL", fr, "model")
    g.link(ck, "VAE", fr, "vae")
    g.link(pos, "CONDITIONING", fr, "positive")
    g.link(neg, "CONDITIONING", fr, "negative")
    return fr


def depth_block(g, image):
    """Core Depth Anything 3 on the first frame: real parallax for 3d camera moves."""
    with g.block(B_DEPTH, C_PREVIZ, col=0):
        dm = g.add("LoadDA3Model", size=(340, 90), model_name="depth_anything_3_mono_large.safetensors",
                   weight_dtype="default")
        di = g.add("DA3Inference", size=(320, 130), resolution=1008, resize_method="upper_bound_resize",
                   mode="mono")
        g.link(dm, "DA3_MODEL", di, "da3_model")
        g.link(image[0], image[1], di, "image")
        dr = g.add("DA3Render", size=(300, 110), output="depth")
        g.link(di, "da3_geometry", dr, "da3_geometry")
        pv = g.add("PreviewImage", size=(300, 220), title="Depth (white = near)")
        g.link(dr, "IMAGE", pv, "images")
    return dr, "IMAGE"


def polish_block(g, src, fm):
    """Re-paint keyframes at low denoise: they are warped (soft) and they set H3's look."""
    ck, pos, neg = fm
    with g.block(B_POLISH, C_FILL, col=0):
        g.note("## Keyframe Polish\n\nThe keyframes H3 follows come from warping one image along the camera, "
               "so they get softer the further the camera travels, and H3 copies that softness. This pass "
               "re-paints each keyframe with the image model at low denoise (`clean restyle`, 0.35): sharp "
               "detail back, same composition.\n\n- 0.25 = just sharpen, 0.45 = re-imagine textures.\n"
               "- The prompt is the Fill prompt: describe the scene and the look.", size=(360, 260))
        rs = g.add("Difforum_Restyle", size=(360, 520), title="Restyle (keyframe polish)", style="clean restyle",
                   denoise=0.35, steps=24, cfg=5.0, sampler_name="dpmpp_2m", scheduler="karras", cadence=1,
                   long_edge=0)
        g.link(src[0], src[1], rs, "video")
        g.link(ck, "MODEL", rs, "model")
        g.link(ck, "VAE", rs, "vae")
        g.link(pos, "CONDITIONING", rs, "positive")
        g.link(neg, "CONDITIONING", rs, "negative")
    return rs, "frames"


def restyle_block(g, video, ck, pos, neg, d, col, style="deforum morph"):
    with g.block(B_RESTYLE, C_STYLE, col=col, row=0):
        g.note("## Restyle: the look, on H3's motion\n\nAn image model re-paints every H3 frame in the "
               "chosen look, with the previous painted frame carried along the video's motion and mixed "
               "in - the feedback that made Deforum morph and AnimateDiff boil.\n\n- `deforum morph` smear, "
               "`animatediff boil` re-rolling texture, `disco flicker`, `clean restyle`.\n- `denoise` 0.3 "
               "keeps H3's image, 0.6 re-imagines it. Switch off for the clean H3 render.",
               size=(380, 300))
        rs = g.add("Difforum_Restyle", size=(360, 520), style=style, denoise=0.45, steps=6, cfg=1.5,
                   sampler_name="euler_ancestral", scheduler="sgm_uniform", cadence=1, long_edge=1024)
        g.link(video[0], video[1], rs, "video")
        g.link(ck, "MODEL", rs, "model")
        g.link(ck, "VAE", rs, "vae")
        g.link(pos, "CONDITIONING", rs, "positive")
        g.link(neg, "CONDITIONING", rs, "negative")
        g.link(d, "direction", rs, "direction")
        rp = g.add("PreviewAny", size=(300, 160), title="Restyle report")
        g.link(rs, "report", rp, "source")
    return rs, "frames"


def wf_h3():
    g = Graph("08 · MiniMax H3 first/last frame (FL2VA) from the Director")
    control(g, "## MiniMax H3 · first / last frame\n\nThe **Director** draws the move, **Guide Frames** "
            "carries your first frame to where the camera ends, and **Fill Reveal (AI)** paints what the "
            "camera uncovers. **H3 Shot** hands H3 the first frame, that last frame, the length (17k+5), the "
            "size and a prompt with the camera move and the Director's **look**.\n\n- **Switches**: Previz, "
            "Fill Reveal, Live Preview (TAEH3), Render, Upscale 2K.\n- Fill model: an inpaint checkpoint "
            "gives the cleanest seams.\n- Turbo LoRA: Ctrl+B, steps 6-8. Live preview needs KJNodes.")
    s, img, d = direction_block(g, 124, target="MiniMax H3", long_edge=640, camera_mode="3d",
                                timeline=tl_json(124, **H3_TIMELINE))
    dp = depth_block(g, (img, "IMAGE"))
    previz(g, d, init=(img, "IMAGE"), depth=dp)
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        un, cl, vv, va = h3_loaders(g, "minimax_h3_fl2va_pruned_int8_convrot.safetensors")
        fm = fill_models(g, SCENE)
    with g.block("3 · H3 shot", C_RENDER, col=3):
        gf = g.add("Difforum_GuideFrames", size=(340, 200), hole_fill="gray")
        g.link(img, "IMAGE", gf, "anchor_image")
        g.link(d, "direction", gf, "direction")
        g.link(dp[0], dp[1], gf, "depth")
        h3 = g.add("Difforum_H3Shot", size=(340, 320), segment_length="124 (~5s)",
                   shot_description="A misty ancient forest at dawn, light shafts cutting through the canopy.",
                   soundscape="Soft birdsong and a low wind moving through the trees.")
        g.link(gf, "guide_frames", h3, "frames")
        g.link(gf, "masks", h3, "masks")
        g.link(d, "direction", h3, "direction")
    with g.block(B_FILL, C_FILL, col=4):
        fr = fill_node(g, fm, "all", (h3, "last_frame"), (h3, "last_mask"))
        p1 = g.add("PreviewImage", size=(360, 260), title="Last frame, AI-filled")
        g.link(fr, "images", p1, "images")
    pl = polish_block(g, (fr, "images"), fm)
    with g.block("3 · H3 shot", C_RENDER, col=3):
        i2v = g.add("MiniMaxH3ImageToVideo", size=(420, 260), prompt="", width=832, height=480, length=124)
        g.link(cl, "CLIP", i2v, "clip")
        g.link(vv, "VAE", i2v, "vae")
        g.link(h3, "first_frame", i2v, "first_frame")
        g.link(pl[0], pl[1], i2v, "last_frame")
        for k in ("prompt", "width", "height", "length"):
            g.link(h3, k, i2v, k)
    video, audio = h3_render(g, (i2v, "positive"), (i2v, "LATENT"), un, vv, va,
                             "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors", col=5)
    up = upscale(g, video, col=6)
    output(g, up, col=7, prefix="video/difforum_h3", audio=audio)
    return g


def h3_guides(g, s, d, kf_source, kf_masks, anchor, cl, vv, va, fm=None, max_guides=4, cp_prefix="",
              col=3, soundscape=""):
    """Keyframes -> (Fill Reveal) -> H3 Guides on the official ref2va conditioning."""
    with g.block("3 · H3 guides", C_RENDER, col=col):
        kf = g.add("Difforum_Keyframes", size=(340, 240), grid="MiniMax H3 (17k+5)",
                   every_seconds=2.0 if max_guides <= 4 else 1.0)
        g.link(kf_source[0], kf_source[1], kf, "frames")
        if kf_masks:
            g.link(kf_masks[0], kf_masks[1], kf, "masks")
        g.link(s, "fps", kf, "fps")
        cp = g.add("Difforum_CameraPrompt", size=(340, 320), format="H3 structured", prefix=cp_prefix,
                   h3_mode="reference (ref2va / guides)", soundscape=soundscape)
        g.link(d, "direction", cp, "direction")
    keys = (kf, "keyframes")
    if fm is not None:
        with g.block(B_FILL, C_FILL, col=col + 1):
            fr = fill_node(g, fm, "all", (kf, "keyframes"), (kf, "key_masks"))
            keys = (fr, "images")
        keys = polish_block(g, keys, fm)
    with g.block("3 · H3 guides", C_RENDER, col=col):
        r2v = g.add("MiniMaxH3ReferenceToVideo", size=(420, 300), prompt="", width=832, height=480,
                    length=124, ref_image_size="match")
        g.link(cl, "CLIP", r2v, "clip")
        g.link(vv, "VAE", r2v, "vae")
        g.link(va, "VAE", r2v, "audio_vae")
        g.link(anchor[0], anchor[1], r2v, "ref_images.ref_image_0")
        g.link(cp, "text", r2v, "prompt")
        g.link(s, "width", r2v, "width")
        g.link(s, "height", r2v, "height")
        g.link(kf, "length", r2v, "length")
        hg = g.add("Difforum_H3Guides", size=(420, 220), max_guides=max_guides)
        g.link(r2v, "positive", hg, "positive")
        g.link(r2v, "LATENT", hg, "latent")
        g.link(vv, "VAE", hg, "vae")
        g.link(keys[0], keys[1], hg, "keyframes")
        g.link(kf, "indices", hg, "indices")
        pv = g.add("PreviewImage", size=(420, 280), title="Keyframes sent to H3")
        g.link(keys[0], keys[1], pv, "images")
    return (hg, "positive"), (r2v, "LATENT")


def wf_h3_guides():
    g = Graph("10 · MiniMax H3 multi-keyframe guides from the Director")
    control(g, "## MiniMax H3 · keyframes along your camera\n\n**Guide Frames** carries the anchor along "
            "the Director's move, **Keyframes** picks one every 2 s on the H3 grid, **Fill Reveal (AI)** "
            "paints what the camera uncovers, and **H3 Guides** anchors them inside the generation (core "
            "`MiniMaxH3AddGuide`).\n\n- The anchor is also `<Picture 1>`; **Camera → Prompt** adds the move "
            "and the **look**.\n- **Switches**: Previz, Fill Reveal, Live Preview (TAEH3), Render, Upscale "
            "2K.\n- Turbo LoRA: Ctrl+B, steps 4. Live preview needs KJNodes.")
    s, img, d = direction_block(g, 124, target="MiniMax H3", long_edge=640, camera_mode="3d",
                                timeline=tl_json(124, **H3_TIMELINE),
                                image_title="Anchor image (also the H3 reference)")
    dp = depth_block(g, (img, "IMAGE"))
    previz(g, d, init=(img, "IMAGE"), depth=dp)
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        un, cl, vv, va = h3_loaders(g, "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
        fm = fill_models(g, SCENE)
    with g.block("3 · H3 guides", C_RENDER, col=3):
        gf = g.add("Difforum_GuideFrames", size=(340, 200), hole_fill="gray")
        g.link(img, "IMAGE", gf, "anchor_image")
        g.link(d, "direction", gf, "direction")
        g.link(dp[0], dp[1], gf, "depth")
    cond, lat = h3_guides(g, s, d, (gf, "guide_frames"), (gf, "masks"), (img, "IMAGE"), cl, vv, va, fm, 4,
                          "a misty ancient forest at dawn, light shafts cutting through the canopy",
                          soundscape="Soft birdsong and a low wind moving through the trees.")
    video, audio = h3_render(g, cond, lat, un, vv, va,
                             "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", col=5)
    up = upscale(g, video, col=6)
    output(g, up, col=7, prefix="video/difforum_h3", audio=audio)
    return g


def wf_h3_deforum():
    g = Graph("11 · Deforum / AnimateDiff look on MiniMax H3")
    control(g, "## Deforum / AnimateDiff / Disco look, H3 motion\n\nThree stages, each on a switch:\n\n"
            "1. **Look pass** - a turbo **Feedback Sampler** renders the shot the Deforum way; its frames "
            "guide H3 (**H3 Guides**, one per second), so the morphs and dissolves happen in H3.\n"
            "2. **Restyle** - the same image model re-paints every H3 frame in the look, carried along H3's "
            "motion (`deforum morph`, `animatediff boil`, `disco flicker`).\n"
            "3. **Look Mix** - optional grain / flicker cuts from the look pass.\n\nThen **Upscale 2K**. "
            "Change the Director's `look` to steer all of it.", size=(480, 400))
    tl = tl_json(124, scenes=[
        {"start": 0, "mood": "dream", "prompt": "misty ancient forest at dawn, painterly, volumetric light"},
        {"start": 40, "mood": "build", "prompt": "the forest dissolves into glowing bioluminescent coral"},
        {"start": 82, "mood": "climax", "prompt": "a vast nebula of light, cosmic, swirling colours"},
    ], camera=[
        {"start": 0, "move": "zoom_in", "speed": 1.0, "intensity": 1.0, "ease": "ease_in_out"},
        {"start": 62, "move": "spiral", "speed": 1.2, "intensity": 1.0, "ease": "ease_in"},
    ])
    s, img, d = direction_block(g, 124, target="MiniMax H3", long_edge=640, camera_mode="2d",
                                look="deforum_morph", timeline=tl)
    previz(g, d, init=(img, "IMAGE"))
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        ck, pos, neg = sd_models(g, ckpt="sd_xl_turbo_1.0_fp16.safetensors", title="Look model (SDXL-Turbo / DMD2)",
                                 pos_text="painterly, dreamy, rich detail, glowing light",
                                 neg_text="blurry, text, watermark")
        g.link(ck, "CLIP", d, "clip")
        un, cl, vv, va = h3_loaders(g, "minimax_h3_ref2va_pruned_int8_convrot.safetensors")
    with g.block(B_LOOKPASS, C_STYLE, col=3):
        fb = g.add("Difforum_FeedbackSampler", size=(360, 460), steps=4, cfg=1.0,
                   sampler_name="euler_ancestral", scheduler="sgm_uniform", cadence=2,
                   title="Feedback Sampler (look pass)")
        for a_, b_ in (("MODEL", "model"), ("VAE", "vae")):
            g.link(ck, a_, fb, b_)
        g.link(pos, "CONDITIONING", fb, "positive")
        g.link(neg, "CONDITIONING", fb, "negative")
        g.link(img, "IMAGE", fb, "init_image")
        g.link(d, "direction", fb, "direction")
    cond, lat = h3_guides(g, s, d, (fb, "frames"), None, (img, "IMAGE"), cl, vv, va, None, 6,
                          "the opening image of the piece", col=4,
                          soundscape="A low airy hum with faint crackles of static.")
    video, audio = h3_render(g, cond, lat, un, vv, va,
                             "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", col=5)
    rs = restyle_block(g, video, ck, pos, neg, d, col=6)
    with g.block(B_LOOKMIX, C_STYLE, col=6, row=1):
        lm = g.add("Difforum_LookMix", size=(300, 300), blend="detail transfer", amount=0.5)
        g.link(rs[0], rs[1], lm, "video")
        g.link(fb, "frames", lm, "look_pass")
    up = upscale(g, (lm, "frames"), col=7)
    output(g, up, col=8, prefix="video/difforum_h3_look", audio=audio)
    return g


def wf_restyle_video():
    g = Graph("13 · Restyle any video (Deforum / AnimateDiff look)")
    control(g, "## Restyle any video\n\nLoad a clip you already rendered (MiniMax H3, LTX, Seedance, live "
            "action) and give it the Deforum / AnimateDiff / Disco look: an image model re-paints every "
            "frame, with the previous painted frame carried along the clip's own motion.\n\n- `style`: "
            "deforum morph, animatediff boil, disco flicker, clean restyle, custom.\n- With **Structure** on "
            "(depth ControlNet) `denoise` 0.6-0.8 re-imagines the texture and keeps the shapes; without it stay "
            "under 0.5.\n- For the real AnimateDiff look (motion module), use template 14.\n- The audio and "
            "fps of the source are kept. **Upscale 2K** on a switch.", size=(480, 360))
    with g.block("1 · Source", C_DIRECT, col=1):
        lv = g.add("LoadVideo", size=(360, 420), file="input.mp4", upload="image")
        gc = g.add("GetVideoComponents", size=(260, 120))
        g.link(lv, "VIDEO", gc, "video")
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        ck, pos, neg = sd_models(g, ckpt="sd_xl_turbo_1.0_fp16.safetensors", title="Look model (SDXL-Turbo / DMD2)",
                                 pos_text="oil painting, thick impasto brush strokes, vivid colour, "
                                          "painterly, dreamy",
                                 neg_text="blurry, text, watermark, photo")
    with g.block(B_STRUCT, C_FILL, col=2):
        g.note("## Structure\n\nDepth Anything 3 reads every frame of the clip and a ControlNet holds that "
               "structure while Restyle re-paints, so `denoise` can go to 0.6-0.8 - the range where the "
               "Deforum / AnimateDiff look actually appears - without losing the shapes.\n\n- SDXL: a union "
               "ControlNet (e.g. xinsir promax) set to depth; SD1.5: control_v11f1p_sd15_depth.\n"
               "- Switch off for a pure feedback restyle (keep denoise under 0.5 then).", size=(360, 260))
        dm = g.add("LoadDA3Model", size=(340, 90), model_name="depth_anything_3_mono_large.safetensors",
                   weight_dtype="default")
        di = g.add("DA3Inference", size=(320, 130), resolution=504, resize_method="upper_bound_resize", mode="mono")
        g.link(dm, "DA3_MODEL", di, "da3_model")
        g.link(gc, "images", di, "image")
        dr = g.add("DA3Render", size=(300, 110), output="depth")
        g.link(di, "da3_geometry", dr, "da3_geometry")
        cn = g.add("ControlNetLoader", size=(360, 80), control_net_name="xinsir-controlnet-union-sdxl-1.0-promax.safetensors")
        cu = g.add("SetUnionControlNetType", size=(300, 80), type="depth")
        g.link(cn, "CONTROL_NET", cu, "control_net")
    with g.block(B_RESTYLE, C_STYLE, col=2):
        rs = g.add("Difforum_Restyle", size=(360, 600), style="deforum morph", denoise=0.65, steps=6, cfg=1.5,
                   sampler_name="euler_ancestral", scheduler="sgm_uniform", cadence=1, long_edge=1024,
                   control_strength=0.6)
        g.link(gc, "images", rs, "video")
        g.link(cu, "CONTROL_NET", rs, "control_net")
        g.link(dr, "IMAGE", rs, "control_image")
        g.link(ck, "MODEL", rs, "model")
        g.link(ck, "VAE", rs, "vae")
        g.link(pos, "CONDITIONING", rs, "positive")
        g.link(neg, "CONDITIONING", rs, "negative")
        rp = g.add("PreviewAny", size=(300, 160), title="Restyle report")
        g.link(rs, "report", rp, "source")
    up = upscale(g, (rs, "frames"), col=3)
    output(g, up, col=4, prefix="video/difforum_restyle", audio=(gc, "audio"), fps_src=(gc, "fps"))
    return g


def wf_animatediff_video():
    g = Graph("14 · AnimateDiff look on any video")
    control(g, "## AnimateDiff on any video\n\nThe real AnimateDiff look over a clip you already have (MiniMax "
            "H3, LTX, live action): an SD1.5 model with an AnimateDiff motion module re-draws the clip 16 frames "
            "at a time, while depth and edge ControlNets taken from the clip keep its motion and composition.\n\n"
            "- Needs [AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved), an SD1.5 "
            "checkpoint, a motion module (`v3_sd15_mm.ckpt`, or AnimateLCM for speed) and SD1.5 depth / canny "
            "ControlNets.\n- `denoise` 0.55 keeps the clip, 0.75 is the classic AnimateDiff re-draw.\n"
            "- Switch **AnimateDiff** off for per-frame img2img: the flickering Deforum / Disco look.\n"
            "- SD1.5 works at ~0.3 MP; **Upscale 2K** brings it to delivery size.", size=(500, 420))
    with g.block("1 · Source", C_DIRECT, col=1):
        lv = g.add("LoadVideo", size=(360, 420), file="input.mp4", upload="image")
        gc = g.add("GetVideoComponents", size=(260, 120))
        g.link(lv, "VIDEO", gc, "video")
        sc = g.add("ImageScaleToTotalPixels", size=(320, 110), title="Working size (SD1.5)",
                   upscale_method="lanczos", megapixels=0.3, resolution_steps=8)
        g.link(gc, "images", sc, "image")
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        ck, pos, neg = sd_models(g, ckpt="dreamshaper_8.safetensors", title="SD1.5 checkpoint",
                                 pos_text="masterpiece, painterly animation, thick brush strokes, glowing colours, "
                                          "dreamlike, intricate detail",
                                 neg_text="photo, blurry, low quality, watermark, text, deformed")
    with g.block(B_STRUCT, C_FILL, col=2):
        dm = g.add("LoadDA3Model", size=(340, 90), model_name="depth_anything_3_mono_large.safetensors",
                   weight_dtype="default")
        di = g.add("DA3Inference", size=(320, 130), resolution=504, resize_method="upper_bound_resize", mode="mono")
        g.link(dm, "DA3_MODEL", di, "da3_model")
        g.link(sc, "IMAGE", di, "image")
        dr = g.add("DA3Render", size=(300, 110), output="depth")
        g.link(di, "da3_geometry", dr, "da3_geometry")
        ed = g.add("Canny", size=(300, 100), low_threshold=0.2, high_threshold=0.5)
        g.link(sc, "IMAGE", ed, "image")
        cd = g.add("ControlNetLoader", size=(340, 80), title="Depth ControlNet",
                   control_net_name="control_v11f1p_sd15_depth.pth")
        ce = g.add("ControlNetLoader", size=(340, 80), title="Edge ControlNet",
                   control_net_name="control_v11p_sd15_canny.pth")
        a1 = g.add("ControlNetApplyAdvanced", size=(320, 170), title="Apply depth", strength=0.65,
                   start_percent=0.0, end_percent=1.0)
        g.link(pos, "CONDITIONING", a1, "positive")
        g.link(neg, "CONDITIONING", a1, "negative")
        g.link(cd, "CONTROL_NET", a1, "control_net")
        g.link(dr, "IMAGE", a1, "image")
        a2 = g.add("ControlNetApplyAdvanced", size=(320, 170), title="Apply edges", strength=0.35,
                   start_percent=0.0, end_percent=0.7)
        g.link(a1, "positive", a2, "positive")
        g.link(a1, "negative", a2, "negative")
        g.link(ce, "CONTROL_NET", a2, "control_net")
        g.link(ed, "IMAGE", a2, "image")
    with g.block(B_AD, C_STYLE, col=3):
        cx = g.add("ADE_StandardUniformContextOptions", size=(340, 220), context_length=16, context_stride=1,
                   context_overlap=4, fuse_method="pyramid", use_on_equal_length=False, start_percent=0.0,
                   guarantee_steps=1)
        ad = g.add("ADE_AnimateDiffLoaderGen1", size=(340, 170), model_name="v3_sd15_mm.ckpt",
                   beta_schedule="autoselect")
        g.link(ck, "MODEL", ad, "model")
        g.link(cx, "CONTEXT_OPTS", ad, "context_options")
    with g.block("3 · Render · AnimateDiff", C_RENDER, col=4):
        en = g.add("VAEEncode", size=(200, 60))
        g.link(sc, "IMAGE", en, "pixels")
        g.link(ck, "VAE", en, "vae")
        ks = g.add("KSampler", size=(300, 260), seed=7, steps=20, cfg=7.0, sampler_name="euler_ancestral",
                   scheduler="normal", denoise=0.7)
        g.link(ad, "MODEL", ks, "model")
        g.link(a2, "positive", ks, "positive")
        g.link(a2, "negative", ks, "negative")
        g.link(en, "LATENT", ks, "latent_image")
        dv = g.add("VAEDecode", size=(200, 60))
        g.link(ks, "LATENT", dv, "samples")
        g.link(ck, "VAE", dv, "vae")
    up = upscale(g, (dv, "IMAGE"), col=5)
    output(g, up, col=6, prefix="video/difforum_animatediff", audio=(gc, "audio"), fps_src=(gc, "fps"))
    return g


# ---------------------------------------------------------------------------
# 09, 12
# ---------------------------------------------------------------------------

def wf_export():
    g = Graph("09 · Camera to After Effects / Blender (and back)")
    control(g, "## Camera interchange\n\n**Camera Export** writes to `output/difforum/`: a `.jsx` for After "
            "Effects (File > Scripts > Run Script File), a `.py` for Blender (Text Editor > Run) and a "
            "`.json`.\n\n**Camera Import** reads a `.json` from `input/` - from Camera Export or the helper "
            "scripts in `tools/` - so a move blocked in Blender or AE drives the AI shot. Switch **Import** "
            "off until a file is in `input/`.")
    s, img, d = direction_block(g, 120, camera_mode="3d", image_title="Image")
    previz(g, d, init=(img, "IMAGE"))
    with g.block("3 · Export", C_MISC, col=3):
        ex = g.add("Difforum_CameraExport", size=(360, 200))
        g.link(d, "direction", ex, "direction")
        t = g.add("PreviewAny", size=(360, 160), title="Written files")
        g.link(ex, "paths", t, "source")
    with g.block("4 · Import", C_MISC, col=3, row=1):
        im = g.add("Difforum_CameraImport", size=(360, 200))
        g.link(s, "params", im, "params")
        sb = g.add("Difforum_Storyboard", size=(320, 220))
        g.link(img, "IMAGE", sb, "init_image")
        g.link(s, "params", sb, "params")
        g.link(im, "camera", sb, "camera")
        pv = g.add("PreviewImage", size=(520, 320))
        g.link(sb, "sheet", pv, "images")
    return g


def wf_long_shot():
    g = Graph("12 · Long shot with key moments (installations)")
    fps, secs_ = 24, 30
    n = fps * secs_
    tl = tl_json(n, scenes=[
        {"start": 0, "mood": "calm", "prompt": "an empty white gallery, soft daylight, concrete floor"},
        {"start": 6 * fps, "mood": "dream", "prompt": "the walls breathe, ink blooms across the plaster"},
        {"start": 12 * fps, "mood": "build", "prompt": "a forest grows out of the ink, roots on the floor"},
        {"start": 19 * fps, "mood": "climax", "prompt": "the forest burns into light, a nebula fills the room"},
        {"start": 25 * fps, "mood": "resolve", "prompt": "stars settle into an empty white gallery"},
    ], camera=[
        {"start": 0, "move": "zoom_in", "speed": 0.5, "intensity": 0.6, "ease": "ease_in_out"},
        {"start": 6 * fps, "move": "drift", "speed": 0.8, "intensity": 0.8, "ease": "ease_in_out"},
        {"start": 12 * fps, "move": "pan_right", "speed": 0.9, "intensity": 0.8, "ease": "ease_in_out"},
        {"start": 19 * fps, "move": "vortex", "speed": 1.4, "intensity": 1.1, "ease": "ease_in"},
        {"start": 25 * fps, "move": "zoom_out", "speed": 0.5, "intensity": 0.6, "ease": "ease_out"},
    ], keys=[
        {"start": 0, "label": "opening image"},
        {"start": 12 * fps, "label": "the forest"},
        {"start": 25 * fps, "label": "back to the room"},
    ])
    control(g, "## Long shot, key moments\n\nA 30 s shot built for installations: five scenes, five camera "
            "moves and three **Keys** (the pink diamonds on the Director). **Keyframe Images** pins one "
            "picture to each key, and the **Feedback Sampler** travels *through* them, landing on each "
            "picture on cue (`key_pull`, `key_approach`).\n\n1. Switch **Render** off and Queue: the "
            "Animatic shows the 30 s in seconds.\n2. Mouse wheel on the timeline zooms, shift+wheel scrolls; "
            "**Fit** shows it all.\n3. Switch **Render** on. For H3 / LTX, feed Keyframe Images into their "
            "Guides nodes instead.\n\nLook: `disco_diffusion`. Try `flicker_experimental`, `vqgan_clip`.",
            size=(480, 420))
    with g.block(B_DIRECT, C_DIRECT, col=1, row=0):
        s = g.add("Difforum_Setup", size=(320, 300), duration=float(secs_), long_edge=768)
        d = g.add("Difforum_Director", size=(800, 880), camera_mode="2d", look="disco_diffusion", timeline=tl)
        g.link(s, "params", d, "params")
    with g.block("1 · Key images", C_DIRECT, col=1, row=1):
        imgs = [g.add("LoadImage", size=(320, 340), image="example.png", title=t)
                for t in ("Key 1 · opening", "Key 2 · the forest", "Key 3 · back to the room")]
        bt = g.add("BatchImagesNode", size=(260, 120), title="Batch Images (one per key)")
        for i, im in enumerate(imgs):
            g.link(im, "IMAGE", bt, f"images.image{i}")
        ki = g.add("Difforum_KeyframeImages", size=(340, 160))
        g.link(bt, "IMAGE", ki, "images")
        g.link(d, "direction", ki, "direction")
        kinfo = g.add("PreviewAny", size=(340, 140), title="Where the keys land")
        g.link(ki, "info", kinfo, "source")
    with g.block(B_MODELS, C_MODELS, col=0, row=1):
        ck, pos, neg = sd_models(g, ckpt="sd_xl_turbo_1.0_fp16.safetensors",
                                 pos_text="painterly, dreamy, rich detail, glowing light")
        g.link(ck, "CLIP", d, "clip")
    previz(g, d, init=(imgs[0], "IMAGE"), keys=(ki,))
    with g.block("3 · Render · Feedback Sampler", C_RENDER, col=3):
        fb = g.add("Difforum_FeedbackSampler", size=(360, 520), steps=4, cfg=1.0,
                   sampler_name="euler_ancestral", scheduler="sgm_uniform", cadence=2, key_pull=0.65,
                   key_approach=24)
        for a_, b_ in (("MODEL", "model"), ("VAE", "vae")):
            g.link(ck, a_, fb, b_)
        g.link(pos, "CONDITIONING", fb, "positive")
        g.link(neg, "CONDITIONING", fb, "negative")
        g.link(imgs[0], "IMAGE", fb, "init_image")
        g.link(d, "direction", fb, "direction")
        g.link(ki, "keyframes", fb, "key_images")
        g.link(ki, "indices", fb, "key_indices")
        st = g.add("Difforum_FlowStabilize", size=(300, 130), strength=0.35)
        g.link(fb, "frames", st, "frames")
        rp = g.add("PreviewAny", size=(300, 200), title="Run report")
        g.link(fb, "report", rp, "source")
    up = upscale(g, (st, "frames"), col=4)
    output(g, up, col=5, prefix="video/difforum_long", fps_src=(s, "fps"))
    return g


WORKFLOWS = {
    "01_storyboard_no_model.json": wf_storyboard,
    "02_feedback_sdxl.json": wf_feedback_2d,
    "03_parallax_3d_depth.json": wf_parallax,
    "04_audio_reactive.json": wf_audio,
    "05_live_turbo.json": wf_live,
    "06_seamless_loop.json": wf_loop,
    "07_ltx_guides.json": wf_ltx,
    "08_h3_first_last_frame.json": wf_h3,
    "09_camera_to_ae_blender.json": wf_export,
    "10_h3_multikeyframe_guides.json": wf_h3_guides,
    "11_h3_deforum_look.json": wf_h3_deforum,
    "12_long_shot_keys.json": wf_long_shot,
    "13_restyle_any_video.json": wf_restyle_video,
    "14_animatediff_on_video.json": wf_animatediff_video,
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
