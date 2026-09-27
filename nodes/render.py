"""Render nodes: the Feedback Sampler (Deforum look, any SD/SDXL/Flux model)
and the Live Sampler (realtime loop with live preview and VJ outputs)."""

from __future__ import annotations

import time

import torch

from ..core.color import COLOR_MODES
from ..core.engine import (
    ANCHOR_MODES, BORDER_MODES, DEPTH_TRACKING, EngineConfig, FeedbackEngine,
    iter_feedback, resize_bhwc,
)
from ..core.schedule import Schedule
from ..core.symmetry import SYMMETRY_MODES
from ._common import (
    CAMERA, CAT_RENDER, PARAMS, PROMPT, SCHEDULE, check_interrupt, progress_bar,
)
from .direction import DIRECTION

OPTIONS = "DIFFORUM_OPTIONS"


def _samplers():
    try:
        import comfy.samplers
        return comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS
    except Exception:  # outside ComfyUI
        return ["euler", "lcm"], ["normal", "sgm_uniform"]


def _resolve(direction, params, camera, strength, cfg_curve, prompts):
    """Explicit sockets win over the direction wire."""
    if direction is not None:
        params = params or direction.params
        camera = camera or direction.camera
        strength = strength or direction.strength
        cfg_curve = cfg_curve or direction.cfg
        prompts = prompts if prompts is not None else direction.prompts
    if params is None or camera is None:
        raise ValueError("Connect a Director `direction` wire, or both `params` and `camera`.")
    return params, camera, strength, cfg_curve, prompts


def make_diffuser(model, vae, positive, negative, steps, cfg, sampler_name, scheduler,
                  strength_at, cfg_at, prompts, seed, seed_mode,
                  control_net=None, control_image=None, control_strength=0.6):
    """The img2img step the engine calls on key frames."""
    from nodes import common_ksampler

    cn_apply = None
    if control_net is not None and control_strength > 0.0:
        from nodes import ControlNetApplyAdvanced
        cn_apply = ControlNetApplyAdvanced().apply_controlnet

    def diffuse(img: torch.Tensor, f: int) -> torch.Tensor:
        check_interrupt()
        denoise = max(0.0, min(1.0, float(strength_at(f))))
        pos = prompts[f] if prompts is not None and len(prompts) else positive
        neg = negative
        if cn_apply is not None:
            if control_image is not None:
                hint = control_image[f % control_image.shape[0]].unsqueeze(0)
            else:
                hint = img[..., :3]
            pos, neg = cn_apply(pos, neg, control_net, hint, float(control_strength),
                                0.0, 1.0, vae=vae)
        latent = {"samples": vae.encode(img[..., :3])}
        seed_f = int(seed) if seed_mode == "fixed" else int(seed) + f
        out = common_ksampler(model, seed_f, int(steps), float(cfg_at(f)), sampler_name,
                              scheduler, pos, neg, latent, denoise=denoise)[0]
        image = vae.decode(out["samples"])
        if image.dim() == 5:                 # video VAEs return [B,T,H,W,C]
            image = image.reshape(-1, *image.shape[-3:])
        return image[:1]

    return diffuse


def _options(direction, options) -> dict:
    """Engine settings: a Render Options node wins; otherwise the Director's
    style sets the look; otherwise the defaults."""
    opts = {k: spec.get("default") for k, (_t, spec) in _COMMON_OPTIONAL.items()}
    if options is not None:
        opts.update(options)
    elif direction is not None:
        opts.update(direction.look)
    return opts


_COMMON_OPTIONAL = {
    "color_coherence": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                        "tooltip": "How hard colours are held to the anchor."}),
    "color_mode": (list(COLOR_MODES), {"default": "lab"}),
    "anchor_mode": (list(ANCHOR_MODES), {"default": "scene",
                    "tooltip": "scene = re-anchor colour at each prompt scene; first = hold "
                               "frame 0; rolling = previous key; none = free."}),
    "sharpen": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
    "noise": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 0.5, "step": 0.005}),
    "hole_noise": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05,
                   "tooltip": "Extra noise where the camera reveals new area, so it is repainted."}),
    "symmetry": (list(SYMMETRY_MODES), {"default": "none"}),
    "symmetry_segments": ("INT", {"default": 6, "min": 2, "max": 64}),
    "border": (list(BORDER_MODES), {"default": "reflection"}),
    "depth_tracking": (list(DEPTH_TRACKING), {"default": "follow",
                       "tooltip": "follow = depth is re-projected with the image every frame."}),
    "translation_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05,
                          "tooltip": "3D motion strength vs. the depth map's range."}),
    "near": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1000.0}),
    "far": ("FLOAT", {"default": 100.0, "min": 0.02, "max": 10000.0}),
    "invert_depth": ("BOOLEAN", {"default": False, "tooltip": "Enable if near things are dark in your depth map."}),
    "seed_mode": (["fixed", "increment"], {"default": "fixed",
                  "tooltip": "fixed = same noise every frame (calmer texture)."}),
    "start_frame": ("INT", {"default": 0, "min": 0, "max": 1000000,
                    "tooltip": "Feedback Sampler: render a chunk; init_image is the frame at start_frame."}),
    "end_frame": ("INT", {"default": 0, "min": 0, "max": 1000000, "tooltip": "0 = to the end."}),
}


class DifforumRenderOptions:
    """Everything the samplers can fine-tune, kept off the sampler itself.

    Without this node the look (colour lock, sharpen, grain) comes from the
    Director's style and the rest uses sensible defaults. Connect it to take
    manual control: colour anchoring, detail guard, in-loop symmetry, 3D depth
    calibration, noise seeding and chunked rendering.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": dict(_COMMON_OPTIONAL)}

    RETURN_TYPES = (OPTIONS,)
    RETURN_NAMES = ("options",)
    FUNCTION = "run"
    CATEGORY = CAT_RENDER

    def run(self, **kw):
        return (dict(kw),)


class DifforumFeedbackSampler:
    """The Deforum look on any image model: every frame is the last one,
    moved by the camera and re-imagined by the sampler.

    Plug a Director `direction` wire and a first frame; that is the whole
    setup. Works with SD1.5, SDXL, Flux, SD3.5 and turbo/LCM/DMD2 distills
    (plain MODEL / VAE / CONDITIONING). Connect a depth map (e.g. Depth
    Anything V2 on the first frame) and set the Director to 3d for real
    parallax - the depth then follows the image frame by frame.

    Outputs the frames, the tracked depth per frame (for comp / relight /
    video-model guides) and a run report.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        samplers, schedulers = _samplers()
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", {"tooltip": "Used when no prompt travel is connected."}),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "init_image": ("IMAGE", {"tooltip": "The first frame."}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 200}),
                "cfg": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "euler"}),
                "scheduler": (schedulers, {"default": "normal"}),
                "cadence": ("INT", {"default": 1, "min": 1, "max": 12,
                            "tooltip": "Diffuse every Nth frame; the rest are crossfaded warps. ~N x faster."}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "options": (OPTIONS, {"tooltip": "Difforum · Render Options, for manual control."}),
                "params": (PARAMS,),
                "camera": (CAMERA,),
                "strength": (SCHEDULE, {"tooltip": "Overrides the Director's energy curve."}),
                "cfg_curve": (SCHEDULE,),
                "prompts": (PROMPT,),
                "depth": ("IMAGE", {"tooltip": "Depth of the first frame (white = near), or a per-frame batch."}),
                "control_net": ("CONTROL_NET",),
                "control_image": ("IMAGE",),
                "control_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05}),
                "energy": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                           "tooltip": "Denoise when neither a Director nor a strength curve is connected."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING")
    RETURN_NAMES = ("frames", "depth", "report")
    FUNCTION = "run"
    CATEGORY = CAT_RENDER

    def run(self, model, positive, negative, vae, init_image, steps, cfg, sampler_name,
            scheduler, cadence, direction=None, options=None, params=None, camera=None,
            strength=None, cfg_curve=None, prompts=None, depth=None, control_net=None,
            control_image=None, control_strength=0.6, energy=0.5):
        kw = _options(direction, options)
        start_frame, end_frame = int(kw["start_frame"]), int(kw["end_frame"])
        params, camera, strength, cfg_curve, prompts = _resolve(
            direction, params, camera, strength, cfg_curve, prompts)
        n = int(params["max_frames"])
        w, h = int(params["width"]), int(params["height"])
        seed = int(params.get("seed", 0))
        strength = strength or Schedule(values=[float(energy)] * n, fps=params["fps"])
        cfg_at = cfg_curve.at if cfg_curve is not None else (lambda _f: float(cfg))

        ctrl = resize_bhwc(control_image, w, h) if control_image is not None else None
        diffuse = make_diffuser(model, vae, positive, negative, steps, cfg, sampler_name,
                                scheduler, strength.at, cfg_at, prompts, seed,
                                kw["seed_mode"], control_net, ctrl, control_strength)

        cfg_e = EngineConfig(
            width=w, height=h, border=kw["border"], symmetry=kw["symmetry"],
            symmetry_segments=int(kw["symmetry_segments"]), hole_noise=float(kw["hole_noise"]),
            cadence=int(cadence), anchor_mode=kw["anchor_mode"], near=float(kw["near"]),
            far=float(kw["far"]), invert_depth=bool(kw["invert_depth"]),
            translation_scale=float(kw["translation_scale"]),
            depth_tracking=kw["depth_tracking"], seed=seed,
            color_coherence=float(kw["color_coherence"]), color_mode=kw["color_mode"],
            sharpen=float(kw["sharpen"]), noise=float(kw["noise"]),
        )
        engine = FeedbackEngine(camera, cfg_e, depth=depth)
        pbar = progress_bar(n)
        frames = []
        for f, img in iter_feedback(engine, init_image, n, diffuse=diffuse, prompt_track=prompts,
                                    start_frame=start_frame, end_frame=end_frame):
            frames.append(img)
            pbar.update_absolute(f + 1, n)

        out = torch.cat(frames, dim=0)
        depth_out = self._depth_batch(engine, start_frame, len(frames), w, h)
        report = engine.report.text()
        if direction is not None and direction.direction.warnings:
            report += "\n" + "\n".join(f"  ! {x}" for x in direction.direction.warnings)
        return (out, depth_out, report)

    @staticmethod
    def _depth_batch(engine, start, count, w, h):
        if not engine.depth_log:
            return torch.full((1, h, w, 3), 0.5)
        keys = sorted(engine.depth_log)
        seq = []
        last = engine.depth_log[keys[0]]
        for f in range(start, start + count):
            last = engine.depth_log.get(f, last)
            seq.append(last.reshape(1, h, w))
        d = torch.cat(seq, dim=0).clamp(0, 1)
        return d.unsqueeze(-1).expand(-1, -1, -1, 3).contiguous()


# ---------------------------------------------------------------------------
# Live
# ---------------------------------------------------------------------------

def _to_preview(image, max_size=512):
    from PIL import Image
    arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return ("JPEG", Image.fromarray(arr), max_size)


class _SpoutSink:
    def __init__(self, name):
        self.sender = None
        if not name:
            return
        try:
            import SpoutGL
            self.sender = SpoutGL.SpoutSender()
            self.sender.setSenderName(name)
        except Exception as e:
            print(f"[Difforum] Spout output disabled ({e}); pip install SpoutGL to enable.")

    def send(self, image):
        if self.sender is None:
            return
        try:
            from OpenGL import GL
            arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            self.sender.sendImage(arr.tobytes(), arr.shape[1], arr.shape[0], GL.GL_RGB, False, 0)
            self.sender.setFrameSync(self.sender.getName())
        except Exception:
            self.sender = None

    def close(self):
        try:
            if self.sender is not None:
                self.sender.releaseSender()
        except Exception:
            pass


class _LiveSource:
    """Webcam index ("0") or a video path (looped) via OpenCV."""

    def __init__(self, spec):
        self.cap = None
        self.is_file = False
        spec = (spec or "").strip()
        if not spec:
            return
        try:
            import cv2
            src = int(spec) if spec.lstrip("-").isdigit() else spec
            self.is_file = not isinstance(src, int)
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise RuntimeError(f"cannot open {spec!r}")
        except Exception as e:
            print(f"[Difforum] live source disabled ({e})")
            self.cap = None

    def read(self, w, h):
        if self.cap is None:
            return None
        import cv2
        ok, frame = self.cap.read()
        if not ok and self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
        if not ok:
            return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return resize_bhwc(torch.from_numpy(frame).float().div(255.0).unsqueeze(0), w, h)

    def close(self):
        if self.cap is not None:
            self.cap.release()


class DifforumLiveSampler:
    """Realtime Difforum: queue once and watch it play inside the node.

    Pair with a 1-4 step model (SD-Turbo, SDXL-Turbo, LCM, DMD2) at ~512 px.
    `live_source` turns it into a magic mirror (webcam "0" or a video path);
    `stream_dir` / `spout_name` feed OBS, Resolume or TouchDesigner. Only the
    last `keep_frames` frames are returned, so long sessions do not fill RAM.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        samplers, schedulers = _samplers()
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "init_image": ("IMAGE",),
                "run_frames": ("INT", {"default": 480, "min": 1, "max": 10_000_000}),
                "steps": ("INT", {"default": 2, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 15.0, "step": 0.1}),
                "sampler_name": (samplers, {"default": "lcm" if "lcm" in samplers else samplers[0]}),
                "scheduler": (schedulers, {"default": "sgm_uniform" if "sgm_uniform" in schedulers else schedulers[0]}),
                "cadence": ("INT", {"default": 1, "min": 1, "max": 12}),
                "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.5,
                               "tooltip": "0 = as fast as possible."}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "options": (OPTIONS,),
                "params": (PARAMS,),
                "camera": (CAMERA,),
                "strength": (SCHEDULE,),
                "prompts": (PROMPT,),
                "depth": ("IMAGE",),
                "energy": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "loop_camera": ("BOOLEAN", {"default": True}),
                "live_source": ("STRING", {"default": "", "tooltip": "'' off, '0' webcam, or a video path."}),
                "source_blend": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "stream_dir": ("STRING", {"default": "", "tooltip": "Folder to write live PNG frames to."}),
                "spout_name": ("STRING", {"default": ""}),
                "keep_frames": ("INT", {"default": 240, "min": 1, "max": 100000}),
                "live_preview": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "report")
    FUNCTION = "run"
    CATEGORY = CAT_RENDER

    def run(self, model, positive, negative, vae, init_image, run_frames, steps, cfg,
            sampler_name, scheduler, cadence, target_fps, direction=None, options=None,
            params=None, camera=None, strength=None, prompts=None, depth=None, energy=0.5,
            loop_camera=True, live_source="", source_blend=0.9, stream_dir="",
            spout_name="", keep_frames=240, live_preview=True):
        from collections import deque
        from pathlib import Path

        from ..core.camera import CameraTrack

        kw = _options(direction, options)
        params, camera, strength, _cfgc, prompts = _resolve(
            direction, params, camera, strength, None, prompts)
        w, h = int(params["width"]), int(params["height"])
        n_cam = max(1, len(camera.deltas))
        total = int(run_frames)
        seed = int(params.get("seed", 0))

        # stretch the camera track to the run length (looped or held)
        idx = [(i % n_cam) if loop_camera else min(i, n_cam - 1) for i in range(total)]
        cam = CameraTrack(deltas=[camera.deltas[i] for i in idx], poses=[camera.poses[i] for i in idx],
                          zoom=[camera.zoom[i] for i in idx], fov=[camera.fov[i] for i in idx],
                          mode=camera.mode)
        sched = strength or Schedule(values=[float(energy)], fps=params["fps"])

        def strength_at(f):
            return sched.at(idx[f] if f < total else f)

        def prompt_view():
            if prompts is None:
                return None

            class _P:
                def __len__(self):
                    return total

                def __getitem__(self, f):
                    return prompts[idx[f]]
            return _P()

        diffuse = make_diffuser(model, vae, positive, negative, steps, cfg, sampler_name,
                                scheduler, strength_at, lambda _f: float(cfg), prompt_view(),
                                seed, kw["seed_mode"])
        cfg_e = EngineConfig(
            width=w, height=h, border=kw["border"], symmetry=kw["symmetry"],
            symmetry_segments=int(kw["symmetry_segments"]), hole_noise=float(kw["hole_noise"]),
            cadence=int(cadence), anchor_mode="first" if kw["anchor_mode"] == "scene" else kw["anchor_mode"],
            near=float(kw["near"]), far=float(kw["far"]), invert_depth=bool(kw["invert_depth"]),
            translation_scale=float(kw["translation_scale"]), depth_tracking=kw["depth_tracking"],
            seed=seed, color_coherence=float(kw["color_coherence"]), color_mode=kw["color_mode"],
            sharpen=float(kw["sharpen"]), noise=float(kw["noise"]),
        )
        engine = FeedbackEngine(cam, cfg_e, depth=depth)

        src = _LiveSource(live_source)
        sink = _SpoutSink(spout_name)
        out_dir = Path(stream_dir) if stream_dir else None
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)

        def pre_warp(prev, f):
            cam_frame = src.read(w, h)
            if cam_frame is None:
                return prev
            b = float(source_blend)
            return cam_frame.to(prev.dtype) * b + prev * (1.0 - b)

        pbar = progress_bar(total)
        kept = deque(maxlen=int(keep_frames))
        min_dt = 1.0 / target_fps if target_fps and target_fps > 0 else 0.0
        t_last = time.perf_counter()
        t0 = t_last
        try:
            for f, img in iter_feedback(engine, init_image, total, diffuse=diffuse,
                                        pre_warp=pre_warp if src.cap is not None else None):
                check_interrupt()
                kept.append(img)
                if live_preview:
                    pbar.update_absolute(f + 1, total, _to_preview(img))
                else:
                    pbar.update_absolute(f + 1, total)
                if out_dir is not None:
                    from PIL import Image
                    arr = (img[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
                    Image.fromarray(arr).save(out_dir / f"live_{f:07d}.png")
                sink.send(img)
                if min_dt:
                    dt = time.perf_counter() - t_last
                    if dt < min_dt:
                        time.sleep(min_dt - dt)
                t_last = time.perf_counter()
        finally:
            src.close()
            sink.close()
        fps_real = total / max(1e-6, time.perf_counter() - t0)
        return (torch.cat(list(kept), dim=0),
                engine.report.text() + f"\n  {fps_real:.2f} fps measured")


NODE_CLASS_MAPPINGS = {
    "Difforum_FeedbackSampler": DifforumFeedbackSampler,
    "Difforum_LiveSampler": DifforumLiveSampler,
    "Difforum_RenderOptions": DifforumRenderOptions,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_FeedbackSampler": "Difforum · Feedback Sampler",
    "Difforum_LiveSampler": "Difforum · Live Sampler",
    "Difforum_RenderOptions": "Difforum · Render Options",
}
