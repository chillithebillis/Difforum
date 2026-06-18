"""
Difforum Live Step - one warp+diffuse iteration per execution, for realtime.

Realtime hosts (ComfyStream, StreamDiffusion, webcam loops) re-run the graph
every frame, so the right primitive is a *single-step* node: take the previous
output frame + a frame index + live params, warp by the camera, do a 1-2 step
(Turbo/LCM) re-diffuse, colour-match, and return ONE frame. Wire the output back
to `prev_frame` (feedback) and increment `frame_index` each tick.

This composes with realtime runtimes and stays cheap (no internal N-frame loop).
Pair with an SD-Turbo / LCM model for live FPS, and an NDI/Spout sink node to
push the stream to Resolume / TouchDesigner / OBS. See REALTIME.md.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.color import COLOR_MODES, match_color  # noqa: E402
from core.symmetry import SYMMETRY_MODES, apply_symmetry  # noqa: E402
from core.warp import warp_2d, warp_3d  # noqa: E402

CATEGORY = "Difforum/live"


def _resize_bhwc(image, width, height):
    if image.dim() == 3:
        image = image.unsqueeze(0)
    b, h, w, c = image.shape
    if h == height and w == width:
        return image
    chw = image.permute(0, 3, 1, 2)
    chw = torch.nn.functional.interpolate(chw, size=(height, width), mode="bilinear", align_corners=False)
    return chw.permute(0, 2, 3, 1)


class DifforumLiveStep:
    """One realtime frame: warp the previous frame + 1-2 step re-diffuse."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "params": ("DIFFORUM_PARAMS",),
                "camera": ("DIFFORUM_CAMERA",),
                "prev_frame": ("IMAGE",),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 2, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 15.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "color_coherence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_mode": (list(COLOR_MODES), {"default": "lab"}),
            },
            "optional": {
                "anchor": ("IMAGE",),
                "positive_schedule": ("DIFFORUM_PROMPT",),
                "depth": ("IMAGE",),
                "control_net": ("CONTROL_NET",),
                "control_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05}),
                "loop_camera": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frame", "next_index")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, positive, negative, vae, params, camera, prev_frame,
            frame_index, strength, steps, cfg, sampler_name, scheduler,
            color_coherence, color_mode="lab", anchor=None, positive_schedule=None,
            depth=None, control_net=None, control_strength=0.6, loop_camera=True, seed=0):
        from nodes import common_ksampler

        w, h = params["width"], params["height"]
        n = max(1, len(camera.deltas))
        cur = _resize_bhwc(prev_frame, w, h)[:1]

        # frame 0 is the seed: pass it through untouched
        if int(frame_index) <= 0:
            return (cur, 1)

        idx = (int(frame_index) % n) if loop_camera else min(int(frame_index), n - 1)
        delta = torch.as_tensor(camera.deltas[idx], dtype=torch.float32)
        zoom, fov = float(camera.zoom[idx]), float(camera.fov[idx])

        if camera.mode == "3d" and depth is not None:
            d = _resize_bhwc(depth, w, h)
            d = d.mean(dim=-1) if d.shape[-1] == 3 else d[..., 0]
            warped, _m = warp_3d(cur, d, delta, fov_deg=fov)
        else:
            tx, ty = float(delta[0, 3]), float(delta[1, 3])
            angle = math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0])))
            warped, _m = warp_2d(cur, tx, ty, angle, zoom)

        pos_f = positive
        if positive_schedule is not None and len(positive_schedule) > 0:
            pos_f = positive_schedule[min(idx, len(positive_schedule) - 1)]
        neg_f = negative

        if control_net is not None and control_strength > 0.0:
            from nodes import ControlNetApplyAdvanced
            pos_f, neg_f = ControlNetApplyAdvanced().apply_controlnet(
                pos_f, neg_f, control_net, warped[:, :, :, :3],
                float(control_strength), 0.0, 1.0, vae=vae,
            )

        latent = {"samples": vae.encode(warped[:, :, :, :3])}
        out = common_ksampler(
            model, int(seed) + int(frame_index), int(steps), float(cfg),
            sampler_name, scheduler, pos_f, neg_f, latent,
            denoise=max(0.0, min(1.0, float(strength))),
        )[0]
        image = vae.decode(out["samples"])

        ref = _resize_bhwc(anchor, w, h)[:1] if anchor is not None else cur
        if color_coherence > 0.0 and color_mode != "none":
            image = match_color(image, ref, strength=float(color_coherence), mode=color_mode)

        return (image[:1], int(frame_index) + 1)


def _to_preview(image, max_size=512):
    """[1,H,W,3] 0..1 tensor -> ('JPEG', PIL.Image, max_size) for live preview."""
    from PIL import Image
    arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
    return ("JPEG", Image.fromarray(arr), max_size)


class _SpoutSink:
    """Best-effort Spout sender (Windows). No-op + one warning if SpoutGL is
    not installed, so the node never crashes for users without it."""

    def __init__(self, name):
        self.ok = False
        self.sender = None
        if not name:
            return
        try:
            import SpoutGL  # noqa: F401
            self.sender = SpoutGL.SpoutSender()
            self.sender.setSenderName(name)
            self.ok = True
        except Exception as e:
            print(f"[Difforum] Spout sink disabled ({e}). pip install SpoutGL to enable.")

    def send(self, image):
        if not self.ok:
            return
        try:
            import SpoutGL
            from OpenGL import GL
            arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
            h, w = arr.shape[:2]
            self.sender.sendImage(arr.tobytes(), w, h, GL.GL_RGB, False, 0)
            self.sender.setFrameSync(self.sender.getName())
        except Exception:
            self.ok = False

    def close(self):
        try:
            if self.sender is not None:
                self.sender.releaseSender()
        except Exception:
            pass


class _LiveSource:
    """Optional live input for the Live Sampler: a webcam device or a video file.
    `spec` = "" (off, pure generative feedback), a digit like "0"/"1" (cv2 camera
    device), or a path to a video file (looped). Frames come back as [1,H,W,3]
    0..1 RGB. Degrades to off + one warning if it can't open."""

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
                raise RuntimeError(f"could not open source {spec!r}")
        except Exception as e:
            print(f"[Difforum] live source disabled ({e}). Using generative feedback.")
            self.cap = None

    @property
    def active(self):
        return self.cap is not None

    def read(self, w, h):
        if self.cap is None:
            return None
        import cv2
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file:  # loop the video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if not ok:
                return None
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(frame).float().div(255.0).unsqueeze(0)
        return _resize_bhwc(t, w, h)

    def close(self):
        if self.cap is not None:
            self.cap.release()


class DifforumLiveSampler:
    """Native realtime engine: a resident-model internal loop that warps, folds
    (symmetry) and re-diffuses one frame at a time, streaming a LIVE preview into
    the ComfyUI node as it runs - no external runtime needed. Queue once and watch
    it generate. Feed a webcam/video via `live_source` for a live "magic mirror"
    (camera -> kaleidoscope -> diffusion). Optional folder/Spout sinks feed a VJ
    app. Pair with a Turbo/LCM model (1-2 steps) for live FPS."""

    @classmethod
    def INPUT_TYPES(cls):
        import comfy.samplers
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "params": ("DIFFORUM_PARAMS",),
                "camera": ("DIFFORUM_CAMERA",),
                "init_image": ("IMAGE",),
                "duration_frames": ("INT", {"default": 240, "min": 1, "max": 1000000}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "steps": ("INT", {"default": 2, "min": 1, "max": 50}),
                "cfg": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 15.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "lcm"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "sgm_uniform"}),
                "color_coherence": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "color_mode": (list(COLOR_MODES), {"default": "lab"}),
                "symmetry": (list(SYMMETRY_MODES), {"default": "none"}),
                "symmetry_segments": ("INT", {"default": 6, "min": 2, "max": 64}),
                "target_fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 60.0, "step": 0.5}),
            },
            "optional": {
                "positive_schedule": ("DIFFORUM_PROMPT",),
                "depth": ("IMAGE",),
                "control_net": ("CONTROL_NET",),
                "control_strength": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 3.0, "step": 0.05}),
                "live_preview": ("BOOLEAN", {"default": True}),
                "live_source": ("STRING", {"default": ""}),
                "source_blend": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.05}),
                "stream_dir": ("STRING", {"default": ""}),
                "spout_name": ("STRING", {"default": ""}),
                "loop_camera": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, model, positive, negative, vae, params, camera, init_image,
            duration_frames, strength, steps, cfg, sampler_name, scheduler,
            color_coherence, color_mode, symmetry, symmetry_segments, target_fps,
            positive_schedule=None, depth=None, control_net=None, control_strength=0.6,
            live_preview=True, live_source="", source_blend=0.9, stream_dir="",
            spout_name="", loop_camera=True, seed=0):
        import time as _time
        from pathlib import Path as _Path

        import comfy.utils
        from nodes import common_ksampler

        w, h = params["width"], params["height"]
        n_cam = max(1, len(camera.deltas))
        anchor = _resize_bhwc(init_image, w, h)[:1]
        prev = anchor.clone()
        depth_b = _resize_bhwc(depth, w, h) if depth is not None else None

        out_dir = None
        if stream_dir:
            out_dir = _Path(stream_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        sink = _SpoutSink(spout_name)
        src = _LiveSource(live_source)

        cn_apply = None
        if control_net is not None and control_strength > 0.0:
            from nodes import ControlNetApplyAdvanced
            cn_apply = ControlNetApplyAdvanced().apply_controlnet

        pbar = comfy.utils.ProgressBar(int(duration_frames))
        frames = [prev]
        if live_preview:
            pbar.update_absolute(0, int(duration_frames), _to_preview(prev))

        min_dt = (1.0 / target_fps) if target_fps and target_fps > 0 else 0.0
        try:
            for i in range(1, int(duration_frames)):
                t0 = _time.perf_counter()
                idx = (i % n_cam) if loop_camera else min(i, n_cam - 1)
                delta = torch.as_tensor(camera.deltas[idx], dtype=torch.float32)
                zoom, fov = float(camera.zoom[idx]), float(camera.fov[idx])

                if camera.mode == "3d" and depth_b is not None:
                    d = depth_b.mean(dim=-1) if depth_b.shape[-1] == 3 else depth_b[..., 0]
                    warped, _m = warp_3d(prev, d, delta, fov_deg=fov)
                else:
                    tx, ty = float(delta[0, 3]), float(delta[1, 3])
                    angle = math.degrees(math.atan2(float(delta[1, 0]), float(delta[0, 0])))
                    warped, _m = warp_2d(prev, tx, ty, angle, zoom)

                # live "magic mirror": blend a fresh camera/video frame over the
                # warped feedback, then the symmetry below kaleidoscopes it
                if src.active:
                    cam = src.read(w, h)
                    if cam is not None:
                        b = float(source_blend)
                        warped = cam.to(warped.dtype) * b + warped * (1.0 - b)

                if symmetry != "none":
                    warped = apply_symmetry(warped, mode=symmetry, segments=int(symmetry_segments))

                pos_f = positive
                if positive_schedule is not None and len(positive_schedule) > 0:
                    pos_f = positive_schedule[min(idx, len(positive_schedule) - 1)]
                neg_f = negative
                if cn_apply is not None:
                    pos_f, neg_f = cn_apply(pos_f, neg_f, control_net, warped[:, :, :, :3],
                                            float(control_strength), 0.0, 1.0, vae=vae)

                latent = {"samples": vae.encode(warped[:, :, :, :3])}
                out = common_ksampler(model, int(seed) + i, int(steps), float(cfg),
                                      sampler_name, scheduler, pos_f, neg_f, latent,
                                      denoise=max(0.0, min(1.0, float(strength))))[0]
                image = vae.decode(out["samples"])[:1]
                if color_coherence > 0.0 and color_mode != "none":
                    image = match_color(image, anchor, strength=float(color_coherence), mode=color_mode)

                frames.append(image)
                prev = image
                if live_preview:
                    pbar.update_absolute(i, int(duration_frames), _to_preview(image))
                if out_dir is not None:
                    self._save_frame(out_dir, i, image)
                sink.send(image)

                if min_dt:
                    dt = _time.perf_counter() - t0
                    if dt < min_dt:
                        _time.sleep(min_dt - dt)
        finally:
            sink.close()
            src.close()

        return (torch.cat(frames, dim=0),)

    @staticmethod
    def _save_frame(out_dir, i, image):
        from PIL import Image
        arr = (image[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")
        Image.fromarray(arr).save(out_dir / f"live_{i:06d}.png")


NODE_CLASS_MAPPINGS = {
    "DifforumLiveStep": DifforumLiveStep,
    "DifforumLiveSampler": DifforumLiveSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumLiveStep": "Difforum · Live Step (realtime)",
    "DifforumLiveSampler": "Difforum · Live Sampler (realtime loop)",
}
