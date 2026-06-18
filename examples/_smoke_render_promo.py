"""Promo reel for Difforum: a 16:9 cosmic prompt-travel journey with camera
motion, exported to MP4 (social) and GIF (README). Drives a running ComfyUI.
Usage: python _smoke_render_promo.py [port]"""
import io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "v1-5-pruned-emaonly-fp16.safetensors"
PROMPTS = (
    "0: ornate symmetric mandala, stained glass, iridescent jewel tones, intricate fractal detail, glowing\n"
    "12: glowing sacred geometry, golden spirals, luminous neon lines, deep detail\n"
    "24: a blooming fractal flower of life, emerald and gold, ornate filigree\n"
    "36: a radiant rose window cathedral mandala, ruby and sapphire light\n"
    "47: a cosmic mandala of fire and light, vivid blooming symmetry, hypnotic"
)
FRAME0 = "ornate symmetric mandala, stained glass, iridescent jewel tones, intricate fractal detail, glowing"
NEG = "blurry, low quality, watermark, text, jpeg artifacts, deformed, asymmetric, washed out"
W, H, N = 768, 432, 48
HERE = Path(__file__).resolve().parent
TMP = HERE / "_promo_frames"
MP4 = HERE / "difforum_promo.mp4"
GIF = HERE / "difforum_promo.gif"


def _post(p, d):
    r = urllib.request.Request(BASE + p, data=json.dumps(d).encode(),
                               headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())


def _get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=30).read())


def _getbin(p):
    return urllib.request.urlopen(BASE + p, timeout=30).read()


def wait_ready(t=240):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            if "DifforumFeedbackSampler" in _get("/object_info"):
                return print("server ready")
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server not ready")


def graph():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": FRAME0, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
              "latent_image": ["4", 0], "seed": 11, "steps": 24, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup", "inputs": {"width": W, "height": H, "fps": 12, "max_frames": N, "seed": 11}},
        "8": {"class_type": "DifforumCamera", "inputs": {"params": ["7", 0], "mode": "2d", "fov": 40.0,
              "translation_x": "0:(0)", "translation_y": "0:(0)", "translation_z": "0:(0)",
              "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)", "rotation_3d_z": "0:(0.6)", "zoom": "0:(1.012)"}},
        "9": {"class_type": "DifforumSchedule", "inputs": {"params": ["7", 0], "schedule": "0:(0.44)", "easing": "linear"}},
        "12": {"class_type": "DifforumPromptSchedule", "inputs": {"params": ["7", 0], "clip": ["1", 1], "prompts": PROMPTS, "easing": "ease_in_out"}},
        "10": {"class_type": "DifforumFeedbackSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
               "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0], "init_image": ["6", 0],
               "strength_schedule": ["9", 0], "positive_schedule": ["12", 0],
               "steps": 24, "cfg": 7.5, "sampler_name": "euler", "scheduler": "normal",
               "color_coherence": 0.8, "color_mode": "lab",
               "symmetry": "kaleidoscope", "symmetry_segments": 6}},
        "13": {"class_type": "DifforumEchoTrails", "inputs": {"frames": ["10", 0], "decay": 0.5, "mix": 0.3}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "Difforum_promo"}},
    }


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def export(images):
    from PIL import Image
    TMP.mkdir(exist_ok=True)
    for f in TMP.glob("*.png"):
        f.unlink()
    for i, im in enumerate(images):
        q = f"/view?filename={im['filename']}&subfolder={im.get('subfolder','')}&type={im.get('type','output')}"
        Image.open(io.BytesIO(_getbin(q))).convert("RGB").save(TMP / f"f_{i:04d}.png")
    ff = _ffmpeg()
    # MP4: upscale to 720p, blend-interpolate 12 -> 24 fps for smoothness, h264
    subprocess.run([ff, "-y", "-framerate", "12", "-i", str(TMP / "f_%04d.png"),
                    "-vf", "scale=1280:720:flags=lanczos,minterpolate=fps=24:mi_mode=blend",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(MP4)], check=True)
    print(f"MP4 -> {MP4}  ({MP4.stat().st_size//1024} KB)")
    # GIF: ping-pong, palette, for the README
    pal = HERE / "_palette.png"
    subprocess.run([ff, "-y", "-i", str(TMP / "f_%04d.png"), "-update", "1",
                    "-vf", "scale=512:288:flags=lanczos,palettegen=max_colors=96", str(pal)], check=True)
    subprocess.run([ff, "-y", "-framerate", "12", "-i", str(TMP / "f_%04d.png"), "-i", str(pal),
                    "-lavfi", "scale=512:288:flags=lanczos[x];[x][1:v]paletteuse", str(GIF)], check=True)
    pal.unlink(missing_ok=True)
    print(f"GIF -> {GIF}  ({GIF.stat().st_size//1024} KB)")


def main():
    wait_ready()
    pid = _post("/prompt", {"prompt": graph()})["prompt_id"]
    print(f"submitted {pid}; rendering {N} frames @ {W}x{H} (promo)...")
    t0 = time.time()
    while True:
        h = _get(f"/history/{pid}")
        if pid in h and ("outputs" in h[pid] or h[pid].get("status", {}).get("completed")):
            imgs = h[pid].get("outputs", {}).get("11", {}).get("images", [])
            print(f"render DONE in {time.time()-t0:.1f}s - {len(imgs)} frames")
            if imgs:
                export(imgs)
            return
        if time.time() - t0 > 1200:
            raise SystemExit("timeout")
        time.sleep(2)


if __name__ == "__main__":
    main()
