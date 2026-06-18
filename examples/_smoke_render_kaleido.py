"""16:9 kaleidoscope showcase: drives a running ComfyUI to render the living
kaleidoscope (in-loop symmetry + Echo Trails) and assemble a GIF for the README.
Usage: python _smoke_render_kaleido.py [port]"""
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "v1-5-pruned-emaonly-fp16.safetensors"
FRAME0 = ("ornate symmetric mandala, stained glass, iridescent, "
          "intricate fractal detail, glowing, kaleidoscope")
NEG = "blurry, low quality, watermark, text, asymmetric, jpeg artifacts"
W, H, N = 768, 432, 24
OUT_GIF = Path(__file__).resolve().parent / "difforum_kaleidoscope.gif"


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
            if "DifforumEchoTrails" in _get("/object_info"):
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
              "latent_image": ["4", 0], "seed": 7, "steps": 22, "cfg": 7.5, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup", "inputs": {"width": W, "height": H, "fps": 12, "max_frames": N, "seed": 7}},
        "8": {"class_type": "DifforumCamera", "inputs": {"params": ["7", 0], "mode": "2d", "fov": 40.0,
              "translation_x": "0:(0)", "translation_y": "0:(0)", "translation_z": "0:(0)",
              "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)", "rotation_3d_z": "0:(0.7)", "zoom": "0:(1.012)"}},
        "9": {"class_type": "DifforumSchedule", "inputs": {"params": ["7", 0], "schedule": "0:(0.42)", "easing": "linear"}},
        "10": {"class_type": "DifforumFeedbackSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
               "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0], "init_image": ["6", 0],
               "strength_schedule": ["9", 0],
               "steps": 22, "cfg": 7.5, "sampler_name": "euler", "scheduler": "normal",
               "color_coherence": 0.85, "color_mode": "lab",
               "symmetry": "kaleidoscope", "symmetry_segments": 6}},
        "13": {"class_type": "DifforumEchoTrails", "inputs": {"frames": ["10", 0], "decay": 0.55, "mix": 0.35}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "Difforum_kaleido"}},
    }


def build_gif(images):
    from PIL import Image
    frames = []
    for im in images:
        q = f"/view?filename={im['filename']}&subfolder={im.get('subfolder','')}&type={im.get('type','output')}"
        frames.append(Image.open(io.BytesIO(_getbin(q))).convert("RGB"))
    # ping-pong so the loop is seamless
    seq = frames + frames[-2:0:-1]
    seq[0].save(OUT_GIF, save_all=True, append_images=seq[1:], duration=90, loop=0, optimize=True)
    print(f"GIF -> {OUT_GIF}  ({len(seq)} frames, {OUT_GIF.stat().st_size//1024} KB)")


def main():
    wait_ready()
    pid = _post("/prompt", {"prompt": graph()})["prompt_id"]
    print(f"submitted {pid}; rendering {N} frames @ {W}x{H} (kaleidoscope)...")
    t0 = time.time()
    while True:
        h = _get(f"/history/{pid}")
        if pid in h and ("outputs" in h[pid] or h[pid].get("status", {}).get("completed")):
            imgs = h[pid].get("outputs", {}).get("11", {}).get("images", [])
            print(f"render DONE in {time.time()-t0:.1f}s - {len(imgs)} frames")
            if imgs:
                build_gif(imgs)
            return
        if time.time() - t0 > 900:
            raise SystemExit("timeout")
        time.sleep(2)


if __name__ == "__main__":
    main()
