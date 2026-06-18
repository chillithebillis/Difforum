"""16:9 showcase render: drives a running ComfyUI to render a widescreen
Classic+ animation (768x432) for the README. Usage: python _smoke_render_169.py [port]"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "v1-5-pruned-emaonly-fp16.safetensors"
PROMPTS = ("0: a luminous alien jungle, giant glowing flora, volumetric light\n"
           "10: a crystalline ice canyon, aurora overhead\n"
           "19: a golden desert of dunes under twin suns")
FRAME0 = "a luminous alien jungle, giant glowing flora, volumetric light"
W, H, N = 768, 432, 20


def _post(p, d):
    r = urllib.request.Request(BASE + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())


def _get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=30).read())


def wait_ready(t=180):
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
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality, watermark", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
              "latent_image": ["4", 0], "seed": 5, "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup", "inputs": {"width": W, "height": H, "fps": 12, "max_frames": N, "seed": 5}},
        "8": {"class_type": "DifforumCamera", "inputs": {"params": ["7", 0], "mode": "2d", "fov": 40.0,
              "translation_x": "0:(3)", "translation_y": "0:(0)", "translation_z": "0:(0)",
              "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)", "rotation_3d_z": "0:(0.4)", "zoom": "0:(1.02)"}},
        "9": {"class_type": "DifforumSchedule", "inputs": {"params": ["7", 0], "schedule": "0:(0.5)", "easing": "linear"}},
        "12": {"class_type": "DifforumPromptSchedule", "inputs": {"params": ["7", 0], "clip": ["1", 1], "prompts": PROMPTS, "easing": "ease_in_out"}},
        "10": {"class_type": "DifforumFeedbackSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
               "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0], "init_image": ["6", 0],
               "strength_schedule": ["9", 0], "positive_schedule": ["12", 0],
               "steps": 20, "cfg": 7.0, "sampler_name": "euler", "scheduler": "normal", "color_coherence": 0.8, "color_mode": "lab"}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": "Difforum_169"}},
    }


def main():
    wait_ready()
    pid = _post("/prompt", {"prompt": graph()})["prompt_id"]
    print(f"submitted {pid}; rendering {N} frames @ {W}x{H}...")
    t0 = time.time()
    while True:
        h = _get(f"/history/{pid}")
        if pid in h and ("outputs" in h[pid] or h[pid].get("status", {}).get("completed")):
            imgs = h[pid].get("outputs", {}).get("11", {}).get("images", [])
            print(f"DONE in {time.time()-t0:.1f}s - {len(imgs)} frames")
            return
        if time.time() - t0 > 600:
            raise SystemExit("timeout")
        time.sleep(2)


if __name__ == "__main__":
    main()
