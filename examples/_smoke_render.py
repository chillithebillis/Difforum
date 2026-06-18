"""
Smoke render: drive a running ComfyUI via its HTTP API to actually render a
short Difforum Classic+ animation end-to-end with the real SD1.5 checkpoint.

Usage: python _smoke_render.py [port]
Assumes the ComfyUI server is already up on 127.0.0.1:<port>.
"""

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "v1-5-pruned-emaonly-fp16.safetensors"


def _post(path, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(BASE + path, data=data,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def _get(path):
    return json.loads(urllib.request.urlopen(BASE + path, timeout=30).read())


def wait_ready(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            oi = _get("/object_info")
            if "DifforumFeedbackSampler" in oi and "CheckpointLoaderSimple" in oi:
                print("server ready; Difforum nodes registered")
                return True
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server did not become ready in time")


def build_prompt():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "psychedelic cosmic tunnel, vivid neon colors, highly detailed",
                         "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "blurry, low quality, watermark, text", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "seed": 7, "steps": 15, "cfg": 7.0,
                         "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup",
              "inputs": {"width": 512, "height": 512, "fps": 12, "max_frames": 12, "seed": 7}},
        "8": {"class_type": "DifforumCamera",
              "inputs": {"params": ["7", 0], "mode": "2d", "fov": 40.0,
                         "translation_x": "0:(3)", "translation_y": "0:(0)", "translation_z": "0:(0)",
                         "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)", "rotation_3d_z": "0:(1.0)",
                         "zoom": "0:(1.02)"}},
        "9": {"class_type": "DifforumSchedule",
              "inputs": {"params": ["7", 0], "schedule": "0:(0.5)", "easing": "linear"}},
        "10": {"class_type": "DifforumFeedbackSampler",
               "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                          "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0],
                          "init_image": ["6", 0], "strength_schedule": ["9", 0],
                          "steps": 15, "cfg": 7.0, "sampler_name": "euler",
                          "scheduler": "normal", "color_coherence": 0.8}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "Difforum_smoke"}},
    }


def main():
    wait_ready()
    prompt = build_prompt()
    res = _post("/prompt", {"prompt": prompt})
    pid = res["prompt_id"]
    print(f"submitted prompt {pid}; rendering 12 frames...")

    t0 = time.time()
    while True:
        hist = _get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("completed") or "outputs" in entry:
                outs = entry.get("outputs", {})
                imgs = outs.get("11", {}).get("images", [])
                print(f"DONE in {time.time()-t0:.1f}s - {len(imgs)} frames saved:")
                for im in imgs:
                    print(f"  {im.get('subfolder','')}/{im['filename']}")
                if status.get("status_str") == "error" or not imgs:
                    print("status:", json.dumps(status, indent=2)[:2000])
                return
            if status.get("status_str") == "error":
                print("ERROR:", json.dumps(status, indent=2)[:3000])
                return
        if time.time() - t0 > 600:
            raise SystemExit("render timed out")
        time.sleep(2)


if __name__ == "__main__":
    main()
