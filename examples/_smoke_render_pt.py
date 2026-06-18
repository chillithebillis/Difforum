"""
Smoke render WITH prompt travel: drives a running ComfyUI to render a short
Classic+ animation whose prompt morphs across keyframes (forest -> cave ->
galaxy) while the camera moves. Validates DifforumPromptSchedule end-to-end.

Usage: python _smoke_render_pt.py [port]
"""

import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "v1-5-pruned-emaonly-fp16.safetensors"

PROMPTS = ("0: a serene misty forest, soft volumetric light, lush\n"
           "8: a glowing crystal cave, bioluminescent, deep blues\n"
           "15: a vast starry galaxy, swirling nebula, vivid colors")
FRAME0 = "a serene misty forest, soft volumetric light, lush"


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
            if "DifforumPromptSchedule" in oi and "DifforumFeedbackSampler" in oi:
                print("server ready; Difforum prompt-travel node registered")
                return True
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server did not become ready in time")


def build_prompt():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": FRAME0, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "blurry, low quality, watermark, text", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "seed": 11, "steps": 16, "cfg": 7.0,
                         "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup",
              "inputs": {"width": 512, "height": 512, "fps": 12, "max_frames": 16, "seed": 11}},
        "8": {"class_type": "DifforumCamera",
              "inputs": {"params": ["7", 0], "mode": "2d", "fov": 40.0,
                         "translation_x": "0:(2)", "translation_y": "0:(0)", "translation_z": "0:(0)",
                         "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)", "rotation_3d_z": "0:(0.7)",
                         "zoom": "0:(1.025)"}},
        "9": {"class_type": "DifforumSchedule",
              "inputs": {"params": ["7", 0], "schedule": "0:(0.55)", "easing": "linear"}},
        "12": {"class_type": "DifforumPromptSchedule",
               "inputs": {"params": ["7", 0], "clip": ["1", 1], "prompts": PROMPTS,
                          "easing": "ease_in_out"}},
        "10": {"class_type": "DifforumFeedbackSampler",
               "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                          "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0],
                          "init_image": ["6", 0], "strength_schedule": ["9", 0],
                          "positive_schedule": ["12", 0],
                          "steps": 16, "cfg": 7.0, "sampler_name": "euler",
                          "scheduler": "normal", "color_coherence": 0.7}},
        "11": {"class_type": "SaveImage",
               "inputs": {"images": ["10", 0], "filename_prefix": "Difforum_pt"}},
    }


def main():
    wait_ready()
    res = _post("/prompt", {"prompt": build_prompt()})
    pid = res["prompt_id"]
    print(f"submitted {pid}; rendering 16 frames with prompt travel...")
    t0 = time.time()
    while True:
        hist = _get(f"/history/{pid}")
        if pid in hist:
            entry = hist[pid]
            status = entry.get("status", {})
            if status.get("completed") or "outputs" in entry:
                imgs = entry.get("outputs", {}).get("11", {}).get("images", [])
                print(f"DONE in {time.time()-t0:.1f}s - {len(imgs)} frames")
                for im in imgs:
                    print(f"  {im['filename']}")
                if not imgs:
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
