"""Smoke test for DifforumLiveSampler: runs a short internal-loop live session
through a running ComfyUI and checks it returns frames. Usage: python _smoke_live.py [port]"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "juggernautXL_v9.safetensors"


def _post(p, d):
    r = urllib.request.Request(BASE + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())


def _get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=30).read())


def wait_ready(t=300):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            oi = _get("/object_info")
            if "DifforumLiveSampler" in oi and CKPT in json.dumps(oi.get("CheckpointLoaderSimple", {})):
                return print("server ready")
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("not ready")


def graph():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "psychedelic mandala, vivid colors", "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": "blurry, low quality", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
              "latent_image": ["4", 0], "seed": 1, "steps": 8, "cfg": 4.0, "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup", "inputs": {"width": 512, "height": 512, "fps": 12, "max_frames": 30, "seed": 1}},
        "8": {"class_type": "DifforumCameraMove", "inputs": {"params": ["7", 0], "preset": "spiral", "speed": 1.0, "intensity": 1.0, "mode": "2d", "fov": 40.0}},
        "9": {"class_type": "DifforumLiveSampler", "inputs": {
            "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0], "vae": ["1", 2],
            "params": ["7", 0], "camera": ["8", 0], "init_image": ["6", 0],
            "duration_frames": 8, "strength": 0.5, "steps": 2, "cfg": 2.0,
            "sampler_name": "euler", "scheduler": "normal", "color_coherence": 0.5,
            "color_mode": "lab", "symmetry": "kaleidoscope", "symmetry_segments": 6,
            "target_fps": 0.0, "live_preview": True}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "Difforum_livetest"}},
    }


def main():
    wait_ready()
    pid = _post("/prompt", {"prompt": graph()})["prompt_id"]
    print("submitted", pid)
    t0 = time.time()
    while time.time() - t0 < 300:
        h = _get(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            if "outputs" in h[pid] or st.get("completed"):
                imgs = h[pid].get("outputs", {}).get("10", {}).get("images", [])
                print(f"LIVE SAMPLER OK - returned {len(imgs)} frames in {time.time()-t0:.1f}s")
                return
            if st.get("status_str") == "error":
                print("ERROR:", json.dumps(st)[:500])
                raise SystemExit(1)
        time.sleep(2)
    raise SystemExit("timeout")


if __name__ == "__main__":
    main()
