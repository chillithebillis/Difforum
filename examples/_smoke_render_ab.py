"""A/B proof render for the detail guard: two identical SDXL feedback runs
(same seed, camera, prompt), A with the old settings (zeros border, no sharpen,
no noise) and B with the detail guard on. Exports a side-by-side GIF + final
frame PNG. Usage: python _smoke_render_ab.py [port]"""
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "juggernautXL_v9.safetensors"
W, H, N = 1024, 576, 48
STEPS, CFG, SAMPLER, SCHED = 22, 6.5, "dpmpp_2m", "karras"
PROMPT = ("intricate ancient stone temple wall covered in detailed carvings, moss, "
          "roots, volumetric light, highly detailed, sharp focus, cinematic")
NEG = "blurry, low quality, watermark, text, jpeg artifacts, washed out"
HERE = Path(__file__).resolve().parent

VARIANTS = {
    "off": {"border": "zeros", "sharpen": 0.0, "noise": 0.0},
    "guard": {"border": "reflection", "sharpen": 0.3, "noise": 0.03},
}


def _post(p, d):
    r = urllib.request.Request(BASE + p, data=json.dumps(d).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=30).read())


def _get(p):
    return json.loads(urllib.request.urlopen(BASE + p, timeout=30).read())


def _getbin(p):
    return urllib.request.urlopen(BASE + p, timeout=60).read()


def wait_ready(t=300):
    t0 = time.time()
    while time.time() - t0 < t:
        try:
            oi = _get("/object_info")
            fb = oi.get("DifforumFeedbackSampler", {})
            if "border" in json.dumps(fb) and CKPT in json.dumps(oi.get("CheckpointLoaderSimple", {})):
                return print("server ready (new sampler visible)")
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server/sampler not ready")


def graph(v):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
              "latent_image": ["4", 0], "seed": 42, "steps": 26, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHED, "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup", "inputs": {"width": W, "height": H, "fps": 12, "max_frames": N, "seed": 42}},
        "8": {"class_type": "DifforumCameraMove", "inputs": {"params": ["7", 0], "preset": "spiral",
              "speed": 1.2, "intensity": 1.2, "mode": "2d", "fov": 40.0}},
        "9": {"class_type": "DifforumSchedule", "inputs": {"params": ["7", 0], "schedule": "0:(0.5)", "easing": "linear"}},
        "10": {"class_type": "DifforumFeedbackSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
               "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0], "init_image": ["6", 0],
               "strength_schedule": ["9", 0],
               "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHED,
               "color_coherence": 0.8, "color_mode": "lab", "symmetry": "none", "symmetry_segments": 6,
               "border": v["border"], "sharpen": v["sharpen"], "noise": v["noise"]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["10", 0], "filename_prefix": f"Difforum_ab_{v['border']}"}},
    }


def run(name, v):
    pid = _post("/prompt", {"prompt": graph(v)})["prompt_id"]
    print(f"[{name}] submitted {pid}; {N} frames...")
    t0 = time.time()
    while True:
        h = _get(f"/history/{pid}")
        if pid in h and ("outputs" in h[pid] or h[pid].get("status", {}).get("completed")):
            imgs = h[pid].get("outputs", {}).get("11", {}).get("images", [])
            print(f"[{name}] done in {time.time()-t0:.1f}s - {len(imgs)} frames")
            return imgs
        if time.time() - t0 > 2400:
            raise SystemExit(f"[{name}] timeout")
        time.sleep(3)


def fetch(images):
    from PIL import Image
    out = []
    for im in images:
        q = f"/view?filename={im['filename']}&subfolder={im.get('subfolder','')}&type={im.get('type','output')}"
        out.append(Image.open(io.BytesIO(_getbin(q))).convert("RGB"))
    return out


def label(img, text):
    from PIL import ImageDraw
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 150, 22], fill=(0, 0, 0))
    d.text((6, 4), text, fill=(255, 255, 255))
    return img


def export(a, b):
    from PIL import Image
    tw, th = 480, 270
    frames = []
    for fa, fb in zip(a, b):
        left = label(fa.resize((tw, th), Image.LANCZOS), "detail guard OFF")
        right = label(fb.resize((tw, th), Image.LANCZOS), "detail guard ON")
        sheet = Image.new("RGB", (tw * 2 + 4, th), (16, 16, 16))
        sheet.paste(left, (0, 0))
        sheet.paste(right, (tw + 4, 0))
        frames.append(sheet)
    gif = HERE / "difforum_detail_ab.gif"
    q = [f.quantize(colors=128, dither=Image.Dither.NONE) for f in frames]
    q[0].save(gif, save_all=True, append_images=q[1:], duration=110, loop=0, optimize=True)
    print(f"GIF -> {gif.name} ({gif.stat().st_size//1024} KB)")
    # final-frame still, full res halves
    fw, fh = 640, 360
    still = Image.new("RGB", (fw * 2 + 4, fh), (16, 16, 16))
    still.paste(label(a[-1].resize((fw, fh), Image.LANCZOS), "detail guard OFF"), (0, 0))
    still.paste(label(b[-1].resize((fw, fh), Image.LANCZOS), "detail guard ON"), (fw + 4, 0))
    png = HERE / "difforum_detail_ab_final.png"
    still.save(png)
    print(f"PNG -> {png.name} ({png.stat().st_size//1024} KB)")


def main():
    wait_ready()
    a = fetch(run("off", VARIANTS["off"]))
    b = fetch(run("guard", VARIANTS["guard"]))
    export(a, b)
    print("A/B DONE")


if __name__ == "__main__":
    main()
