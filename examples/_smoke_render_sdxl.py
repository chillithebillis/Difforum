"""SDXL high-quality re-render of the showcase visuals (1024x576, dpmpp_2m
karras). Renders two clips: the kaleidoscope promo (sacred-geometry travel) and
a clean cosmic prompt-travel showcase. Exports MP4 + GIF for each.
Drives a running ComfyUI. Usage: python _smoke_render_sdxl.py [port]"""
import io
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "juggernautXL_v9.safetensors"
W, H = 1024, 576
STEPS, CFG, SAMPLER, SCHED = 28, 6.5, "dpmpp_2m", "karras"
QUAL = ", highly detailed, sharp focus, intricate, cinematic lighting, vivid colors, masterpiece"
NEG = ("blurry, low quality, watermark, text, jpeg artifacts, deformed, washed out, "
       "oversaturated, plain background, low contrast")
HERE = Path(__file__).resolve().parent

JOBS = [
    {
        "name": "promo", "n": 40, "symmetry": "kaleidoscope", "segments": 6,
        "strength": "0:(0.44)", "mode": "2d", "rot_z": "0:(0.6)", "zoom": "0:(1.012)",
        "tz": "0:(0)", "ty": "0:(0)", "echo": (0.5, 0.3),
        "frame0": "ornate symmetric mandala, stained glass, iridescent jewel tones, intricate fractal detail, glowing",
        "prompts": (
            "0: ornate symmetric mandala, stained glass, iridescent jewel tones, intricate fractal detail, glowing\n"
            "10: glowing sacred geometry, golden spirals, luminous neon lines, deep detail\n"
            "20: a blooming fractal flower of life, emerald and gold, ornate filigree\n"
            "30: a radiant rose window cathedral mandala, ruby and sapphire light\n"
            "39: a cosmic mandala of fire and light, vivid blooming symmetry, hypnotic"),
    },
    {
        "name": "showcase", "n": 40, "symmetry": "none", "segments": 6,
        "strength": "0:(0.45)", "mode": "3d", "rot_z": "0:(0.2)", "zoom": "0:(1.02)",
        "tz": "0:(1.2)", "ty": "0:(0)", "echo": (0.5, 0.15),
        "frame0": "a vast purple nebula, distant stars, volumetric cosmic dust, deep space, cinematic",
        "prompts": (
            "0: a vast purple nebula, distant stars, volumetric cosmic dust, deep space, cinematic\n"
            "10: a swirling spiral galaxy edge on, glowing core, dense star clusters\n"
            "20: the bright accretion disk of a black hole, gravitational lensing, intense light\n"
            "30: a newborn star igniting, brilliant rays piercing the dark, lens flare\n"
            "39: a radiant cosmic vista, glowing nebulae and stars, awe, epic scale"),
    },
]


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
            if CKPT in json.dumps(_get("/object_info").get("CheckpointLoaderSimple", {})):
                return print("server ready (checkpoint visible)")
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server/checkpoint not ready")


def graph(j):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": j["frame0"] + QUAL, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
              "latent_image": ["4", 0], "seed": 21, "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHED, "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup", "inputs": {"width": W, "height": H, "fps": 12, "max_frames": j["n"], "seed": 21}},
        "8": {"class_type": "DifforumCamera", "inputs": {"params": ["7", 0], "mode": j["mode"], "fov": 45.0,
              "translation_x": "0:(0)", "translation_y": j["ty"], "translation_z": j["tz"],
              "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)", "rotation_3d_z": j["rot_z"], "zoom": j["zoom"]}},
        "9": {"class_type": "DifforumSchedule", "inputs": {"params": ["7", 0], "schedule": j["strength"], "easing": "linear"}},
        "12": {"class_type": "DifforumPromptSchedule", "inputs": {"params": ["7", 0], "clip": ["1", 1], "prompts": j["prompts"], "easing": "ease_in_out"}},
        "10": {"class_type": "DifforumFeedbackSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
               "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0], "init_image": ["6", 0],
               "strength_schedule": ["9", 0], "positive_schedule": ["12", 0],
               "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHED,
               "color_coherence": 0.8, "color_mode": "lab",
               "symmetry": j["symmetry"], "symmetry_segments": j["segments"]}},
        "13": {"class_type": "DifforumEchoTrails", "inputs": {"frames": ["10", 0], "decay": j["echo"][0], "mix": j["echo"][1]}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "Difforum_sdxl_" + j["name"]}},
    }


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def export(name, images):
    from PIL import Image
    tmp = HERE / f"_sdxl_{name}"
    tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    for i, im in enumerate(images):
        q = f"/view?filename={im['filename']}&subfolder={im.get('subfolder','')}&type={im.get('type','output')}"
        Image.open(io.BytesIO(_getbin(q))).convert("RGB").save(tmp / f"f_{i:04d}.png")
    ff = _ffmpeg()
    mp4 = HERE / f"difforum_{name}.mp4"
    gif = HERE / f"difforum_{name}.gif"
    pal = HERE / "_pal.png"
    subprocess.run([ff, "-y", "-framerate", "12", "-i", str(tmp / "f_%04d.png"),
                    "-vf", "scale=1280:720:flags=lanczos,minterpolate=fps=24:mi_mode=blend",
                    "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(mp4)], check=True)
    subprocess.run([ff, "-y", "-i", str(tmp / "f_%04d.png"), "-update", "1",
                    "-vf", "scale=600:338:flags=lanczos,palettegen=max_colors=128", str(pal)], check=True)
    subprocess.run([ff, "-y", "-framerate", "12", "-i", str(tmp / "f_%04d.png"), "-i", str(pal),
                    "-lavfi", "scale=600:338:flags=lanczos[x];[x][1:v]paletteuse", str(gif)], check=True)
    pal.unlink(missing_ok=True)
    print(f"  {name}: MP4 {mp4.stat().st_size//1024}KB  GIF {gif.stat().st_size//1024}KB")


def run(j):
    pid = _post("/prompt", {"prompt": graph(j)})["prompt_id"]
    print(f"[{j['name']}] submitted {pid}; {j['n']} frames @ {W}x{H}...")
    t0 = time.time()
    while True:
        h = _get(f"/history/{pid}")
        if pid in h and ("outputs" in h[pid] or h[pid].get("status", {}).get("completed")):
            imgs = h[pid].get("outputs", {}).get("11", {}).get("images", [])
            print(f"[{j['name']}] render done in {time.time()-t0:.1f}s - {len(imgs)} frames")
            if imgs:
                export(j["name"], imgs)
            return
        if time.time() - t0 > 2400:
            raise SystemExit(f"[{j['name']}] timeout")
        time.sleep(3)


def main():
    wait_ready()
    for j in JOBS:
        run(j)
    print("ALL SDXL RENDERS DONE")


if __name__ == "__main__":
    main()
