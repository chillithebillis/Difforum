"""SDXL extras: renders the audio-reactive showcase (bass pumps zoom, beats
pulse strength + spin) and refreshes the kaleidoscope effect GIF, both at
1024x576 with Juggernaut XL. Synthesizes a 120 BPM test beat first.
Drives a running ComfyUI. Usage: python _smoke_render_extras.py [port]"""
import io
import json
import struct
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8199
BASE = f"http://127.0.0.1:{PORT}"
CKPT = "juggernautXL_v9.safetensors"
W, H = 1024, 576
STEPS, CFG, SAMPLER, SCHED = 28, 6.5, "dpmpp_2m", "karras"
QUAL = ", highly detailed, sharp focus, intricate, cinematic lighting, vivid colors, masterpiece"
NEG = ("blurry, low quality, watermark, text, jpeg artifacts, deformed, washed out, "
       "oversaturated, plain background, low contrast")
HERE = Path(__file__).resolve().parent
COMFY_INPUT = Path(r"D:\ComfyUI-victor\input")
AUDIO_NAME = "difforum_beat.wav"


def make_beat(path, bpm=120, seconds=4.0, sr=44100):
    """A simple 4-on-the-floor kick + bass + offbeat hat, for the demo."""
    n = int(seconds * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, dtype=np.float32)
    beat = 60.0 / bpm
    # kick on every beat: 55 Hz sine with a fast exponential decay
    k = 0.0
    while k < seconds:
        s = int(k * sr)
        env = np.exp(-np.linspace(0, 1, int(0.18 * sr)) * 18.0)
        tone = np.sin(2 * np.pi * 55 * np.arange(len(env)) / sr)
        e = min(len(out), s + len(env))
        out[s:e] += (tone * env)[: e - s] * 0.9
        k += beat
    # offbeat hi-hat: short noise burst
    h = beat / 2
    while h < seconds:
        s = int(h * sr)
        env = np.exp(-np.linspace(0, 1, int(0.05 * sr)) * 40.0)
        noise = np.random.randn(len(env)).astype(np.float32)
        e = min(len(out), s + len(env))
        out[s:e] += (noise * env)[: e - s] * 0.15
        h += beat
    # sustained sub bass
    out += np.sin(2 * np.pi * 82 * t).astype(np.float32) * 0.12
    out = np.clip(out / (np.abs(out).max() + 1e-6), -1, 1)
    pcm = (out * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(b"".join(struct.pack("<h", v) for v in pcm))
    print(f"beat -> {path} ({seconds}s, {bpm} BPM)")


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
                return print("server ready")
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit("server/checkpoint not ready")


def base_nodes(frame0, n):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": frame0 + QUAL, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "5": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
              "latent_image": ["4", 0], "seed": 33, "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHED, "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "DifforumAnimSetup", "inputs": {"width": W, "height": H, "fps": 12, "max_frames": n, "seed": 33}},
    }


def graph_audio():
    n = 48
    frame0 = "abstract liquid light show, vivid neon colors, glowing particles, kinetic, reactive"
    prompts = (
        "0: abstract liquid light show, vivid neon colors, glowing particles, kinetic\n"
        "16: pulsing energy waves, neon bloom, rhythmic concentric patterns\n"
        "32: explosive bursts of color, fractal bloom, dynamic streaks\n"
        "47: a vortex of sound made visible, vibrant, swirling light")
    g = base_nodes(frame0, n)
    g.update({
        "20": {"class_type": "LoadAudio", "inputs": {"audio": AUDIO_NAME}},
        "21": {"class_type": "DifforumAudioAnalyzer", "inputs": {"params": ["7", 0], "audio": ["20", 0],
               "smoothing": 0.2, "normalize": True, "beat_sensitivity": 1.5}},
        "8": {"class_type": "DifforumCamera", "inputs": {"params": ["7", 0], "audio": ["21", 0], "mode": "2d", "fov": 40.0,
              "translation_x": "0:(0)", "translation_y": "0:(0)", "translation_z": "0:(0)",
              "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)",
              "rotation_3d_z": "0:(0.2 + 3.0*beat)", "zoom": "0:(1.0 + 0.5*low)"}},
        "22": {"class_type": "DifforumAudioSchedule", "inputs": {"params": ["7", 0], "audio": ["21", 0],
               "source": "beat", "mode": "add", "base": 0.4, "amount": 0.3, "smoothing": 0.2}},
        "12": {"class_type": "DifforumPromptSchedule", "inputs": {"params": ["7", 0], "clip": ["1", 1], "prompts": prompts, "easing": "ease_in_out"}},
        "10": {"class_type": "DifforumFeedbackSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
               "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0], "init_image": ["6", 0],
               "strength_schedule": ["22", 0], "positive_schedule": ["12", 0],
               "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHED,
               "color_coherence": 0.8, "color_mode": "lab", "symmetry": "none", "symmetry_segments": 6}},
        "13": {"class_type": "DifforumEchoTrails", "inputs": {"frames": ["10", 0], "decay": 0.5, "mix": 0.25}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "Difforum_sdxl_audio"}},
    })
    return g, n


def graph_kaleido():
    n = 40
    frame0 = "ornate symmetric mandala, stained glass, iridescent jewel tones, intricate fractal detail, glowing"
    g = base_nodes(frame0, n)
    g.update({
        "8": {"class_type": "DifforumCamera", "inputs": {"params": ["7", 0], "mode": "2d", "fov": 40.0,
              "translation_x": "0:(0)", "translation_y": "0:(0)", "translation_z": "0:(0)",
              "rotation_3d_x": "0:(0)", "rotation_3d_y": "0:(0)", "rotation_3d_z": "0:(0.7)", "zoom": "0:(1.012)"}},
        "9": {"class_type": "DifforumSchedule", "inputs": {"params": ["7", 0], "schedule": "0:(0.44)", "easing": "linear"}},
        "10": {"class_type": "DifforumFeedbackSampler", "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
               "vae": ["1", 2], "params": ["7", 0], "camera": ["8", 0], "init_image": ["6", 0],
               "strength_schedule": ["9", 0],
               "steps": STEPS, "cfg": CFG, "sampler_name": SAMPLER, "scheduler": SCHED,
               "color_coherence": 0.8, "color_mode": "lab", "symmetry": "kaleidoscope", "symmetry_segments": 6}},
        "13": {"class_type": "DifforumEchoTrails", "inputs": {"frames": ["10", 0], "decay": 0.5, "mix": 0.3}},
        "11": {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "Difforum_sdxl_kaleido"}},
    })
    return g, n


def _ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def fetch_frames(name, images):
    from PIL import Image
    tmp = HERE / f"_extra_{name}"
    tmp.mkdir(exist_ok=True)
    for f in tmp.glob("*.png"):
        f.unlink()
    for i, im in enumerate(images):
        q = f"/view?filename={im['filename']}&subfolder={im.get('subfolder','')}&type={im.get('type','output')}"
        Image.open(io.BytesIO(_getbin(q))).convert("RGB").save(tmp / f"f_{i:04d}.png")
    return tmp


def make_gif(tmp, dst, size, colors=128):
    ff = _ffmpeg()
    pal = HERE / "_pal.png"
    subprocess.run([ff, "-y", "-i", str(tmp / "f_%04d.png"), "-update", "1",
                    "-vf", f"scale={size}:flags=lanczos,palettegen=max_colors={colors}", str(pal)], check=True)
    subprocess.run([ff, "-y", "-framerate", "12", "-i", str(tmp / "f_%04d.png"), "-i", str(pal),
                    "-lavfi", f"scale={size}:flags=lanczos[x];[x][1:v]paletteuse", str(dst)], check=True)
    pal.unlink(missing_ok=True)
    print(f"GIF -> {dst.name} ({dst.stat().st_size//1024} KB)")


def make_mp4(tmp, dst):
    ff = _ffmpeg()
    subprocess.run([ff, "-y", "-framerate", "12", "-i", str(tmp / "f_%04d.png"),
                    "-vf", "scale=1280:720:flags=lanczos,minterpolate=fps=24:mi_mode=blend",
                    "-c:v", "libx264", "-crf", "17", "-pix_fmt", "yuv420p", str(dst)], check=True)
    print(f"MP4 -> {dst.name} ({dst.stat().st_size//1024} KB)")


def run(name, builder):
    g, n = builder()
    pid = _post("/prompt", {"prompt": g})["prompt_id"]
    print(f"[{name}] submitted {pid}; {n} frames @ {W}x{H}...")
    t0 = time.time()
    while True:
        h = _get(f"/history/{pid}")
        if pid in h and ("outputs" in h[pid] or h[pid].get("status", {}).get("completed")):
            imgs = h[pid].get("outputs", {}).get("11", {}).get("images", [])
            print(f"[{name}] render done in {time.time()-t0:.1f}s - {len(imgs)} frames")
            return fetch_frames(name, imgs) if imgs else None
        if time.time() - t0 > 2400:
            raise SystemExit(f"[{name}] timeout")
        time.sleep(3)


def main():
    make_beat(COMFY_INPUT / AUDIO_NAME)
    wait_ready()
    a = run("audio", graph_audio)
    if a:
        make_mp4(a, HERE / "difforum_audio.mp4")
        make_gif(a, HERE / "difforum_audio.gif", "600:338", 128)
    k = run("kaleido", graph_kaleido)
    if k:
        make_gif(k, HERE / "difforum_kaleidoscope.gif", "600:338", 128)
    print("ALL EXTRAS DONE")


if __name__ == "__main__":
    main()
