# Difforum: Deforum-style animation for ComfyUI

Keyframe animation driven by math expressions, with camera moves, audio
reactivity and prompt travel. It brings the Deforum workflow to current models
(SD1.5, SDXL, **Flux**, SD3.5, Wan 2.2) and Python 3.12+.

![Difforum kaleidoscope promo](examples/difforum_promo.gif)

*Sacred-geometry prompt travel with in-loop kaleidoscope symmetry and Echo
Trails. SDXL (Juggernaut XL) feedback render at 1024x576. Full 720p clip:
[`examples/difforum_promo.mp4`](examples/difforum_promo.mp4). Made entirely with the
Difforum nodes in ComfyUI.*

![Difforum 16:9 showcase](examples/difforum_showcase.gif)

*16:9 Classic+ render. Prompt travel `nebula, spiral galaxy, black hole,
newborn star` blended per frame, with the camera flying forward and the feedback
loop reshaping the scene. SDXL at 1024x576, full clip:
[`examples/difforum_showcase.mp4`](examples/difforum_showcase.mp4).*

![Difforum audio-reactive](examples/difforum_audio.gif)

*Audio-reactive render: bass pumps the zoom, beats pulse the strength and spin.
One node turns any band into a curve. SDXL, full clip:
[`examples/difforum_audio.mp4`](examples/difforum_audio.mp4).*

## Why it exists

The original [Deforum](https://github.com/deforum-art/sd-webui-deforum) is hard
to run today: its library refuses Python 3.12, it's tied to SD1.5-era img2img,
and the frame-by-frame loop flickers. Difforum rebuilds the same ideas from
scratch and fixes those problems.

What you get:

- The same `0:(expr)` keyframe syntax (`sin`, `cos`, `t`, audio variables),
  camera schedules, audio reactivity and prompt travel.
- A model-agnostic sampler that takes plain ComfyUI `MODEL/VAE/CONDITIONING`, so
  you can drop in SDXL, **Flux**, SD3.5 or an SD-Turbo/LCM model instead of
  2023-era SD1.5. No lock-in.
- Three render paths over one control layer: **Classic+** (depth-aware warp,
  re-diffuse, LAB colour match), **Hybrid** (drives Wan 2.2 VACE), and
  **AnimateDiff** (feeds prompt travel and schedules into AnimateDiff-Evolved).
- No `eval()`, no exotic dependencies (safe AST evaluator, numpy FFT audio, torch
  warps), 12 test suites, and a clean install on Python 3.12 and the Comfy Registry.

See [DESIGN.md](DESIGN.md) for the architecture and [REALTIME.md](REALTIME.md)
for the live-performance direction.

## Status

**Phase 1 - Schedule Engine (done).** GPU-free. The animation "brain":

| Node | What it does |
|---|---|
| **Difforum · Anim Setup** | Global params (size, fps, frames, seed) shared downstream |
| **Difforum · Schedule** | Parses `0:(expr), 60:(expr)` into a per-frame curve |
| **Difforum · Sample Schedule** | Reads a curve value at one frame |
| **Difforum · Schedule Info** | Debug summary + ASCII sparkline of a curve |
| **Difforum · Schedule Plot** | Renders the computed curve as an IMAGE (Preview-ready) |

**Phase Audio - Reactivity (done).** Pure numpy (no librosa):

| Node | What it does |
|---|---|
| **Difforum · Audio Analyzer** | AUDIO → per-frame curves `amp/low/mid/high/onset/beat` |
| **Difforum · Audio Schedule (reactive)** | Turn a band into a ready curve (bass-pump, beat-pulse…) for any schedule input |

Connect Audio Analyzer's `audio_curves` to Schedule's `audio` input, then use
the curve names in expressions: `0:(0.2 + 0.8*amp)`, `0:(beat*0.6)`,
`0:(2*low - high)`.

**Hybrid foundation + render (done).**

| Node | What it does |
|---|---|
| **Difforum · Camera Move (presets)** | Intuitive camera: pick a move (zoom/orbit/spiral/shake…) + speed + intensity |
| **Difforum · Camera (advanced)** | Deforum 2D/3D camera schedules → per-frame poses (audio-reactive) |
| **Difforum · Model Profile** | Auto-resolve model/quant/res/steps for the GPU (12→32GB+, GGUF) |
| **Difforum · Model Catalog** | Classic, trainable model recipes + download/training guide |
| **Difforum · Warp (2D/3D)** | Classic Deforum warp + occlusion mask (force_2d / force_3d) |
| **Difforum · Feedback Sampler** | Classic+ video: warp→re-diffuse→colour-match (LAB/RGB) loop (SD1.5/SDXL) |
| **Difforum · Guide Builder** | Warp anchor along camera path → guide batch for Wan 2.2 VACE |
| **Difforum · Prompt Scenes (travel)** | Prompt travel by scenes - one text box per scene, auto-spaced |
| **Difforum · Prompt Schedule (travel)** | Prompt travel from a `frame: prompt` schedule |
| **Difforum · Prompt Batch (→ AnimateDiff)** | Stacks prompt travel into one batched CONDITIONING for AnimateDiff |

**Effects (done).** Mirror, kaleidoscope and smooth temporal trails:

| Node | What it does |
|---|---|
| **Difforum · Symmetry / Kaleidoscope** | Mirror H/V, 4-fold quad, or N-segment kaleidoscope on a frame or batch |
| **Difforum · Echo Trails** | Long-exposure motion trails across a frame batch (smooth, hypnotic) |

**VJ look (done).** Grade video footage for live visuals, no model required:

| Node | What it does |
|---|---|
| **Difforum · VJ Look (presets)** | One-shot grade: neon / cinematic / vaporwave / film / noir / psychedelic, with an intensity that an audio schedule can pulse to the beat |
| **Difforum · Colour Grade** | Exposure, contrast, saturation, white balance, lift/gamma/gain |
| **Difforum · Glow** | Neon bloom (blur the bright areas, screen-blend them back) |
| **Difforum · Load Video** | Read a clip into an IMAGE batch (frame skip, resize cap), via OpenCV |
| **Difforum · Save Video (MP4)** | Write an IMAGE batch back to an h264 MP4, via ffmpeg |

These are plain IMAGE to IMAGE, so they work on a still, a feedback render or a
whole footage batch. The `difforum_vj_footage.json` template is a complete,
self-contained pipeline (Load Video to VJ Look to Echo Trails to Save Video) that
needs **no checkpoint and no external nodes**: footage in, graded MP4 out. The
video IO uses opencv-python and imageio-ffmpeg (usually already in ComfyUI).

Symmetry is also built into the **Feedback Sampler** (`symmetry` + `symmetry_segments`):
applied *inside* the loop it compounds each frame and the diffusion heals the
seams, giving a living kaleidoscope (`difforum_mesmerize_kaleidoscope.json`).

![Difforum living kaleidoscope](examples/difforum_kaleidoscope.gif)

*In-loop kaleidoscope symmetry on an SD1.5 feedback render, smoothed with Echo
Trails. The pattern folds every frame and the diffusion reseals the seams, so it
keeps growing on itself. Made entirely with the Difforum nodes.*

Three render paths: **Classic+** (Feedback Sampler - self-contained Deforum
video), **Hybrid** (Guide Builder → your Wan 2.2 VACE graph), and **AnimateDiff**
(Prompt Batch → AnimateDiff-Evolved for native motion + Difforum control). See
DESIGN.md §7. Modern bases (SDXL / Flux / SD3.5) drop into the Feedback Sampler -
it is model-agnostic. AnimateDiff motion modules need AnimateDiff's own batch
sampler (the `difforum_animatediff_sd15.json` template wires it).

The 3D warp exposes `translation_scale` to tune motion vs. the depth estimator's
(non-metric) range; colour coherence does perceptual **LAB** matching by default.

![Difforum hybrid guide preview](examples/difforum_guide_demo.gif)

*Hybrid bridge: the Guide Builder warps one anchor image along the camera path
into the guide batch a Wan 2.2 VACE graph consumes (camera by Difforum, fill by Wan).*

> **Black output?** Some ComfyUI builds make SD1.5 fp16 produce NaN (all-black
> frames). Launch ComfyUI with `--force-fp32` - it's a base-model precision
> issue, not Difforum.

## Install

**Manual / git (local or self-hosted):**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/chillithebillis/Difforum.git difforum
# deps are minimal (numpy, already present); only needed for a bare env:
pip install -r difforum/requirements.txt
```

Restart ComfyUI; the console prints `[Difforum] loaded N nodes`.

**ComfyUI Manager:** *Custom Nodes Manager* → install via Git URL with the repo
above (or by name once published to the registry).

**Comfy Registry:** Difforum is registry-ready via `pyproject.toml` (no heavy or
version-locked deps, so it builds on the cloud's Python 3.12). To publish:

```bash
pip install comfy-cli
comfy node publish        # uploads to the Comfy Registry
```

**Cloud GPU (self-host):** the quickest way to run Difforum in the cloud today is
a managed GPU host like [RunPod](https://www.runpod.io) or [Vast.ai](https://vast.ai):
spin up a ComfyUI template, then in the Manager install via the Git URL above (or
`git clone` into `custom_nodes`). Everything runs with no gatekeeping.

**Comfy Cloud** runs a curated set of vetted node packs, so a node has to be
reviewed and added by the Comfy team before it works there. Publishing to the
Registry is the first step; getting onto Comfy Cloud is their manual curation. Use
a self-host GPU (above) in the meantime.

## Example workflows

In `examples/` (drag the `.json` onto the ComfyUI canvas):

| File | Needs assets? | Shows |
|---|---|---|
| `difforum_schedule_basic.json` | no | Anim Setup → Schedule → Schedule Info. Runs anywhere, zero downloads. |
| `difforum_audio_reactive.json` | an audio file | Load Audio → Audio Analyzer → two Schedules (bass-zoom `1.0+0.6*low`, beat-spin `beat*30`) → Info. |
| `difforum_feedback_classic.json` | an SD1.5 checkpoint | Full Classic+ video: txt2img frame 0 → Camera + strength Schedule → Feedback Sampler → Save. Set the checkpoint name to one you have. |
| `difforum_camera_warp.json` | an image | Load Image → Camera → Warp (2D/3D) → Preview. See the Deforum warp + occlusion on one frame. |
| `difforum_hybrid_wan_guides.json` | an anchor image | Camera → Guide Builder → guide batch (wire `guide_frames` into your Wan 2.2 VACE graph). |
| `difforum_models_info.json` | nothing | Model Profile (auto-VRAM) + Model Catalog (recipes) + a sampled schedule. Reference card, runs anywhere. |
| `difforum_intuitive_controls.json` | SD1.5 checkpoint | Camera **Move presets** + **Prompt Scenes** → feedback video. |
| `difforum_deluxe_travel_controlnet_video.json` | SD1.5 ckpt + tile ControlNet | Everything: travel + ControlNet + RIFE interpolation → MP4. |
| `difforum_turbo_live.json` | SD-Turbo / LCM ckpt | Low-step (4) fast feedback, the path toward realtime. |
| `difforum_ipadapter_coherent.json` (16:9) | ckpt + IPAdapter + style image | **Style-locked** feedback via IPAdapter → strong coherence, low drift. |
| `difforum_audio_reactive_video.json` (16:9) | ckpt + audio file | **Audio Schedule**: bass pumps zoom, beats pulse strength → MP4. |
| `difforum_animatediff_sd15.json` (16:9) | SD1.5 ckpt + AnimateDiff-Evolved | Prompt Batch → AnimateDiff native motion + Difforum control. |
| `difforum_qrcode_illusion.json` (16:9) | SD1.5 ckpt + QR-Monster ControlNet + pattern image | Locks a spiral/logo/mask in the scene while the loop morphs - hidden-pattern illusions. |
| `difforum_mesmerize_kaleidoscope.json` | SD1.5 checkpoint | **Living kaleidoscope**: in-loop symmetry folds each warped frame, the diffusion heals the seams, Echo Trails smooths the motion. |
| `difforum_realtime_live.json` | SD-Turbo / LCM ckpt | **Native realtime**: Live Sampler internal loop (resident model, warp + kaleidoscope + 1-step re-diffuse) with a live preview in the node. See Realtime below. |
| `difforum_vj_footage.json` | a video clip (no model) | **VJ look for footage**: Load Video → VJ Look (grade + glow + chroma + grain) → Echo Trails → Save Video (MP4). Complete and self-contained, no checkpoint, no external nodes. |

Templates exercise every node. Regenerate with
`python examples/_build_examples.py`.

**Prompt pack:** [`examples/PROMPTS.md`](examples/PROMPTS.md) has 10 ready-to-paste
prompt-travel presets (Cosmic Voyage, Sacred Geometry kaleidoscope, Neon City,
Elemental Shift, Audio-reactive Pulse and more) with suggested camera, strength
and symmetry settings for each.

**Every workflow has a yellow Note on the canvas** listing exactly which models
to download and the folder they go in, so you can set up without guessing.

## Models & downloads

Quick reference for the files the templates ask for (drop each in the folder
shown under your ComfyUI `models/` directory):

| Model | Folder | Where |
|---|---|---|
| SD1.5 checkpoint (DreamShaper 8) | `checkpoints` | civitai.com/models/4384 |
| SDXL checkpoint (Juggernaut XL) | `checkpoints` | civitai.com (search "Juggernaut XL") |
| Flux.1-dev fp8 / schnell | `checkpoints` or `unet` | hf: Comfy-Org/flux1-dev, black-forest-labs/FLUX.1-schnell |
| SD-Turbo / LCM-LoRA (fast) | `checkpoints` / `loras` | hf: stabilityai/sd-turbo, latent-consistency/lcm-lora-sdv1-5 |
| Wan 2.2 5B GGUF + umt5 + wan vae | `unet` / `text_encoders` / `vae` | hf: QuantStack/Wan2.2-TI2V-5B-GGUF, Comfy-Org/Wan_2.1_ComfyUI_repackaged |
| IP-Adapter + CLIP-Vision (ViT-H) | `ipadapter` / `clip_vision` | hf: h94/IP-Adapter |
| QR Code Monster ControlNet | `controlnet` | hf: monster-labs/control_v1p_sd15_qrcode_monster |
| Tile / depth ControlNet (SD1.5) | `controlnet` | hf: comfyanonymous/ControlNet-v1-1_fp16_safetensors |
| AnimateDiff motion module (mm_sd15_v3) | `animatediff_models` | hf: guoyww/animatediff |

Helper nodes some templates use: ComfyUI-VideoHelperSuite (MP4),
ComfyUI-Frame-Interpolation (RIFE), ComfyUI_IPAdapter_plus, ComfyUI-GGUF,
ComfyUI-WanVideoWrapper, ComfyUI-AnimateDiff-Evolved. The **intuitive nodes**
(`Camera Move` presets, `Prompt Scenes`) are wired into `intuitive_controls`,
`turbo_live` and `camera_warp` for an easier start.

## Quality & coherence tips

- **Lock the style with IPAdapter.** Patch the checkpoint's MODEL through
  IPAdapter (a reference image) *before* the Feedback Sampler - it keeps colour
  and style stable across the whole clip (see `difforum_ipadapter_coherent.json`).
- **ControlNet must match the base family** (SD1.5↔SD1.5, SDXL↔SDXL). Feed the
  Feedback Sampler's `control_net` for structure-guided feedback; lower
  `control_strength` (~0.3-0.5) if it feels too rigid.
- **Use a modern base.** SDXL / Flux / SD3.5 in the Feedback Sampler look far
  cleaner than SD1.5 - it's model-agnostic, just swap the checkpoint.
- **Colour drift?** Keep `color_mode = lab` (default) and `color_coherence`
  around 0.7-0.9. **Flicker?** Lower per-frame `strength` (denoise) to ~0.4-0.5.
- **Audio-reactive** is one node: `Difforum · Audio Schedule` turns a band
  (bass/beat/onset) into a curve for zoom, strength, cfg - any schedule input.
- **Illusions / hidden patterns:** feed a grayscale pattern (spiral, logo, mask)
  to `control_image` with **QR Code Monster** ControlNet - it stays locked in the
  scene while the loop morphs around it (`difforum_qrcode_illusion.json`).
- **Mesmerizing / symmetric video:** set the Feedback Sampler's `symmetry` to
  `kaleidoscope` (or `mirror_h/v/quad`) so it folds *inside* the loop - the
  diffusion heals the seams and the pattern grows on itself. Add **Echo Trails**
  after the sampler for smooth motion blur (`difforum_mesmerize_kaleidoscope.json`).
  A slow rotation/zoom on the camera keeps feeding the kaleidoscope new material.

## Realtime / live

The **Difforum Live Sampler** is a native realtime engine. Stock ComfyUI re-runs
the graph per "Queue Prompt", so the Live Sampler keeps the model resident and
runs its own **internal loop**: each tick it warps (camera), folds (symmetry),
re-diffuses 1-2 steps (Turbo/LCM) and color-matches, **streaming a live preview
into the node** as it goes. Queue once and watch it generate, no external runtime.

- Pair with a **few-step model** (SD-Turbo / SDXL-Turbo / LCM / DMD2), 1-2 steps,
  cfg ~1, at 512px for live FPS. The `difforum_realtime_live.json` template wires
  a Turbo checkpoint to the Live Sampler with a kaleidoscope on.
- **Live webcam ("magic mirror"):** set `live_source` to `0` (webcam device) or a
  video path and the loop stylizes the live feed, kaleidoscoping the camera in
  realtime. `source_blend` mixes camera vs feedback (needs `opencv-python`).
- Live output to a VJ app: set `stream_dir` (writes frames to a folder that
  OBS/Resolume/ffmpeg can read live) or `spout_name` (Spout, needs `SpoutGL`).
- For the highest FPS, add **[TensorRT](https://github.com/comfyanonymous/ComfyUI_TensorRT)**
  + **TAESD** (tiny VAE). Map physical knobs/faders with MIDI/OSC.

If you specifically need browser/WebRTC streaming, the same nodes can run under
ComfyStream. Full plan: [REALTIME.md](REALTIME.md).

## Schedule syntax

Deforum-compatible. Each keyframe holds a math expression:

```
0:(0), 60:(0.5*sin(2*pi*t/30)), 120:(1.0)
```

Variables: `t`/`f` (frame), `s` (seconds), `fps`, `max_f`, `pi`, `e`, `tau`,
plus any audio curve (e.g. `amp`, `low`, `mid`, `high`) when an audio input is
connected. Functions: `sin cos tan asin acos atan atan2 sinh cosh tanh abs
sqrt exp log log10 pow floor ceil round min max sign clamp clip lerp
smoothstep`. Ternaries work: `t if t > 30 else 0`.

Expressions are evaluated with a whitelisted AST walker - **no `eval()` of
arbitrary code**, no external dependencies.

Easing between keyframes: `linear`, `ease_in`, `ease_out`, `ease_in_out`, `step`.

## Develop / test

No GPU or ComfyUI needed - 12 suites cover the engine, warp, colour, effects, look, video,
model catalog, and a full end-to-end orchestration (stub diffusion):

```powershell
$py = "D:\ComfyUI-victor\venv\Scripts\python.exe"
foreach ($t in "core","audio","hybrid","models","warp","color","effects","look","video","prompt","plot","integration") {
  & $py "custom_nodes\difforum\tests\test_$t.py"
}
```

`test_integration.py` wires every node together (AnimSetup → Audio → Camera →
Schedule → Feedback Sampler → frames, plus the Guide Builder hybrid path) and
runs the real production code path with a stub model/VAE - so the only thing
swapped for a real render is the checkpoint inside ComfyUI.

After changes, restart ComfyUI fully to reload the nodes (look for
`[Difforum] loaded N nodes` in the console).

## Contributing

Issues and pull requests are welcome. Keep the dependency surface minimal (numpy
in core; torch only where a node needs it) and run the test suites before opening
a PR. Sign your commits off (`git commit -s`, a Developer Certificate of Origin
acknowledgement). See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE). You can use, modify and redistribute Difforum,
including commercially, as long as the copyright and license notice are kept.
