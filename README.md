# Difforum

**Camera direction and timeline orchestration for ComfyUI.**

Direct a shot on a visual timeline, with scenes, camera moves and an energy
curve, and you never write a camera expression. Render it with a
Deforum-style feedback sampler on any image model, hand the same direction to
**LTX-2 / 2.5** or **MiniMax H3**, and send the camera to **After Effects** or
**Blender** for compositing.

![The Director timeline](docs/media/director.jpg)

![Kaleidoscope feedback render](docs/media/promo.webp)

---

## Why Difforum

Video models are good at rendering and not so good at being directed. Deforum
was good at direction (math-driven camera, keyframes, prompt travel, audio),
but it was locked to 2023 img2img. Difforum keeps the direction and lets you
pick how to render it:

| Direct it once… | …then render it with |
|---|---|
| **Director** timeline: scenes + mood, camera moves from a visual picker, a drawn energy curve, audio reactions per move | **Feedback Sampler**: the Deforum look on SD1.5 / SDXL / Flux / SD3.5 / turbo distills |
| or **Camera (keys)** / **Camera (expressions)** if you prefer typing | **LTX Guides**: keyframes into an LTX-2 / 2.5 latent |
| or **Camera Import** from Blender / After Effects | **H3 Shot**: first/last frame + length + camera prompt for MiniMax H3 |
| | **Live Sampler**: realtime, webcam, Spout / OBS output |
| | **Camera Export**: the same camera in AE (.jsx) and Blender (.py) |

## Quick start (60 seconds, no model)

1. Install (below), restart ComfyUI.
2. **Workflow → Browse Templates → Difforum → `01_storyboard_no_model`**.
3. Load any image, press ▶ on the Director to preview the camera, then Queue.
   The Storyboard renders the whole move in about a second, and **Camera → Prompt**
   writes it out in words.

When the motion feels right, open `02_feedback_sdxl` to render it with a model.

## The Director

Three tracks, all edited with the mouse:

- **Scenes**: a prompt and a mood (calm, build, tense, climax, resolve, dream)
  per block. Moods set sensible defaults for energy, speed and lens.
- **Camera**: blocks that do not have to line up with the scenes. Pick from 25 moves
  (zoom, pan, roll, dolly, orbit, tilt, crane, dolly zoom, spiral, vortex, sway,
  breathe, drift, handheld, shake), then set speed, amount, lens, easing and an
  **audio reaction** (pulse on beat, shake on hits, bass drives speed).
- **Energy**: the denoise curve, which sets how much each moment gets re-imagined.
  It starts from the moods; click it to draw your own.

The preview plays the camera the renderer will actually use: it is computed by the
same Python engine, not by an approximation in the browser. One `direction`
wire carries the camera, energy, cfg and prompt travel to the sampler or the
bridges.

## Nodes (25)

| Group | Nodes |
|---|---|
| Setup | **Setup**: duration in seconds/frames, aspect, and snapping to the target model's grid (LTX 8k+1, H3 17k+5 @ 24 fps, Wan 4k+1) |
| Direction | **Director (timeline)**, **Camera (keys)**, **Camera (expressions)**, **Storyboard** |
| Curves & prompts | **Schedule**, **Schedule Plot**, **Audio Analyzer**, **Audio Curve**, **Prompt Travel** |
| Render | **Feedback Sampler**, **Live Sampler**, **Render Options** |
| Video model bridges | **Guide Frames**, **Keyframes**, **Camera → Prompt**, **LTX Guides**, **H3 Shot** |
| Export | **Camera Export (AE / Blender / JSON)**, **Camera Import** |
| Post | **Loop**, **Symmetry**, **Echo Trails**, **Flow Stabilize**, **Detail Guard** |

Full reference, generated from the code: [docs/NODES.md](docs/NODES.md).

## Templates

All of these are in ComfyUI's template browser, under Difforum.

| Template | Needs |
|---|---|
| `01_storyboard_no_model` | an image |
| `02_feedback_sdxl` | an SDXL (or SD1.5 / Flux) checkpoint |
| `03_parallax_3d_depth` | + [ComfyUI-DepthAnythingV2](https://github.com/kijai/ComfyUI-DepthAnythingV2) |
| `04_audio_reactive` | + an audio file |
| `05_live_turbo` | SDXL-Turbo / SD-Turbo / LCM |
| `06_seamless_loop` | a checkpoint |
| `07_ltx_guides` | your LTX-2 / 2.5 graph |
| `08_h3_first_last` | your MiniMax H3 graph |
| `09_camera_to_ae_blender` | nothing; writes .jsx / .py / .json |

## What the engine does for quality

- **Depth follows the image.** In 3D mode the depth map is re-projected with every
  warp, so parallax stays locked to what is on screen and doesn't drift back to frame 0.
  The tracked depth comes out of the sampler, ready for compositing.
- **No silent freezes.** A 3D move without a depth map runs as pseudo-3D (dolly
  becomes zoom, orbit becomes pan), and the node says so.
- **Cadence without pops.** Only every Nth frame is diffused. The frames in between
  are a crossfade of the previous key warped forward and the next key warped
  back, so a cadence of 2-3 costs almost nothing in smoothness.
- **Revealed areas are repainted.** The warp's occlusion mask gets extra noise, so the
  sampler invents new content at the edges instead of smearing them.
- **Colour follows the scenes.** The anchor re-locks at every prompt scene and releases
  during transitions. Choose `first` for the 0.x behaviour.
- **Deterministic on every device.** The 3D z-buffer resolves identically on CUDA, MPS
  and CPU.
- **Light on memory.** Prompt travel is blended lazily, and the Live Sampler keeps a
  ring buffer.

Speed and quality levers, Apple Silicon notes and measured numbers are in
[docs/PERFORMANCE.md](docs/PERFORMANCE.md).

## Bridges and camera interchange

[docs/BRIDGES.md](docs/BRIDGES.md) explains how to drive LTX-2 / 2.5 with Difforum
keyframes, how to build MiniMax H3 first/last-frame shots (including clips longer
than 20 s), and how to round-trip the camera through After Effects and Blender.
Helper scripts for exporting a camera *from* Blender or AE are in `tools/`.

## Install

**ComfyUI Manager:** search for **Difforum**, or *Install via Git URL* with this repository.

**Manual:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/chillithebillis/Difforum.git difforum
```

Restart ComfyUI. The console prints `[Difforum] 25 nodes ready`. The only
runtime dependency is numpy, which ships with ComfyUI. OpenCV (Flow Stabilize, Loop
crossfade, live webcam) is present in most installs.

## Upgrading from 0.x

Old workflows still open: every 0.x node is kept, hidden from search, under
**Difforum/legacy**. [docs/MIGRATION.md](docs/MIGRATION.md) maps each old node
to its replacement. Legacy nodes are removed in 2.0.

## Develop

```bash
pip install torch numpy opencv-python-headless pytest ruff
pytest                                  # no GPU or ComfyUI needed
python tools/build_workflows.py         # templates are generated from the node definitions
python tools/build_docs.py              # docs/NODES.md is generated too
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## License

MIT, see [LICENSE](LICENSE).
