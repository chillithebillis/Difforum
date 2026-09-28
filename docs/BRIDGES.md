# Bridges: video models and camera interchange

Difforum stays the director (camera, timing, prompts) and the video model does
the temporally coherent rendering. The bridges output plain `IMAGE`, `MASK`,
`INT` and `STRING` values, and the LTX bridge calls ComfyUI's own core
`LTXVAddGuide`, so none of them depend on another pack's internals.

## The building blocks

| Node | What it gives you |
|---|---|
| **Guide Frames** | One anchor image carried along the Director's camera path. Revealed areas are filled with neutral gray and marked in a mask (`1 = generate` is the convention VACE / LTX / H3 use). Works in 3D with a depth map. |
| **Keyframes** | Picks keyframes from *any* frame batch (Guide Frames, a Storyboard, a Feedback render) on a model's grid. Outputs the keyframes, their indices, a full-length sparse batch, the first and last frame, and the snapped length. |
| **Fill Reveal (AI)** | Inpaints the masked (revealed) area of the frames a video model will see, with any image model; known pixels stay untouched. |
| **Camera → Prompt** | The camera in words, for prompt-driven models. It uses the Director's blocks when present; otherwise it reads any camera track and names the moves. |

What to feed Keyframes:

- **Guide Frames**: exact geometry, gray where the camera reveals new area. Best
  for moves that stay inside the frame (push-in, orbit, spiral).
- **Feedback render**: every frame is a finished image, revealed areas included.
  Best for pans and long travels.
- **Storyboard**: the cheapest option. Warps only, no diffusion.

## LTX-2 / 2.5

Template: `07_ltx_guides`.

1. **Setup** target `LTX-2 / 2.5` keeps the length on 8k+1 and the size on 32 px.
2. **Director** → **Guide Frames** (or a Feedback render) → **Keyframes** (grid LTX,
   every 1 s; spacing snaps to the 8-frame latent stride).
3. **LTX Guides** takes your LTX positive / negative / VAE and an empty LTX latent
   whose length comes from Keyframes. It adds one guide per keyframe: frame 0 at
   `first_strength` (1.0 locks the look), the rest at `strength` (0.6-0.8 lets LTX
   move naturally between them).
4. Append **Camera → Prompt** (`prompt suffix`) to your LTX prompt, so the text
   agrees with the guides.

To use KJNodes' `LTXVAddGuidesFromBatch` instead, feed it Keyframes'
`sparse_batch`: every frame is black except the keys.

## MiniMax H3 (multi-keyframe guides)

Template: `10_h3_multikeyframe_guides`. This is the official *Multiframe Reference*
setup (ref2va model), with the guide images coming from your camera.

1. **Setup** target `MiniMax H3` locks 24 fps, the 17k+5 length grid and a 32 px size.
2. **Director** → **Guide Frames** (the anchor carried along the camera) →
   **Keyframes** (grid H3, every 2 s).
3. **MiniMax H3 Reference to Video** gets the anchor as `<Picture 1>` (identity
   and style), the prompt from **Camera → Prompt**, and the size and length
   from Setup / Keyframes.
4. **H3 Guides** anchors each keyframe at its frame, using the core
   `MiniMaxH3AddGuide`. `max_guides` (default 4) keeps the first, the last and
   evenly spaced ones in between. Connect an `audio` (and the audio VAE) to
   anchor a soundtrack at frame 0.
5. The rest is the official sampling tail: guider → SamplerCustomAdvanced →
   video + audio decode → Create Video. The Turbo LoRA node is bypassed; enable it
   with Ctrl+B and drop the steps to 4.

## MiniMax H3 (first / last frame)

Template: `08_h3_first_last_frame` (fl2va model), wired end to end.

1. **Setup** target `MiniMax H3` locks 24 fps and the 17k+5 length grid.
2. **H3 Shot** takes the frames (Guide Frames, a Feedback render or a Storyboard)
   and returns, for the chosen `segment`:
   `first_frame`, `last_frame`, `length`, `width`, `height` (32 px grid) and a
   `prompt` built from your shot description plus the camera move of that segment.
3. These feed core **MiniMax H3 Image to Video** directly. The last frame is
   the anchor carried to where the camera ends, so H3 performs the move you drew
   instead of improvising one.

**Longer than one generation (20 s):** `segments` tells you how many there are.
Render segment 0, then set `segment` 1 and use the previous render's last frame
as the next first frame. Segments share their boundary frame, so the joins are
seamless.

## After Effects and Blender

**Camera Export** writes to `ComfyUI/output/difforum/`:

- `.jsx`: *File → Scripts → Run Script File* in After Effects. In 3D mode it
  creates a keyed 3D camera, and layers at z = 0 sit at the render's reference depth.
  In 2D mode it creates a null whose transform reproduces the image motion
  exactly; parent your footage or titles to it.
- `_blender.py`: run it from Blender's Text Editor. It creates an animated camera
  and matches fps, resolution and frame range. `blender_unit_scale` sets how many
  metres one Difforum unit is.
- `.json`: camera-to-world matrices plus the focal length per frame (OpenCV
  convention, world = first camera), for Nuke, Houdini, Unreal or your own tools.

Set `translation_scale` to the same value as the sampler so the exported move
matches the render.

**Camera Import** goes the other way: block a move in Blender or After Effects,
export it with `tools/blender/difforum_export_camera.py` or
`tools/aftereffects/Difforum Export Camera.jsx`, drop the `.json` into
`ComfyUI/input/`, and it drives the Storyboard, Feedback Sampler or Guide Frames
like any Difforum camera. `retime = match frames` resamples it to your clip length.

Round trips (Difforum → JSON / Blender / AE → Difforum) are covered by the test
suite. After Effects rotations use X/Y/Z Rotation composed as Rz·Ry·Rx. If a
move looks mirrored in your AE version, check that the camera has no Orientation
values.

## Looks

The Director's `look` is one aesthetic choice for every renderer. On the
Feedback Sampler it sets colour lock, detail, grain and adds energy; on H3 /
LTX it appends a style sentence (Camera → Prompt and H3 Shot, `include_look`).

To push a model like H3 toward the Deforum or AnimateDiff look, use template
`11_h3_deforum_look`: a turbo Feedback Sampler pass renders the shot in the
chosen look, Keyframes takes one per second, and H3 Guides anchors them - H3
animates between Deforum frames. More guides stay closer to the look pass;
fewer give H3 more freedom.

### Upscaling H3: latent first, pixels last

| stage | where | what it adds |
|---|---|---|
| **H3 Latent Upscale (x2)** | inside H3, before decoding | H3 renders at 640 px; the Minimax H3 Latent Upscaler (3D) doubles the latent and H3 re-samples 4 steps (sigmas from 0.63) at full size. The new detail is generated by H3 itself, so it stays consistent from frame to frame, and the 5B VAE is never round-tripped. Saves time, not VRAM. |
| **Upscale 2K** | after decoding (and after Restyle) | a per-frame pixel upscaler to the delivery size. Sharper, but it invents no motion-aware detail. |

With the Turbo LoRA on, set the refine sigmas to `0.6316, 0.3158, 0.0000`. Higher first
sigma (0.8-0.9) = more new detail but the motion can drift; 0.5 = keeps the render.
Restyle runs on the decoded 2x render, then Upscale 2K.

### How the look gets onto H3 (and what each stage does)

H3 makes smooth, coherent motion and smooths away the boiling texture that made
Deforum, AnimateDiff and Disco Diffusion feel alive. Three stages bring it back,
each one a block you can switch off:

| stage | what it does | what you see |
|---|---|---|
| **Look pass** (template 11) | a turbo Feedback Sampler renders the shot the Deforum way; its frames guide H3 | the morphs and dissolves happen inside H3's motion |
| **Restyle** | an image model re-paints every H3 frame (img2img at `denoise` 0.3-0.6), mixing in the previous painted frame moved along the clip's optical flow | the actual Deforum / AnimateDiff texture, following H3's motion |
| **Look Mix** | compositing only: high-frequency detail, colour or cut-ins from the look pass | extra grain, strobe, stutter |

Restyle styles:

| style | feedback | seed | colour | reads as |
|---|---|---|---|---|
| `clean restyle` | 0 | fixed | held | a steady painted version of the clip |
| `animatediff boil` | 0.15 | per frame | mostly held | texture re-rolling every frame |
| `deforum morph` | 0.55 | fixed | loose | smear and morph along the motion |
| `disco flicker` | 0.25 | per frame, +25 % denoise | free | high-energy flicker |
| `custom` | your `feedback` | your `seed_mode` | your `color_hold` | |

Look Mix on its own does **not** re-paint anything: it layers texture from a
look pass that was rendered separately, so where the two drift apart the texture
swims. Use it on top of Restyle, or for flicker cuts and stutter.

### Getting closer to the real AnimateDiff / Deforum look

Per-frame img2img only stylises what is already there: at low `denoise` the clip
barely changes, at high `denoise` the structure falls apart. The classic looks
came from two things Restyle alone does not have:

| look | what actually produces it | where in Difforum |
|---|---|---|
| AnimateDiff | a motion module that denoises 16 frames together (sliding context), plus ControlNet (depth / canny / lineart) from the source | template 14 (AnimateDiff-Evolved). Turn the AnimateDiff block off for per-frame flicker |
| Deforum hybrid | high denoise with the structure held by ControlNet, plus optical-flow feedback of the previous frame | template 13: Restyle `deforum morph` + Structure (ControlNet) block, denoise 0.6-0.8 |

Template 14 needs [ComfyUI-AnimateDiff-Evolved](https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved),
an SD 1.5 checkpoint, a motion module (`v3_sd15_mm.ckpt`, or AnimateLCM for fewer
steps) and SD 1.5 ControlNets (`control_v11f1p_sd15_depth`, `control_v11p_sd15_canny`).
The ControlNets go through [ComfyUI-Advanced-ControlNet](https://github.com/Kosinkadink/ComfyUI-Advanced-ControlNet)
(`Load Advanced ControlNet Model` + `Apply Advanced ControlNet`): the core ControlNet
nodes fail inside AnimateDiff's sliding context window with *"Control type ControlNet
may not support required features for sliding context window"*.
Keep it at about 0.3 MP (SD 1.5 native size) and let Upscale 2K do the rest.
Template 13 needs an SDXL union ControlNet (`xinsir-controlnet-union-sdxl-1.0-promax`)
matching the SDXL / turbo image model.

### AnimateLCM: speed and quality (template 15)

AnimateLCM is a consistency-distilled AnimateDiff: its motion module plus its
LoRA render in 4-10 steps instead of 20-30. Template 15 spends the time saved on
a second pass at 1.5x, which is where SD 1.5 gets its detail back.

| setting | base pass | hi-res pass |
|---|---|---|
| motion module | `AnimateLCM_sd15_t2v.ckpt`, beta_schedule `lcm avg(sqrt_linear,linear)` | same model |
| LoRA | `AnimateLCM_sd15_t2v_lora.safetensors` at 0.8-1.0 | same |
| size | ~0.3 MP | latent x1.5 (`bislerp`) |
| sampler / scheduler | `lcm` / `sgm_uniform` | `lcm` / `sgm_uniform` |
| steps / cfg | 8 / 1.8 | 6 / 1.5 |
| denoise | 0.65-0.8 | 0.4-0.5 |

- cfg above ~2 burns the image with LCM; below 1.3 the negative prompt stops working.
- More steps do not add much past 10; raise the hi-res pass instead.
- Switch **Hi-res pass** off to preview a clip quickly, on for the final render.
- Both files are in the `wangfuyun/AnimateLCM` repository on Hugging Face: the
  `.ckpt` goes in `models/animatediff_models`, the LoRA in `models/loras`.

## Multikeyframing and previz

Put markers on the Director's **Keys** track (click the lane, drag to move,
alt-click to remove, name each one in the inspector). **Keyframe Images** pins
one picture per key (or takes times such as `0, 4s, 9.5s`) and outputs
`keyframes` + `indices`, which go to:

- **Feedback Sampler** `key_images` / `key_indices`: the travel steers toward each
  picture during `key_approach` frames, `key_pull` 1.0 lands on it exactly;
- **H3 Guides** / **LTX Guides**: anchored at those frames;
- **Animatic**: shown at their moments, with a flash.

The **Animatic** is the previz of the whole shot (low resolution, a few seconds
even for a minute of footage). Press **Previz only** on the Director to mute every
render output, queue, check, then press it again to render.

## Keyframes for video models

The guide keyframes are what H3 / LTX copy: their sharpness, palette and texture.

**How many.** The Director's **Keys** track marks exact moments (with Keyframe
Images). **Keyframes** samples a clip every `every_seconds` (or at explicit
`indices` such as `0, 48, 96, -1`) on the model's latent grid. **H3 Guides**
keeps up to `max_guides` of them: first, last, and the best spread in time. More
guides follow the frames closely; fewer leave more to the model.

**Quality.** Guide Frames warps one image along the camera, so keyframes soften as
the camera travels and the uncovered areas are empty. The H3 templates run three
switchable blocks before H3 Guides:

| block | what it does |
|---|---|
| **Depth (Depth Anything 3)** | core depth estimation on the anchor, so 3d moves have real parallax |
| **Fill Reveal (AI)** | an image model paints the uncovered areas (inpainting checkpoints seam best) |
| **Keyframe Polish** | Restyle `clean restyle` at denoise ~0.35 re-paints each keyframe: detail back, composition kept |

Consistent keyframes (same model, prompt and seed) interpolate better than a mix of
styles. Hand-made stills on the Keys track beat warped frames for important moments.

## H3 prompts from the timeline

Camera → Prompt (`format = H3 structured`) and H3 Shot (`prompt_style = H3
structured`) write the Director timeline in MiniMax H3's native prompt format:

| timeline | prompt |
|---|---|
| look | style that opens `[Shot 1]` |
| scenes | what is on screen; a transformation in one take, or a new `[Shot N]` at its cut time with `cuts` on |
| camera blocks | H3 camera vocabulary (push in, truck, arc, pedestal...) with amplitude and speed |
| key labels | the event at that moment |
| `soundscape` / `music` | `overall_soundscape` / `non_diegetic_music` |

`h3_mode` adds the I2VA or FL2VA alignment line, or writes the full-reference
sections for ref2va with H3 Guides. Scene prompts and key labels work best as
things that can be seen or heard.

## Depth

Set the Director to **3d** and connect a depth map (white = near) to Guide Frames
or the Feedback Sampler. Templates 03, 08 and 10 use the core Depth Anything 3
nodes (`depth_anything_3_mono_large` in `models/geometry_estimation`, `v2_style`
render). The Feedback Sampler also outputs the tracked depth per frame for
compositing. Lower `translation_scale` if the parallax is too strong.
