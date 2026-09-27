# Node reference

Generated from the code by `tools/build_docs.py`. Every node also shows this text as its description inside ComfyUI.

## 1 · Setup

### Difforum · Setup

`Difforum_Setup`

Duration, framing and the grid of the model you will render with.

Say how long the clip is (seconds or frames) and how it is framed (aspect +
long edge); the resolution is derived, snapped to what the target model's
VAE accepts, and the frame count is snapped to the model's length grid
(LTX 8k+1, MiniMax H3 17k+5 at 24 fps, Wan 4k+1). Every other Difforum node
reads these params, so the whole graph agrees on one timeline.

| input | type | default | notes |
|---|---|---|---|
| `target` | choice (feedback (SD/SDXL/Flux), LTX-2 / 2.5, MiniMax H3, Wan 2.x) | feedback (SD/SDXL/Flux) | Snaps size and length to what this model family accepts. |
| `duration_mode` | choice (seconds, frames) | seconds |  |
| `duration` | FLOAT | 5.0 |  |
| `fps` | FLOAT | 24.0 |  |
| `aspect` | choice (16:9 landscape, 9:16 vertical, 1:1 square, 4:5 social portrait, 4:3 classic, 3:4 portrait, ...) | 16:9 landscape |  |
| `long_edge` | INT | 768 | Pixels on the longer side before snapping. |
| `seed` | INT | 0 |  |
| `custom_aspect_w` *(optional)* | FLOAT | 16.0 |  |
| `custom_aspect_h` *(optional)* | FLOAT | 9.0 |  |
| `max_megapixels` *(optional)* | FLOAT | 0.0 | 0 = no cap. Otherwise shrink to fit (e.g. 0.35 for a 16 GB Mac). |

**Outputs:** `params` (DIFFORUM_PARAMS), `width` (INT), `height` (INT), `frames` (INT), `fps` (FLOAT), `info` (STRING)

## 2 · Direction

### Difforum · Director (timeline)

`Difforum_Director`

Direct the whole clip on a visual timeline.

Three tracks, edited with the mouse: **Scenes** (prompt + mood), **Camera**
(pick a move, set speed / amplitude / lens / easing / audio reaction) and
**Energy** (the denoise curve - how much each moment is re-imagined). The
preview panel plays the camera back with the same engine that renders it.

One `direction` wire carries camera, strength, cfg and prompt travel into
the Feedback Sampler or a video-model bridge. The separate outputs are
there for custom graphs.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  |  |
| `timeline` | STRING | {"version": 2, "scenes": [{"start": 0... |  |
| `camera_mode` | choice (2d, 3d) | 2d | 3d = real parallax when a depth map reaches the sampler (pseudo-3D otherwise). |
| `look` | choice (cinematic, documentary, deforum_morph, animatediff_dream, psychedelic, music_video, ...) | cinematic | Render aesthetic. Feedback Sampler: colour lock, detail, grain and energy. H3 / LTX: a look sentence added to the prompt. |
| `transition` | FLOAT | 1.0 | How much of each camera block is spent easing into the next. 0 = hard cuts. |
| `camera_scale` | FLOAT | 1.0 | Master multiplier on every move. |
| `energy_bias` | FLOAT | 0.0 | Shifts the whole denoise curve up or down. |
| `variation` | FLOAT | 0.0 | 0 = exactly as drawn. Above 0, seeded nudges per block. |
| `variation_seed` | INT | 0 |  |
| `clip` *(optional)* | CLIP |  | Connect to encode the scene prompts (prompt travel). |
| `audio` *(optional)* | DIFFORUM_AUDIO |  | From Audio Analyzer - enables per-block audio reactions. |

**Outputs:** `direction` (DIFFORUM_DIRECTION), `camera` (DIFFORUM_CAMERA), `strength` (DIFFORUM_SCHEDULE), `prompts` (DIFFORUM_PROMPT), `camera_text` (STRING), `info` (STRING)

### Difforum · Camera (keys)

`Difforum_Camera`

Keyframed camera as text, one key per line:

    frame: move [speed] [intensity] [lens] [ease]

Moves blend into each other (`transition` 1.0 = continuous crane move,
0.0 = hard cuts). `loop_mode` makes the path periodic over `cycle_frames`
so the clip loops without a crossfade. The Director timeline writes the
same thing visually - use this node when you prefer typing.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  |  |
| `keys` | STRING | 0: zoom_in 0.6 0.5 40 ease_in_out 30:... |  |
| `mode` | choice (2d, 3d) | 2d |  |
| `transition` | FLOAT | 1.0 |  |
| `speed` | FLOAT | 1.0 |  |
| `loop_mode` | choice (harmonic, zero_mean, off) | off |  |
| `cycle_frames` | INT | 0 | 0 = whole clip is one cycle. |
| `harmonics` | INT | 3 |  |
| `audio` *(optional)* | DIFFORUM_AUDIO |  |  |

**Outputs:** `camera` (DIFFORUM_CAMERA), `cycle_frames` (INT), `info` (STRING)

### Difforum · Camera (expressions)

`Difforum_CameraExpr`

Deforum-style camera from math expressions, per axis.

`0:(0), 60:(2*sin(2*pi*t/30))`, with `t` (frame), `s` (seconds) and any audio
band (`low`, `beat`...) when audio is connected. Each axis also has an
optional schedule socket (e.g. an Audio Curve) that overrides its text.
Values are per-frame increments; zoom is a per-frame scale factor.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  |  |
| `mode` | choice (2d, 3d) | 2d |  |
| `fov` | FLOAT | 40.0 |  |
| `translation_x` | STRING | 0:(0) |  |
| `translation_y` | STRING | 0:(0) |  |
| `translation_z` | STRING | 0:(0) |  |
| `rotation_3d_x` | STRING | 0:(0) |  |
| `rotation_3d_y` | STRING | 0:(0) |  |
| `rotation_3d_z` | STRING | 0:(0.3) |  |
| `zoom` | STRING | 0:(1.0 + 0.004*sin(2*pi*t/48)) |  |
| `audio` *(optional)* | DIFFORUM_AUDIO |  |  |
| `translation_x_curve` *(optional)* | DIFFORUM_SCHEDULE |  |  |
| `translation_y_curve` *(optional)* | DIFFORUM_SCHEDULE |  |  |
| `translation_z_curve` *(optional)* | DIFFORUM_SCHEDULE |  |  |
| `rotation_3d_x_curve` *(optional)* | DIFFORUM_SCHEDULE |  |  |
| `rotation_3d_y_curve` *(optional)* | DIFFORUM_SCHEDULE |  |  |
| `rotation_3d_z_curve` *(optional)* | DIFFORUM_SCHEDULE |  |  |
| `zoom_curve` *(optional)* | DIFFORUM_SCHEDULE |  |  |

**Outputs:** `camera` (DIFFORUM_CAMERA), `info` (STRING)

### Difforum · Storyboard

`Difforum_Storyboard`

The whole clip warped by the camera, no diffusion - in about a second.

Same engine as the Feedback Sampler with the sampler switched off, so what
you see is exactly the motion the render will get: contact sheet, the
frames themselves, the camera path from above, and a pacing verdict.

| input | type | default | notes |
|---|---|---|---|
| `init_image` | IMAGE |  |  |
| `every_nth` | INT | 8 |  |
| `columns` | INT | 6 |  |
| `preview_scale` | FLOAT | 0.5 |  |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  |  |
| `params` *(optional)* | DIFFORUM_PARAMS |  |  |
| `camera` *(optional)* | DIFFORUM_CAMERA |  |  |
| `depth` *(optional)* | IMAGE |  |  |
| `symmetry` *(optional)* | choice (none, mirror_h, mirror_v, mirror_quad, kaleidoscope) | none |  |

**Outputs:** `sheet` (IMAGE), `frames` (IMAGE), `camera_path` (IMAGE), `info` (STRING)

## 3 · Curves & Prompts

### Difforum · Schedule

`Difforum_Schedule`

A per-frame curve from Deforum keyframe syntax: `0:(0.4), 48:(0.6+0.1*sin(t/8))`.

Variables: t/f (frame), s (seconds), fps, max_f, and every audio band when
an Audio Analyzer is connected. Easing applies between keyframes.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  |  |
| `schedule` | STRING | 0:(0.45), 60:(0.6), 119:(0.45) |  |
| `easing` | choice (linear, ease_in, ease_out, ease_in_out, step) | ease_in_out |  |
| `audio` *(optional)* | DIFFORUM_AUDIO |  |  |

**Outputs:** `schedule` (DIFFORUM_SCHEDULE), `info` (STRING)

### Difforum · Schedule Plot

`Difforum_SchedulePlot`

Draw any schedule as an image (and a one-line summary).

| input | type | default | notes |
|---|---|---|---|
| `schedule` | DIFFORUM_SCHEDULE |  |  |
| `width` | INT | 512 |  |
| `height` | INT | 200 |  |

**Outputs:** `plot` (IMAGE), `info` (STRING)

### Difforum · Audio Analyzer

`Difforum_AudioAnalyzer`

Audio -> per-frame bands: amp, low, mid, high, onset, beat (0..1).

Use the bands in any expression (`1 + 0.05*low`), in a Director camera
block's audio reaction, or turn one into a curve with Audio Curve.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  |  |
| `audio` | AUDIO |  |  |
| `smoothing` | FLOAT | 0.25 |  |
| `beat_sensitivity` | FLOAT | 1.5 |  |
| `offset_seconds` | FLOAT | 0.0 | Start reading the track here (sync to an edit). |

**Outputs:** `audio_curves` (DIFFORUM_AUDIO), `bands_plot` (IMAGE), `info` (STRING)

### Difforum · Audio Curve

`Difforum_AudioCurve`

One audio band -> a ready curve: e.g. bass-pumped energy
(`low`, add, base 0.45, amount 0.2) or a beat-pulsed cfg.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  |  |
| `audio_curves` | DIFFORUM_AUDIO |  |  |
| `source` | choice (amp, low, mid, high, onset, beat) | low |  |
| `mode` | choice (add, subtract, multiply) | add |  |
| `base` | FLOAT | 0.45 |  |
| `amount` | FLOAT | 0.2 |  |
| `smoothing` | FLOAT | 0.2 |  |

**Outputs:** `schedule` (DIFFORUM_SCHEDULE)

### Difforum · Prompt Travel

`Difforum_PromptTravel`

Prompt travel from `frame: prompt` lines (or `2.5s: prompt`), blended
per frame. Encodes each prompt once and blends lazily, so long clips on big
text encoders stay light. `build_batched` adds one batched conditioning
(frame i = prompt i) for batch samplers such as AnimateDiff.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  |  |
| `clip` | CLIP |  |  |
| `prompts` | STRING | 0: a serene misty forest, soft light ... |  |
| `easing` | choice (linear, ease_in, ease_out, ease_in_out, step) | ease_in_out |  |
| `build_batched` *(optional)* | BOOLEAN | False | Also build the batched output (costs RAM on long clips). |

**Outputs:** `prompts` (DIFFORUM_PROMPT), `first_frame_cond` (CONDITIONING), `batched` (CONDITIONING), `info` (STRING)

## 4 · Render

### Difforum · Feedback Sampler

`Difforum_FeedbackSampler`

The Deforum look on any image model: every frame is the last one,
moved by the camera and re-imagined by the sampler.

Plug a Director `direction` wire and a first frame; that is the whole
setup. Works with SD1.5, SDXL, Flux, SD3.5 and turbo/LCM/DMD2 distills
(plain MODEL / VAE / CONDITIONING). Connect a depth map (e.g. Depth
Anything V2 on the first frame) and set the Director to 3d for real
parallax - the depth then follows the image frame by frame.

Outputs the frames, the tracked depth per frame (for comp / relight /
video-model guides) and a run report.

| input | type | default | notes |
|---|---|---|---|
| `model` | MODEL |  |  |
| `positive` | CONDITIONING |  | Used when no prompt travel is connected. |
| `negative` | CONDITIONING |  |  |
| `vae` | VAE |  |  |
| `init_image` | IMAGE |  | The first frame. |
| `steps` | INT | 20 |  |
| `cfg` | FLOAT | 6.0 |  |
| `sampler_name` | choice (euler, lcm) | euler |  |
| `scheduler` | choice (normal, sgm_uniform) | normal |  |
| `cadence` | INT | 1 | Diffuse every Nth frame; the rest are crossfaded warps. ~N x faster. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  |  |
| `options` *(optional)* | DIFFORUM_OPTIONS |  | Difforum · Render Options, for manual control. |
| `params` *(optional)* | DIFFORUM_PARAMS |  |  |
| `camera` *(optional)* | DIFFORUM_CAMERA |  |  |
| `strength` *(optional)* | DIFFORUM_SCHEDULE |  | Overrides the Director's energy curve. |
| `cfg_curve` *(optional)* | DIFFORUM_SCHEDULE |  |  |
| `prompts` *(optional)* | DIFFORUM_PROMPT |  |  |
| `depth` *(optional)* | IMAGE |  | Depth of the first frame (white = near), or a per-frame batch. |
| `control_net` *(optional)* | CONTROL_NET |  |  |
| `control_image` *(optional)* | IMAGE |  |  |
| `control_strength` *(optional)* | FLOAT | 0.6 |  |
| `energy` *(optional)* | FLOAT | 0.5 | Denoise when neither a Director nor a strength curve is connected. |

**Outputs:** `frames` (IMAGE), `depth` (IMAGE), `report` (STRING)

### Difforum · Live Sampler

`Difforum_LiveSampler`

Realtime Difforum: queue once and watch it play inside the node.

Pair with a 1-4 step model (SD-Turbo, SDXL-Turbo, LCM, DMD2) at ~512 px.
`live_source` turns it into a magic mirror (webcam "0" or a video path);
`stream_dir` / `spout_name` feed OBS, Resolume or TouchDesigner. Only the
last `keep_frames` frames are returned, so long sessions do not fill RAM.

| input | type | default | notes |
|---|---|---|---|
| `model` | MODEL |  |  |
| `positive` | CONDITIONING |  |  |
| `negative` | CONDITIONING |  |  |
| `vae` | VAE |  |  |
| `init_image` | IMAGE |  |  |
| `run_frames` | INT | 480 |  |
| `steps` | INT | 2 |  |
| `cfg` | FLOAT | 1.2 |  |
| `sampler_name` | choice (euler, lcm) | lcm |  |
| `scheduler` | choice (normal, sgm_uniform) | sgm_uniform |  |
| `cadence` | INT | 1 |  |
| `target_fps` | FLOAT | 0.0 | 0 = as fast as possible. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  |  |
| `options` *(optional)* | DIFFORUM_OPTIONS |  |  |
| `params` *(optional)* | DIFFORUM_PARAMS |  |  |
| `camera` *(optional)* | DIFFORUM_CAMERA |  |  |
| `strength` *(optional)* | DIFFORUM_SCHEDULE |  |  |
| `prompts` *(optional)* | DIFFORUM_PROMPT |  |  |
| `depth` *(optional)* | IMAGE |  |  |
| `energy` *(optional)* | FLOAT | 0.5 |  |
| `loop_camera` *(optional)* | BOOLEAN | True |  |
| `live_source` *(optional)* | STRING |  | '' off, '0' webcam, or a video path. |
| `source_blend` *(optional)* | FLOAT | 0.9 |  |
| `stream_dir` *(optional)* | STRING |  | Folder to write live PNG frames to. |
| `spout_name` *(optional)* | STRING |  |  |
| `keep_frames` *(optional)* | INT | 240 |  |
| `live_preview` *(optional)* | BOOLEAN | True |  |

**Outputs:** `frames` (IMAGE), `report` (STRING)

### Difforum · Render Options

`Difforum_RenderOptions`

Everything the samplers can fine-tune, kept off the sampler itself.

Without this node the look (colour lock, sharpen, grain) comes from the
Director's look and the rest uses sensible defaults. Connect it to take
manual control: colour anchoring, detail guard, in-loop symmetry, 3D depth
calibration, noise seeding and chunked rendering.

| input | type | default | notes |
|---|---|---|---|
| `color_coherence` | FLOAT | 0.8 | How hard colours are held to the anchor. |
| `color_mode` | choice (lab, rgb, none) | lab |  |
| `anchor_mode` | choice (scene, first, rolling, none) | scene | scene = re-anchor colour at each prompt scene; first = hold frame 0; rolling = previous key; none = free. |
| `sharpen` | FLOAT | 0.2 |  |
| `noise` | FLOAT | 0.02 |  |
| `hole_noise` | FLOAT | 0.25 | Extra noise where the camera reveals new area, so it is repainted. |
| `symmetry` | choice (none, mirror_h, mirror_v, mirror_quad, kaleidoscope) | none |  |
| `symmetry_segments` | INT | 6 |  |
| `border` | choice (reflection, border, zeros) | reflection |  |
| `depth_tracking` | choice (follow, static) | follow | follow = depth is re-projected with the image every frame. |
| `translation_scale` | FLOAT | 1.0 | 3D motion strength vs. the depth map's range. |
| `near` | FLOAT | 1.0 |  |
| `far` | FLOAT | 100.0 |  |
| `invert_depth` | BOOLEAN | False | Enable if near things are dark in your depth map. |
| `seed_mode` | choice (fixed, increment) | fixed | fixed = same noise every frame (calmer texture). |
| `start_frame` | INT | 0 | Feedback Sampler: render a chunk; init_image is the frame at start_frame. |
| `end_frame` | INT | 0 | 0 = to the end. |

**Outputs:** `options` (DIFFORUM_OPTIONS)

## 5 · Video model bridges

### Difforum · Guide Frames

`Difforum_GuideFrames`

Warp one anchor image along the Director's camera path.

Produces a guide video whose motion is exactly the camera you drew, for
video models that take control/guide frames (LTX guides, Wan VACE, H3
first/last frame). Revealed areas are filled with neutral gray and marked
in the mask, using the convention the target expects.

| input | type | default | notes |
|---|---|---|---|
| `anchor_image` | IMAGE |  |  |
| `mask_convention` | choice (1 = generate (VACE / LTX / H3), 1 = keep) | 1 = generate (VACE / LTX / H3) |  |
| `hole_fill` | choice (gray, stretch edge, black) | gray |  |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  |  |
| `params` *(optional)* | DIFFORUM_PARAMS |  |  |
| `camera` *(optional)* | DIFFORUM_CAMERA |  |  |
| `depth` *(optional)* | IMAGE |  | Depth of the anchor, for real parallax in 3d mode. |
| `translation_scale` *(optional)* | FLOAT | 1.0 |  |

**Outputs:** `guide_frames` (IMAGE), `masks` (MASK), `info` (STRING)

### Difforum · Keyframes

`Difforum_Keyframes`

Pick keyframes out of any frame batch (a Storyboard, a Feedback render,
guide frames) for a video model, on that model's frame grid.

Outputs the keyframes, their indices, a full-length *sparse* batch (black
except at keyframes - what `LTXVAddGuidesFromBatch` expects), the first and
last frame (for first-last-frame models) and the snapped clip length.

| input | type | default | notes |
|---|---|---|---|
| `frames` | IMAGE |  |  |
| `grid` | choice (LTX-2 / 2.5 (8k+1), MiniMax H3 (17k+5), Wan 2.x (4k+1), any) | LTX-2 / 2.5 (8k+1) |  |
| `every_seconds` | FLOAT | 1.0 | Spacing between keyframes. 0 = first and last only. |
| `fps` | FLOAT | 24.0 |  |
| `indices` *(optional)* | STRING |  | Explicit frame list, e.g. 0, 48, 96, -1 (overrides spacing). |
| `masks` *(optional)* | MASK |  | Guide Frames masks, so Fill Reveal can repaint the keyframes. |

**Outputs:** `keyframes` (IMAGE), `indices` (STRING), `sparse_batch` (IMAGE), `first_frame` (IMAGE), `last_frame` (IMAGE), `length` (INT), `key_masks` (MASK), `info` (STRING)

### Difforum · Camera → Prompt

`Difforum_CameraPrompt`

The camera, in words - for prompt-driven video models (MiniMax H3,
LTX, Seedance, Veo...). Uses the Director's blocks when available, or
reads any camera track and names its moves.

| input | type | default | notes |
|---|---|---|---|
| `format` | choice (sentence, timed lines, prompt suffix) | sentence |  |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  |  |
| `params` *(optional)* | DIFFORUM_PARAMS |  |  |
| `camera` *(optional)* | DIFFORUM_CAMERA |  |  |
| `prefix` *(optional)* | STRING |  | Your shot description; the camera sentence is appended. |
| `include_look` *(optional)* | BOOLEAN | True | Append the Director's look sentence (deforum morph, stop-motion...). |

**Outputs:** `text` (STRING)

### Difforum · LTX Guides

`Difforum_LTXGuides`

Put Difforum keyframes into an LTX-2 / 2.5 latent as guides.

Wraps ComfyUI's core `LTXVAddGuide` once per keyframe, so the camera you
drew becomes the LTX shot. Feed keyframes + indices from the Keyframes
node (from a Storyboard, Guide Frames or a Feedback render), and an empty
LTX latent whose length matches (Setup target = LTX).

| input | type | default | notes |
|---|---|---|---|
| `positive` | CONDITIONING |  |  |
| `negative` | CONDITIONING |  |  |
| `vae` | VAE |  |  |
| `latent` | LATENT |  |  |
| `keyframes` | IMAGE |  |  |
| `indices` | STRING | 0 |  |
| `strength` | FLOAT | 0.7 | Guide strength for in-between keyframes. |
| `first_strength` | FLOAT | 1.0 | Frame 0 usually locks the look: keep it high. |
| `last_strength` | FLOAT | 0.7 |  |

**Outputs:** `positive` (CONDITIONING), `negative` (CONDITIONING), `latent` (LATENT), `info` (STRING)

### Difforum · H3 Shot

`Difforum_H3Shot`

Everything a MiniMax H3 first-last-frame shot needs, from Difforum.

Give it the frames of a Storyboard / Feedback render / Guide Frames and it
returns the first and last frame of the chosen segment, the length on the
H3 17k+5 grid, the size on the 32 px grid, and the camera move of that
segment in words for the prompt. Clips longer than one H3 generation are
split into segments: render segment 0, then feed its last frame as the
next segment's first frame.

| input | type | default | notes |
|---|---|---|---|
| `frames` | IMAGE |  |  |
| `segment_length` | choice (auto (from frames), 124 (~5s), 243 (~10s), 362 (~15s), 481 (~20s)) | auto (from frames) |  |
| `segment` | INT | 0 |  |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  |  |
| `params` *(optional)* | DIFFORUM_PARAMS |  |  |
| `camera` *(optional)* | DIFFORUM_CAMERA |  |  |
| `shot_description` *(optional)* | STRING |  |  |
| `masks` *(optional)* | MASK |  | Guide Frames masks: returns the last frame's revealed area for Fill Reveal. |
| `include_look` *(optional)* | BOOLEAN | True |  |

**Outputs:** `first_frame` (IMAGE), `last_frame` (IMAGE), `length` (INT), `width` (INT), `height` (INT), `prompt` (STRING), `segments` (INT), `last_mask` (MASK), `info` (STRING)

### Difforum · H3 Guides

`Difforum_H3Guides`

Anchor Difforum keyframes inside a MiniMax H3 generation.

Wraps ComfyUI's core `MiniMaxH3AddGuide` once per keyframe, so the camera
you drew becomes the H3 shot: feed keyframes + indices from the Keyframes
node (grid H3) and the positive + AV latent from `MiniMax H3 Reference to
Video` (or `Image to Video`). Optionally anchor a soundtrack at frame 0, so
an audio-reactive direction and H3's own audio stay in sync.

H3 is trained with a few guides per clip: `max_guides` keeps the first,
the last and evenly spaced ones in between.

| input | type | default | notes |
|---|---|---|---|
| `positive` | CONDITIONING |  |  |
| `latent` | LATENT |  | The MiniMax H3 AV latent. |
| `vae` | VAE |  | MiniMax H3 video VAE. |
| `keyframes` | IMAGE |  |  |
| `indices` | STRING | 0 |  |
| `max_guides` | INT | 4 |  |
| `skip_first` | BOOLEAN | False | Enable when frame 0 is already set (e.g. Image to Video first_frame). |
| `audio_vae` *(optional)* | VAE |  | MiniMax H3 audio VAE, needed with audio. |
| `audio` *(optional)* | AUDIO |  | Soundtrack anchored at frame 0. |

**Outputs:** `positive` (CONDITIONING), `info` (STRING)

### Difforum · Fill Reveal (AI)

`Difforum_FillReveal`

Complete what the camera reveals, with an image model (AI hole fill).

Guide Frames, H3 Shot and Keyframes mark the area the camera uncovers
(outside the original picture) in a mask. This node repaints only that
area by inpainting, so edges become new, coherent scenery instead of
gray or stretched pixels. The known pixels are kept exactly.

Works with any image model through ComfyUI's core `InpaintModelConditioning`:
a dedicated inpaint model (SDXL inpainting, Flux Fill) gives the cleanest
seams, a regular checkpoint works too. Run it only on the frames a video
model will see (first / last frame, keyframes) - it is one diffusion per frame.

| input | type | default | notes |
|---|---|---|---|
| `images` | IMAGE |  |  |
| `masks` | MASK |  | 1 = area to fill (Guide Frames default convention). |
| `model` | MODEL |  |  |
| `positive` | CONDITIONING |  | Describe the scene so the fill matches it. |
| `negative` | CONDITIONING |  |  |
| `vae` | VAE |  |  |
| `frames` | choice (all, first, last, first + last) | all |  |
| `steps` | INT | 24 |  |
| `cfg` | FLOAT | 5.0 |  |
| `sampler_name` | choice (euler) | euler |  |
| `scheduler` | choice (normal) | normal |  |
| `grow` | INT | 16 | Pixels the mask is grown into the known image, to hide the seam. |
| `feather` | INT | 12 |  |
| `seed` | INT | 0 |  |

**Outputs:** `images` (IMAGE), `info` (STRING)

## 6 · Export

### Difforum · Camera Export (AE / Blender / JSON)

`Difforum_CameraExport`

Send the Difforum camera to After Effects, Blender or any tool (JSON).

Writes to `output/difforum/`:
  * `.jsx` - run in After Effects: a 3D camera (3d mode) or a transform
    null that reproduces the 2D move exactly (2d mode).
  * `.py` - run in Blender's Text Editor: an animated camera, fps,
    resolution and frame range matched.
  * `.json` - camera-to-world matrices + focal length per frame.
Composite titles, 3D elements or a relight pass on top of the AI render
with the exact same camera.

| input | type | default | notes |
|---|---|---|---|
| `formats` | choice (all, after effects, blender, json) | all |  |
| `filename_prefix` | STRING | difforum_camera |  |
| `blender_unit_scale` | FLOAT | 0.1 | Metres per Difforum scene unit. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  |  |
| `params` *(optional)* | DIFFORUM_PARAMS |  |  |
| `camera` *(optional)* | DIFFORUM_CAMERA |  |  |
| `translation_scale` *(optional)* | FLOAT | 1.0 | Match the sampler's value so the exported move matches the render. |

**Outputs:** `paths` (STRING), `json` (STRING)

### Difforum · Camera Import

`Difforum_CameraImport`

Bring a camera in from Blender, After Effects or JSON.

Accepts the files written by Camera Export and by the helper scripts in
`tools/` (export the active Blender camera, or the selected After Effects
camera). The result drives the Feedback Sampler, Storyboard or Guide
Frames like any Difforum camera, so a move blocked in 3D becomes the AI
shot.

| input | type | default | notes |
|---|---|---|---|
| `file` | choice ((put a camera .json in ComfyUI/input)) |  |  |
| `params` | DIFFORUM_PARAMS |  |  |
| `retime` | choice (match frames, keep source length) | match frames |  |
| `json_text` *(optional)* | STRING |  | Paste JSON here instead of choosing a file. |

**Outputs:** `camera` (DIFFORUM_CAMERA), `info` (STRING)

## 7 · Post

### Difforum · Loop

`Difforum_Loop`

Make a clip loop.

* keep settled lap - for renders made with a looping camera (Director /
  Camera loop_mode) over several laps: keeps the last lap, no blending.
* flow crossfade - morphs the tail into the head along optical flow.
* ping-pong - forward then backward (motion reverses).
Reports how visible the seam is.

| input | type | default | notes |
|---|---|---|---|
| `frames` | IMAGE |  |  |
| `method` | choice (keep settled lap, flow crossfade, ping-pong) | flow crossfade |  |
| `cycle_frames` | INT | 0 | keep settled lap: frames per lap (the Camera node outputs it). |
| `blend_frames` | INT | 12 | flow crossfade: overlap length. |

**Outputs:** `frames` (IMAGE), `seam_info` (STRING)

### Difforum · Symmetry

`Difforum_Symmetry`

Mirror or kaleidoscope a frame or batch (the Feedback Sampler can also
do it inside the loop, where it compounds into a living pattern).

| input | type | default | notes |
|---|---|---|---|
| `image` | IMAGE |  |  |
| `mode` | choice (none, mirror_h, mirror_v, mirror_quad, kaleidoscope) | kaleidoscope |  |
| `segments` | INT | 6 |  |
| `mix` | FLOAT | 1.0 |  |
| `flip` *(optional)* | BOOLEAN | False |  |
| `center_x` *(optional)* | FLOAT | 0.5 |  |
| `center_y` *(optional)* | FLOAT | 0.5 |  |
| `angle` *(optional)* | FLOAT | 0.0 |  |

**Outputs:** `image` (IMAGE)

### Difforum · Echo Trails

`Difforum_EchoTrails`

Long-exposure trails across a frame batch.

| input | type | default | notes |
|---|---|---|---|
| `frames` | IMAGE |  |  |
| `decay` | FLOAT | 0.6 |  |
| `mix` | FLOAT | 0.5 |  |

**Outputs:** `frames` (IMAGE)

### Difforum · Flow Stabilize

`Difforum_FlowStabilize`

Anti-flicker: blends history along optical flow, only where it matches,
so texture stops boiling but motion never ghosts.

| input | type | default | notes |
|---|---|---|---|
| `frames` | IMAGE |  |  |
| `strength` | FLOAT | 0.5 |  |
| `flow_scale` | FLOAT | 0.5 |  |
| `error_gate` | FLOAT | 0.15 |  |

**Outputs:** `frames` (IMAGE)

### Difforum · Detail Guard

`Difforum_DetailGuard`

Unsharp + contrast + grain, for frames that went soft.

| input | type | default | notes |
|---|---|---|---|
| `image` | IMAGE |  |  |
| `sharpen` | FLOAT | 0.3 |  |
| `contrast` | FLOAT | 1.0 |  |
| `grain` | FLOAT | 0.0 |  |
| `grain_mode` | choice (gaussian, plasma) | gaussian |  |

**Outputs:** `image` (IMAGE)
