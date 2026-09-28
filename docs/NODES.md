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
| `target` | choice (feedback (SD/SDXL/Flux), LTX-2 / 2.5, MiniMax H3, Wan 2.x) | feedback (SD/SDXL/Flux) | The model you will render with. Snaps length and size to its grid: H3 17k+5 frames @ 24 fps and 32 px, LTX 8k+1 and 32 px, Wan 4k+1 and 16 px, feedback 8 px. |
| `duration_mode` | choice (seconds, frames) | seconds | Give the length in seconds or in frames. |
| `duration` | FLOAT | 5.0 | Clip length (seconds or frames). The target grid may round it slightly. |
| `fps` | FLOAT | 24.0 | Frames per second. MiniMax H3 is always 24. |
| `aspect` | choice (16:9 landscape, 9:16 vertical, 1:1 square, 4:5 social portrait, 4:3 classic, 3:4 portrait, ...) | 16:9 landscape | Frame shape. 'custom' uses custom_aspect_w / h. |
| `long_edge` | INT | 768 | Pixels on the longer side before snapping. H3 two-stage templates render at 640 and upscale the latent 2x; the feedback look is fine at 768-1024. |
| `seed` | INT | 0 | Master seed for every sampler that reads Setup. |
| `custom_aspect_w` *(optional)* | FLOAT | 16.0 | Width part of a custom aspect (e.g. 2.39). |
| `custom_aspect_h` *(optional)* | FLOAT | 9.0 | Height part of a custom aspect (e.g. 1). |
| `max_megapixels` *(optional)* | FLOAT | 0.0 | Hard cap on frame area. 0 = no cap. ~0.35 on a 16 GB Mac, ~0.65 on 24 GB. |

**Outputs:** `params` (DIFFORUM_PARAMS), `width` (INT), `height` (INT), `frames` (INT), `fps` (FLOAT), `info` (STRING)

### Difforum · Workflow Switches

`Difforum_Switches`

Control panel for the workflow: one switch per group.

Turning a group off mutes it when it holds an output (Previz, Render) and
bypasses it when it sits in the middle of the chain (Live Preview, Fill
Reveal, Restyle, Look Mix, Upscale), so the rest of the graph still runs.
⌖ jumps the canvas to the group. The node runs nothing itself.

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
| `params` | DIFFORUM_PARAMS |  | From Setup: the timeline length and fps. |
| `timeline` | STRING | {"version": 2, "scenes": [{"start": 0... | Written by the timeline editor below (scenes, camera, keys, energy). Edit it there. |
| `camera_mode` | choice (2d, 3d) | 2d | 2d: flat moves (zoom, pan, roll). 3d: real parallax when a depth map reaches the renderer; 3D moves without depth become pseudo-3D. |
| `look` | choice (cinematic, documentary, deforum_morph, animatediff_dream, psychedelic, disco_diffusion, ...) | cinematic | One aesthetic for every renderer. Feedback Sampler: colour lock, detail, grain and extra energy. H3 / LTX: a style sentence at the head of the prompt. |
| `transition` | FLOAT | 1.0 | How much of each camera block eases into the next. 1 = one continuous crane move, 0 = hard cuts between blocks. |
| `camera_scale` | FLOAT | 1.0 | Master multiplier on every move. 0.5 = everything half as far, 2 = twice. |
| `energy_bias` | FLOAT | 0.0 | Moves the whole energy (denoise) curve up or down. +0.1 = more re-imagining everywhere. |
| `variation` | FLOAT | 0.0 | 0 = exactly as drawn. Above 0, each block gets small seeded changes in speed and amount - useful for variations of the same shot. |
| `variation_seed` | INT | 0 | Seed for those variations: change it for another take. |
| `clip` *(optional)* | CLIP |  | Connect the CLIP of your image model to encode the scene prompts: the Feedback Sampler and Restyle then travel through them. |
| `audio` *(optional)* | DIFFORUM_AUDIO |  | From Audio Analyzer: enables the per-block audio reactions (beat pulse, bass speed...). |

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
| `params` | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `keys` | STRING | 0: zoom_in 0.6 0.5 40 ease_in_out 30:... | One key per line: frame: move [speed] [intensity] [lens°] [ease]. e.g. '48: orbit_right 1.2 0.8 35 ease_in_out'. Frames can be '2.5s'. |
| `camera_mode` | choice (2d, 3d) | 2d | 2d = flat moves; 3d = parallax with a depth map. |
| `transition` | FLOAT | 1.0 | 1 = moves blend into each other; 0 = hard cuts at each key. |
| `speed` | FLOAT | 1.0 | Master speed on every key. |
| `loop_mode` | choice (harmonic, zero_mean, off) | off | harmonic / zero_mean make the path periodic over cycle_frames, so the clip loops with no crossfade (installations). off = free path. |
| `cycle_frames` | INT | 0 | Frames per loop. 0 = the whole clip is one cycle. Feeds Loop 'keep settled lap'. |
| `harmonics` | INT | 3 | harmonic loop: how many sine terms describe the path. More = closer to the keys, fewer = smoother loop. |
| `audio` *(optional)* | DIFFORUM_AUDIO |  | From Audio Analyzer: per-frame loudness, bands and beats. |

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
| `params` | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `mode` | choice (2d, 3d) | 2d | 2d = flat moves; 3d = parallax with a depth map. |
| `fov` | FLOAT | 40.0 | Lens field of view in degrees. 20-30 long lens, 40 normal, 60+ wide. |
| `translation_x` | STRING | 0:(0) | Horizontal move per frame. Deforum syntax: '0:(0), 60:(2*sin(t/10))'; t = frame. |
| `translation_y` | STRING | 0:(0) | Vertical move per frame, same syntax. |
| `translation_z` | STRING | 0:(0) | Forward (+) / back (-) per frame. 3D only. |
| `rotation_3d_x` | STRING | 0:(0) | Tilt, degrees per frame. 3D only. |
| `rotation_3d_y` | STRING | 0:(0) | Pan (turn), degrees per frame. 3D only. |
| `rotation_3d_z` | STRING | 0:(0.3) | Roll, degrees per frame. |
| `zoom` | STRING | 0:(1.0 + 0.004*sin(2*pi*t/48)) | Scale per frame. 1.0 = none, 1.01 = 1% closer each frame. |
| `audio` *(optional)* | DIFFORUM_AUDIO |  | Audio curves usable in expressions: amp, low, mid, high, onset, beat. |
| `translation_x_curve` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule (Audio Curve, Schedule) that replaces the text for this axis. |
| `translation_y_curve` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule that replaces the text for this axis. |
| `translation_z_curve` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule that replaces the text for this axis. |
| `rotation_3d_x_curve` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule that replaces the text for this axis. |
| `rotation_3d_y_curve` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule that replaces the text for this axis. |
| `rotation_3d_z_curve` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule that replaces the text for this axis. |
| `zoom_curve` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule that replaces the text for this axis. |

**Outputs:** `camera` (DIFFORUM_CAMERA), `info` (STRING)

### Difforum · Storyboard

`Difforum_Storyboard`

The whole clip warped by the camera, no diffusion - in about a second.

Same engine as the Feedback Sampler with the sampler switched off, so what
you see is exactly the motion the render will get: contact sheet, the
frames themselves, the camera path from above, and a pacing verdict.

| input | type | default | notes |
|---|---|---|---|
| `init_image` | IMAGE |  | Any still: the camera move is played over it. |
| `every_nth` | INT | 8 | One contact-sheet cell every N frames. |
| `columns` | INT | 6 | Cells per row on the contact sheet. |
| `preview_scale` | FLOAT | 0.5 | Working size vs. Setup. 0.5 = half size, much faster. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | The Director's wire: camera, energy curve, scene prompts, keys and look in one cable. |
| `params` *(optional)* | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `camera` *(optional)* | DIFFORUM_CAMERA |  | A camera track from Camera (keys), Camera (expressions) or Camera Import. Overrides the Director's camera. |
| `depth` *(optional)* | IMAGE |  | A depth map (white = near), e.g. Depth Anything V2 on the first frame. Needed for real 3D parallax; without it 3D moves fall back to pseudo-3D. |
| `symmetry` *(optional)* | choice (none, mirror_h, mirror_v, mirror_quad, kaleidoscope) | none | Preview an in-loop symmetry (kaleidoscope...) as the Feedback Sampler would apply it. |

**Outputs:** `sheet` (IMAGE), `frames` (IMAGE), `camera_path` (IMAGE), `info` (STRING)

### Difforum · Keyframe Images

`Difforum_KeyframeImages`

Pin your own pictures to moments of the clip (multikeyframing).

Place markers on the Director's **Keys** track (or type times: `0, 4s,
9.5s`) and feed a batch of images in the same order. The keyframes then
drive every renderer: the Feedback Sampler travels *through* them, H3 /
LTX Guides anchor them, the Animatic shows them. Made for installations and
experimental pieces where the image has to hit a picture on a beat.

| input | type | default | notes |
|---|---|---|---|
| `images` | IMAGE |  | One picture per key, in time order (Batch Images). These pictures set the look and the content the render passes through, so make them good: same style, same resolution. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | The Director's wire: camera, energy curve, scene prompts, keys and look in one cable. |
| `params` *(optional)* | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `times` *(optional)* | STRING |  | Override the Director's Keys: '0, 4s, 9.5s' or frame numbers. Empty = use the Keys track; no keys = spread evenly. |

**Outputs:** `keyframes` (IMAGE), `indices` (STRING), `info` (STRING)

### Difforum · Animatic (previz)

`Difforum_Animatic`

Previz the whole shot before rendering anything heavy.

Plays the Director's camera over your first frame (or a stand-in plate) at
low resolution, in about a second, with the timecode, the active scene
prompt, the camera move, the energy level and the keyframe markers burnt
in. Image keyframes, when connected, show up at their moments. Send it to
Create Video + Save Video; the Director's "Previz only" button mutes the
render outputs so only this runs.

| input | type | default | notes |
|---|---|---|---|
| `direction` | DIFFORUM_DIRECTION |  | The Director to previz. |
| `preview_scale` | FLOAT | 0.4 | Size vs. Setup. 0.4 = small and instant; 1.0 = full size. |
| `overlay` | BOOLEAN | True | Burn in timecode, camera move, scene prompt, energy bar and key markers. |
| `init_image` *(optional)* | IMAGE |  | First frame to move over. Empty = a grid plate. |
| `depth` *(optional)* | IMAGE |  | Depth of the first frame, for real parallax in 3d mode. |
| `key_images` *(optional)* | IMAGE |  | Keyframe Images: shown at their moments. |
| `key_indices` *(optional)* | STRING |  | Their frame numbers. |

**Outputs:** `frames` (IMAGE), `fps` (FLOAT), `info` (STRING)

## 3 · Curves & Prompts

### Difforum · Schedule

`Difforum_Schedule`

A per-frame curve from Deforum keyframe syntax: `0:(0.4), 48:(0.6+0.1*sin(t/8))`.

Variables: t/f (frame), s (seconds), fps, max_f, and every audio band when
an Audio Analyzer is connected. Easing applies between keyframes.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `schedule` | STRING | 0:(0.45), 60:(0.6), 119:(0.45) | Values over time: 'frame:(value)', e.g. '0:(0.45), 60:(0.6)'. Frames can be '2s'. Expressions allowed: t = frame, amp/low/beat with audio. |
| `easing` | choice (linear, ease_in, ease_out, ease_in_out, step) | ease_in_out | How values move between keys. |
| `audio` *(optional)* | DIFFORUM_AUDIO |  | From Audio Analyzer: per-frame loudness, bands and beats. |

**Outputs:** `schedule` (DIFFORUM_SCHEDULE), `info` (STRING)

### Difforum · Schedule Plot

`Difforum_SchedulePlot`

Draw any schedule as an image (and a one-line summary).

| input | type | default | notes |
|---|---|---|---|
| `schedule` | DIFFORUM_SCHEDULE |  | Any schedule or curve to draw. |
| `width` | INT | 512 | Plot width in pixels. |
| `height` | INT | 200 | Plot height in pixels. |

**Outputs:** `plot` (IMAGE), `info` (STRING)

### Difforum · Audio Analyzer

`Difforum_AudioAnalyzer`

Audio -> per-frame bands: amp, low, mid, high, onset, beat (0..1).

Use the bands in any expression (`1 + 0.05*low`), in a Director camera
block's audio reaction, or turn one into a curve with Audio Curve.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `audio` | AUDIO |  | Your track (Load Audio). |
| `smoothing` | FLOAT | 0.25 | 0 = every transient, 0.5+ = slow envelope. 0.2-0.3 reads well on camera. |
| `beat_sensitivity` | FLOAT | 1.5 | Higher = fewer, stronger beats detected. |
| `offset_seconds` | FLOAT | 0.0 | Start reading the track here, to sync with an edit. |

**Outputs:** `audio_curves` (DIFFORUM_AUDIO), `bands_plot` (IMAGE), `info` (STRING)

### Difforum · Audio Curve

`Difforum_AudioCurve`

One audio band -> a ready curve: e.g. bass-pumped energy
(`low`, add, base 0.45, amount 0.2) or a beat-pulsed cfg.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `audio_curves` | DIFFORUM_AUDIO |  | From Audio Analyzer. |
| `source` | choice (amp, low, mid, high, onset, beat) | low | Which band drives the curve: amp (loudness), low / mid / high bands, onset (attacks), beat (pulses). |
| `combine` | choice (add, subtract, multiply) | add | How the audio is applied to base: add, subtract or multiply. |
| `base` | FLOAT | 0.45 | Value when the band is silent (e.g. 0.45 denoise). |
| `amount` | FLOAT | 0.2 | How far the band pushes the value at its loudest. |
| `smoothing` | FLOAT | 0.2 | Extra smoothing of the band before it is used. |

**Outputs:** `schedule` (DIFFORUM_SCHEDULE)

### Difforum · Prompt Travel

`Difforum_PromptTravel`

Prompt travel from `frame: prompt` lines (or `2.5s: prompt`), blended
per frame. Encodes each prompt once and blends lazily, so long clips on big
text encoders stay light. `build_batched` adds one batched conditioning
(frame i = prompt i) for batch samplers such as AnimateDiff.

| input | type | default | notes |
|---|---|---|---|
| `params` | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `clip` | CLIP |  | CLIP of the image model. |
| `prompts` | STRING | 0: a serene misty forest, soft light ... | One prompt per line: 'frame: text' or '2.5s: text'. Consecutive prompts blend. |
| `easing` | choice (linear, ease_in, ease_out, ease_in_out, step) | ease_in_out | How each prompt blends into the next. |
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
| `model` | MODEL |  | The image model (SD1.5 / SDXL / Flux / turbo). MODEL output of your checkpoint or LoRA chain. |
| `positive` | CONDITIONING |  | Look / style prompt. Used for every frame when no scene prompts reach the sampler. |
| `negative` | CONDITIONING |  | Negative conditioning (CLIP Text Encode). |
| `vae` | VAE |  | The VAE that matches the model. |
| `init_image` | IMAGE |  | The first frame. Everything grows from this image. |
| `steps` | INT | 20 | Steps for a full denoise. With step scaling (default) a frame at energy 0.5 runs half of them, like Deforum. |
| `cfg` | FLOAT | 6.0 | Prompt strength. Turbo / LCM / DMD2 models want 1-2; regular SDXL 4-7. |
| `sampler_name` | choice (euler, lcm) | euler | Sampler. euler / dpmpp_2m for regular models, euler_ancestral or lcm for turbo. |
| `scheduler` | choice (normal, sgm_uniform) | normal | Noise schedule. karras / normal for regular models, sgm_uniform for turbo. |
| `cadence` | INT | 1 | Diffuse every Nth frame; the frames in between ride the camera and crossfade. 2 is almost invisible and twice as fast. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | The Director's wire: camera, energy curve, scene prompts, keys and look in one cable. |
| `options` *(optional)* | DIFFORUM_OPTIONS |  | Difforum · Render Options, for manual control. |
| `params` *(optional)* | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `camera` *(optional)* | DIFFORUM_CAMERA |  | A camera track from Camera (keys), Camera (expressions) or Camera Import. Overrides the Director's camera. |
| `strength` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule that replaces the Director's energy curve (denoise per frame). |
| `cfg_curve` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule for cfg per frame. |
| `prompts` *(optional)* | DIFFORUM_PROMPT |  | Prompt travel from Prompt Travel (or the Director with CLIP connected). |
| `depth` *(optional)* | IMAGE |  | Depth of the first frame (white = near), or a per-frame batch. |
| `control_net` *(optional)* | CONTROL_NET |  | Optional ControlNet (depth, canny...) applied on every diffused frame. |
| `control_image` *(optional)* | IMAGE |  | Per-frame control images; empty = the frame itself. |
| `control_strength` *(optional)* | FLOAT | 0.6 | ControlNet strength. |
| `energy` *(optional)* | FLOAT | 0.5 | Denoise per frame when no Director or strength curve is connected. It is how much each frame is re-imagined: 0.3 = steady, 0.5 = classic Deforum morph, 0.7+ = the image changes every few frames. |
| `key_images` *(optional)* | IMAGE |  | Keyframe Images: the travel steers toward each picture and lands on it at its frame. This is multikeyframing. |
| `key_indices` *(optional)* | STRING |  | Frame numbers of the key pictures (Keyframe Images 'indices'). |
| `key_pull` *(optional)* | FLOAT | 0.65 | How strongly the frame is pulled toward the next key picture before diffusion. 1 = arrives exactly on it, 0.6 = strong resemblance, 0.3 = a hint. |
| `key_approach` *(optional)* | INT | 12 | Frames before a key over which the pull ramps up. Short (6-12) = sudden arrival, long (24-48) = the image slowly becomes the key. Long shots want long approaches. |

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
| `model` | MODEL |  | The image model (SD1.5 / SDXL / Flux / turbo). MODEL output of your checkpoint or LoRA chain. |
| `positive` | CONDITIONING |  | Positive conditioning (CLIP Text Encode). |
| `negative` | CONDITIONING |  | Negative conditioning (CLIP Text Encode). |
| `vae` | VAE |  | The VAE that matches the model. |
| `init_image` | IMAGE |  | The first frame. Everything grows from this image. |
| `run_frames` | INT | 480 | How many frames to run (the loop wraps the camera). |
| `steps` | INT | 2 | Steps per frame. Turbo models: 1-4. |
| `cfg` | FLOAT | 1.2 | Prompt strength. Turbo / LCM / DMD2 models want 1-2; regular SDXL 4-7. |
| `sampler_name` | choice (euler, lcm) | lcm | Sampler. euler / dpmpp_2m for regular models, euler_ancestral or lcm for turbo. |
| `scheduler` | choice (normal, sgm_uniform) | sgm_uniform | Noise schedule. karras / normal for regular models, sgm_uniform for turbo. |
| `cadence` | INT | 1 | Diffuse every Nth frame for speed. |
| `target_fps` | FLOAT | 0.0 | Cap the speed. 0 = as fast as possible. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | The Director's wire: camera, energy curve, scene prompts, keys and look in one cable. |
| `options` *(optional)* | DIFFORUM_OPTIONS |  | Difforum · Render Options: manual colour, detail, symmetry, depth and chunk settings. Without it the Director's look decides. |
| `params` *(optional)* | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `camera` *(optional)* | DIFFORUM_CAMERA |  | A camera track from Camera (keys), Camera (expressions) or Camera Import. Overrides the Director's camera. |
| `strength` *(optional)* | DIFFORUM_SCHEDULE |  | A schedule for denoise per frame; replaces the Director's energy curve. |
| `prompts` *(optional)* | DIFFORUM_PROMPT |  | Prompt travel from Prompt Travel (or the Director with CLIP connected). |
| `depth` *(optional)* | IMAGE |  | A depth map (white = near), e.g. Depth Anything V2 on the first frame. Needed for real 3D parallax; without it 3D moves fall back to pseudo-3D. |
| `energy` *(optional)* | FLOAT | 0.5 | Denoise per frame when no Director is connected. |
| `loop_camera` *(optional)* | BOOLEAN | True | Repeat the camera when the run is longer than the clip. |
| `live_source` *(optional)* | STRING |  | '' = off, '0' = webcam, or a video path: mixed in before each frame. |
| `source_blend` *(optional)* | FLOAT | 0.9 | How much of the live source enters each frame. |
| `stream_dir` *(optional)* | STRING |  | Folder to write live PNG frames to (for another app). |
| `spout_name` *(optional)* | STRING |  | Spout sender name (Windows) for Resolume / TouchDesigner / OBS. |
| `keep_frames` *(optional)* | INT | 240 | Frames kept in memory for the output (ring buffer). |
| `live_preview` *(optional)* | BOOLEAN | True | Show the frames inside the node while running. |

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
| `color_mode` | choice (lab, rgb, none) | lab | How colour is matched to the anchor: lab (perceptual), rgb, or none. |
| `anchor_mode` | choice (scene, first, rolling, none) | scene | scene = re-anchor colour at each prompt scene; first = hold frame 0; rolling = previous key; none = free. |
| `sharpen` | FLOAT | 0.2 | Sharpening after each diffused frame; keeps feedback from going soft. |
| `noise` | FLOAT | 0.02 | Grain added before each frame is diffused; feeds new detail. |
| `hole_noise` | FLOAT | 0.25 | Extra noise where the camera reveals new area, so it is repainted. |
| `symmetry` | choice (none, mirror_h, mirror_v, mirror_quad, kaleidoscope) | none | Symmetry applied inside the loop (kaleidoscope for mandalas). |
| `symmetry_segments` | INT | 6 | Kaleidoscope segments. |
| `border` | choice (reflection, border, zeros) | reflection | What fills the edges the camera exposes: reflection (mirror), border (stretch), zeros. |
| `depth_tracking` | choice (follow, static) | follow | follow = depth is re-projected with the image every frame. |
| `translation_scale` | FLOAT | 1.0 | 3D motion strength vs. the depth map's range. |
| `near` | FLOAT | 1.0 | Depth calibration: distance of the nearest depth value. |
| `far` | FLOAT | 100.0 | Depth calibration: distance of the farthest depth value. |
| `invert_depth` | BOOLEAN | False | Enable if near things are dark in your depth map. |
| `step_scaling` | choice (by energy (fast), fixed) | by energy (fast) | by energy: a frame at denoise 0.5 runs half the steps, like Deforum (about 2x faster at the same look). fixed: every frame runs all steps. |
| `seed_mode` | choice (fixed, increment) | fixed | fixed = same noise every frame (calmer texture). |
| `start_frame` | INT | 0 | Feedback Sampler: render a chunk; init_image is the frame at start_frame. |
| `end_frame` | INT | 0 | 0 = to the end. |

**Outputs:** `options` (DIFFORUM_OPTIONS)

### Difforum · Restyle (Deforum / AnimateDiff look)

`Difforum_Restyle`

Give a video-model render the Deforum / AnimateDiff / Disco look.

MiniMax H3 or LTX supplies the motion; an image model (SDXL, SD1.5, Flux,
turbo distills) re-paints every frame in your look, with the previous
stylized frame carried along the video's optical flow and mixed in. That
feedback is what made Deforum morph and AnimateDiff boil - here it rides on
the video model's motion instead of a synthetic camera.

Styles: clean restyle (steady), animatediff boil (texture re-rolls each
frame), deforum morph (strong feedback smear), disco flicker (high denoise,
per-frame seed, loose colour). `custom` uses feedback / seed_mode /
color_hold as set. Connect a Director to use its scene prompts and energy.

| input | type | default | notes |
|---|---|---|---|
| `video` | IMAGE |  | Frames from H3 / LTX / any video (Get Video Components). |
| `model` | MODEL |  | An image model in the look you want. Turbo SDXL / DMD2 at 4-6 steps is fast and good. |
| `positive` | CONDITIONING |  | The look, e.g. 'oil painting, thick brush strokes'. |
| `negative` | CONDITIONING |  | Negative conditioning (CLIP Text Encode). |
| `vae` | VAE |  | The VAE that matches the model. |
| `style` | choice (clean restyle, animatediff boil, deforum morph, disco flicker, custom) | deforum morph | deforum morph: strong feedback, smears along motion. animatediff boil: texture re-rolls each frame. disco flicker: high energy, loose colour. clean restyle: steady painted version. custom: uses feedback / seed_mode / color_hold below. |
| `denoise` | FLOAT | 0.45 | How much each frame is re-painted. 0.3 keeps the render, 0.6+ re-imagines it. |
| `steps` | INT | 12 | Full-denoise steps; each frame runs steps x denoise. |
| `cfg` | FLOAT | 5.0 | Prompt strength. Turbo models: 1-2. |
| `sampler_name` | choice (euler, lcm) | euler | Sampler. euler / dpmpp_2m for regular models, euler_ancestral or lcm for turbo. |
| `scheduler` | choice (normal, sgm_uniform) | normal | Noise schedule. karras / normal for regular models, sgm_uniform for turbo. |
| `cadence` | INT | 1 | Re-paint every Nth frame; the rest follow the flow. 2 = twice as fast. |
| `long_edge` | INT | 1024 | Working size. 0 = the video's size. Upscale afterwards for 2K. |
| `seed` | INT | 0 | Noise seed. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | Director: scene prompts (with CLIP) and the energy curve. |
| `prompts` *(optional)* | DIFFORUM_PROMPT |  | Prompt travel; otherwise the Director's scene prompts, otherwise positive. |
| `feedback` *(optional)* | FLOAT | 0.35 | custom: share of the previous stylized frame in the next one. |
| `seed_mode` *(optional)* | choice (fixed, per frame) | fixed | custom: per frame = texture re-rolls every frame (boil / flicker). |
| `color_hold` *(optional)* | FLOAT | 0.5 | custom: how much each frame keeps the source colours. |
| `follow_energy` *(optional)* | BOOLEAN | True | With a Director: denoise follows its energy curve (0.5 = as set). |
| `flow_scale` *(optional)* | FLOAT | 0.5 | Resolution of the motion estimate that carries the previous frame. 0.5 = fast. |
| `control_net` *(optional)* | CONTROL_NET |  | A ControlNet that holds the clip's structure while the look is re-painted, so denoise can go up to 0.6-0.8 (the real Deforum / AnimateDiff range) without losing the shapes. Depth or canny with control_image; tile with the clip itself. |
| `control_image` *(optional)* | IMAGE |  | Per-frame control maps of the clip (Depth Anything 3, Canny...). Empty = the clip frames themselves (tile / union ControlNets). |
| `control_strength` *(optional)* | FLOAT | 0.6 | How hard the structure is held. 0.4 loose, 0.6 balanced, 0.9 locked. |

**Outputs:** `frames` (IMAGE), `report` (STRING)

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
| `anchor_image` | IMAGE |  | The image the camera travels over (usually your first frame). |
| `mask_convention` | choice (1 = generate (VACE / LTX / H3), 1 = keep) | 1 = generate (VACE / LTX / H3) | Which value the mask uses for 'generate': 1 = generate (VACE / LTX / H3 / Fill Reveal) or 1 = keep. |
| `hole_fill` | choice (gray, stretch edge, black) | gray | What goes into the area the camera uncovers before AI fills it: gray (neutral, best for Fill Reveal and H3), stretched edge, or black. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | The Director's wire: camera, energy curve, scene prompts, keys and look in one cable. |
| `params` *(optional)* | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `camera` *(optional)* | DIFFORUM_CAMERA |  | A camera track from Camera (keys), Camera (expressions) or Camera Import. Overrides the Director's camera. |
| `depth` *(optional)* | IMAGE |  | Depth of the anchor, for real parallax in 3d mode. |
| `translation_scale` *(optional)* | FLOAT | 1.0 | Strength of 3D translation vs. the depth range. |

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
| `frames` | IMAGE |  | The clip to pick keyframes from (Guide Frames, Storyboard, a Feedback render...). Keyframes carry its look into H3 / LTX: better frames here = better video. |
| `grid` | choice (LTX-2 / 2.5 (8k+1), MiniMax H3 (17k+5), Wan 2.x (4k+1), any) | LTX-2 / 2.5 (8k+1) | Snap keyframes to the video model's latent grid (H3 17k+5, LTX 8k+1, Wan 4k+1) so each one lands on a real latent frame. |
| `every_seconds` | FLOAT | 1.0 | How many keyframes: one every N seconds. 1 = many, strong guidance; 2 = fewer, more freedom; 0 = first and last only. H3 Guides then keeps up to max_guides. |
| `fps` | FLOAT | 24.0 | Frames per second of the clip (from Setup). |
| `indices` *(optional)* | STRING |  | Exact frames instead of spacing, e.g. '0, 48, 96, -1' (-1 = last). |
| `masks` *(optional)* | MASK |  | Guide Frames masks, so Fill Reveal can repaint what the camera uncovered. |

**Outputs:** `keyframes` (IMAGE), `indices` (STRING), `sparse_batch` (IMAGE), `first_frame` (IMAGE), `last_frame` (IMAGE), `length` (INT), `key_masks` (MASK), `info` (STRING)

### Difforum · Camera → Prompt

`Difforum_CameraPrompt`

The camera, in words - for prompt-driven video models (MiniMax H3,
LTX, Seedance, Veo...). Uses the Director's blocks when available, or
reads any camera track and names its moves.

| input | type | default | notes |
|---|---|---|---|
| `format` | choice (sentence, timed lines, prompt suffix, H3 structured) | sentence | sentence: one line of camera direction (+ look) to append to any prompt. timed lines: [0.0s-2.5s] camera ... per block. prompt suffix: 'Camera: ...'. H3 structured: the whole Director timeline (scenes, camera, keys, look) written in MiniMax H3's native prompt format. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | The Director's wire: camera, energy curve, scene prompts, keys and look in one cable. |
| `params` *(optional)* | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `camera` *(optional)* | DIFFORUM_CAMERA |  | A camera track from Camera (keys), Camera (expressions) or Camera Import. Overrides the Director's camera. |
| `prefix` *(optional)* | STRING |  | What is on screen. sentence / suffix: put before the camera text. H3 structured: the opening composition ([Shot 1]); in reference mode, what <Picture 1> shows. |
| `include_look` *(optional)* | BOOLEAN | True | Append the Director's look sentence (deforum morph, stop-motion...). |
| `h3_mode` *(optional)* | choice (T2VA (text only), I2VA (first frame), FL2VA (first + last frame), reference (ref2va / guides)) | reference (ref2va / guides) | H3 structured only. Which H3 task the prompt is for: sets the alignment line (I2VA / FL2VA) or the six full-reference sections (ref2va with H3 Guides, <Picture 1> as the first frame). |
| `soundscape` *(optional)* | STRING |  | H3 structured: overall_soundscape - ambience and physical sounds only (wind, footsteps, rain). Empty = soft natural ambience. |
| `music` *(optional)* | STRING |  | H3 structured: non_diegetic_music - instruments, tempo, dynamics, no mood words. Empty = N/A (no score). |
| `cuts` *(optional)* | BOOLEAN | False | H3 structured: off = one continuous take (scene changes become transformations). On = every scene starts a new [Shot N] at its cut time. |

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
| `positive` | CONDITIONING |  | LTX positive conditioning. |
| `negative` | CONDITIONING |  | LTX negative conditioning. |
| `vae` | VAE |  | LTX video VAE. |
| `latent` | LATENT |  | Empty LTX latent (length and size from Keyframes / Setup). |
| `keyframes` | IMAGE |  | Keyframe pictures (from Keyframes or Keyframe Images). |
| `indices` | STRING | 0 | Frame number of each keyframe, comma separated, e.g. 0,48,96,123. |
| `strength` | FLOAT | 0.7 | Guide strength of the keyframes in between. 0.6-0.8 = follows the path, 1 = locked. |
| `first_strength` | FLOAT | 1.0 | Frame 0 usually locks the look: keep it high. |
| `last_strength` | FLOAT | 0.7 | Strength of the last keyframe. High = the shot must end on it. |

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
| `frames` | IMAGE |  | The clip whose first and last frame H3 should connect (Guide Frames, Storyboard, Feedback render). |
| `segment_length` | choice (auto (from frames), 124 (~5s), 243 (~10s), 362 (~15s), 481 (~20s)) | auto (from frames) | Frames per H3 generation. Clips longer than this are split into segments. |
| `segment` | INT | 0 | Which segment to render: 0, then 1 with segment 0's last frame as its first frame... |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | The Director's wire: camera, energy curve, scene prompts, keys and look in one cable. |
| `params` *(optional)* | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `camera` *(optional)* | DIFFORUM_CAMERA |  | A camera track from Camera (keys), Camera (expressions) or Camera Import. Overrides the Director's camera. |
| `shot_description` *(optional)* | STRING |  | What is on screen at the start: subject, place, light. Goes into the prompt. |
| `masks` *(optional)* | MASK |  | Guide Frames masks: returns the last frame's revealed area for Fill Reveal. |
| `include_look` *(optional)* | BOOLEAN | True | Add the Director's look as the style of the shot. |
| `prompt_style` *(optional)* | choice (H3 structured, sentence) | H3 structured | H3 structured: this segment's scenes, camera, keys and look in MiniMax H3's native FL2VA format (alignment line + fields). sentence: description + camera sentence + look, as plain text. |
| `soundscape` *(optional)* | STRING |  | overall_soundscape: ambience and physical sounds only. |
| `music` *(optional)* | STRING |  | non_diegetic_music: instruments, tempo, dynamics. Empty = N/A. |

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
| `positive` | CONDITIONING |  | Positive conditioning from MiniMax H3 Reference to Video. |
| `latent` | LATENT |  | The MiniMax H3 AV latent. |
| `vae` | VAE |  | MiniMax H3 video VAE. |
| `keyframes` | IMAGE |  | Keyframe pictures (Keyframes, Keyframe Images or Fill Reveal). |
| `indices` | STRING | 0 | Frame number of each keyframe, comma separated, e.g. 0,48,96,123. |
| `max_guides` | INT | 4 | Most keyframes anchored inside the generation. The first and last are kept, the rest evenly spaced. More = H3 follows your frames closely; fewer = more of H3's own motion and invention. 4 for a camera path, 6-8 for a look pass. |
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
| `images` | IMAGE |  | Frames with holes where the camera uncovered new area (gray from Guide Frames). |
| `masks` | MASK |  | 1 = area to fill (Guide Frames default convention). |
| `model` | MODEL |  | An image model; an inpainting checkpoint (SDXL inpainting, Flux Fill) gives the cleanest seams. |
| `positive` | CONDITIONING |  | Describe the scene so the fill matches it. |
| `negative` | CONDITIONING |  | Negative conditioning (CLIP Text Encode). |
| `vae` | VAE |  | The VAE that matches the model. |
| `frames` | choice (all, first, last, first + last) | all | Which frames to paint: all keyframes, or only the first / last (FL2VA). |
| `steps` | INT | 24 | Inpainting steps. |
| `cfg` | FLOAT | 5.0 | Prompt strength. Turbo / LCM / DMD2 models want 1-2; regular SDXL 4-7. |
| `sampler_name` | choice (euler) | euler | Sampler. euler / dpmpp_2m for regular models, euler_ancestral or lcm for turbo. |
| `scheduler` | choice (normal) | normal | Noise schedule. karras / normal for regular models, sgm_uniform for turbo. |
| `grow` | INT | 16 | Pixels the mask is grown into the known image, to hide the seam. |
| `feather` | INT | 12 | Soft edge, in pixels, where the painted area meets the original. |
| `seed` | INT | 0 | Fill seed; change it for another fill. |

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
| `formats` | choice (all, after effects, blender, json) | all | Which files to write to output/difforum: After Effects .jsx, Blender .py, JSON. |
| `filename_prefix` | STRING | difforum_camera | File name prefix. |
| `blender_unit_scale` | FLOAT | 0.1 | Metres per Difforum scene unit. |
| `direction` *(optional)* | DIFFORUM_DIRECTION |  | The Director's wire: camera, energy curve, scene prompts, keys and look in one cable. |
| `params` *(optional)* | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `camera` *(optional)* | DIFFORUM_CAMERA |  | A camera track from Camera (keys), Camera (expressions) or Camera Import. Overrides the Director's camera. |
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
| `file` | choice ((put a camera .json in ComfyUI/input)) |  | A camera .json in ComfyUI/input: from Camera Export, the Blender exporter or the After Effects script in tools/. |
| `params` | DIFFORUM_PARAMS |  | From Setup: frame count, fps, width and height. Plug the same Setup everywhere. |
| `retime` | choice (match frames, keep source length) | match frames | match frames = stretch the move to Setup's length; keep source length = frame for frame. |
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
| `frames` | IMAGE |  | A batch of frames (a clip). |
| `method` | choice (keep settled lap, flow crossfade, ping-pong) | flow crossfade | keep settled lap: render several laps, keep the last (perfect loop). flow crossfade: blend the end into the start along motion. ping-pong: forward then back. |
| `cycle_frames` | INT | 0 | keep settled lap: frames per lap (the Camera node outputs it). |
| `blend_frames` | INT | 12 | flow crossfade: overlap length. |

**Outputs:** `frames` (IMAGE), `seam_info` (STRING)

### Difforum · Look Mix

`Difforum_LookMix`

Keep the experimental look while a video model supplies the motion.

Video models such as MiniMax H3 render fluid, coherent motion - and smooth
away the boiling, flickering texture that makes Deforum, AnimateDiff and
Disco Diffusion feel alive. Feed the video-model render and the Feedback
Sampler look pass (same length or not; it is resampled in time):

* detail transfer - H3's motion, the look pass's high-frequency texture
* colour + detail - also pulls each frame's palette to the look pass
* flicker cuts - every Nth frame (seeded jitter) swaps to the look pass
* crossfade - plain mix

`step_fps` then holds frames to 8-12 fps for a stop-motion / AnimateDiff
stutter, independently of the blend (works without a look pass too).

| input | type | default | notes |
|---|---|---|---|
| `video` | IMAGE |  | The video-model render (H3, LTX...). Its motion and light are kept. |
| `blend` | choice (detail transfer, colour + detail, flicker cuts, crossfade, none) | detail transfer | detail transfer: the look pass's fine texture on top. colour + detail: also its palette. flicker cuts: cut to the look pass every few frames. crossfade: plain mix. |
| `amount` | FLOAT | 0.6 | How much of the look pass comes through. |
| `detail_radius` | INT | 3 | Texture scale taken from the look pass. |
| `flicker_every` | INT | 3 | flicker cuts: roughly one cut every N frames. |
| `step_fps` | FLOAT | 0.0 | 0 = off. 8-12 gives a hand-made stutter. |
| `fps` | FLOAT | 24.0 | Clip fps, for step_fps. |
| `seed` | INT | 0 | flicker cuts: which frames flicker. |
| `look_pass` *(optional)* | IMAGE |  | Frames in the look (Feedback Sampler or Restyle). Resampled to the video's length. |

**Outputs:** `frames` (IMAGE)

### Difforum · Symmetry

`Difforum_Symmetry`

Mirror or kaleidoscope a frame or batch (the Feedback Sampler can also
do it inside the loop, where it compounds into a living pattern).

| input | type | default | notes |
|---|---|---|---|
| `image` | IMAGE |  | An image or a batch of frames. |
| `symmetry` | choice (none, mirror_h, mirror_v, mirror_quad, kaleidoscope) | kaleidoscope | Mirror or kaleidoscope the image. |
| `segments` | INT | 6 | Kaleidoscope segments. |
| `mix` | FLOAT | 1.0 | How much of the symmetric image is used. |
| `flip` *(optional)* | BOOLEAN | False | Mirror the other way. |
| `center_x` *(optional)* | FLOAT | 0.5 | Symmetry centre, 0-1 across. |
| `center_y` *(optional)* | FLOAT | 0.5 | Symmetry centre, 0-1 down. |
| `angle` *(optional)* | FLOAT | 0.0 | Rotation of the symmetry axes, degrees. |

**Outputs:** `image` (IMAGE)

### Difforum · Echo Trails

`Difforum_EchoTrails`

Long-exposure trails across a frame batch.

| input | type | default | notes |
|---|---|---|---|
| `frames` | IMAGE |  | A batch of frames (a clip). |
| `decay` | FLOAT | 0.6 | How long trails last. 0.9 = long exposure. |
| `mix` | FLOAT | 0.5 | How much of the trail is laid over the frame. |

**Outputs:** `frames` (IMAGE)

### Difforum · Flow Stabilize

`Difforum_FlowStabilize`

Anti-flicker: blends history along optical flow, only where it matches,
so texture stops boiling but motion never ghosts.

| input | type | default | notes |
|---|---|---|---|
| `frames` | IMAGE |  | A batch of frames (a clip). |
| `strength` | FLOAT | 0.5 | How much history is blended in along motion. 0.3-0.5 calms boiling texture; 0 = off. |
| `flow_scale` | FLOAT | 0.5 | Resolution of the motion estimate. 0.5 = fast, 1.0 = precise. |
| `error_gate` | FLOAT | 0.15 | Where the motion estimate is unsure (occlusions, new content), blending turns off. Lower = safer, higher = stronger smoothing. |

**Outputs:** `frames` (IMAGE)

### Difforum · Detail Guard

`Difforum_DetailGuard`

Unsharp + contrast + grain, for frames that went soft.

| input | type | default | notes |
|---|---|---|---|
| `image` | IMAGE |  | An image or a batch of frames. |
| `sharpen` | FLOAT | 0.3 | Unsharp amount. |
| `contrast` | FLOAT | 1.0 | Contrast multiplier. |
| `grain` | FLOAT | 0.0 | Film grain amount. |
| `grain_mode` | choice (gaussian, plasma) | gaussian | gaussian = fine film grain, plasma = soft organic noise. |

**Outputs:** `image` (IMAGE)

### Difforum · Upscale (2K / 4K)

`Difforum_Upscale`

Finish at 2K (or 1080p / 1440p / 4K / xN) for delivery.

With an upscale model (Load Upscale Model: 4x-UltraSharp, RealESRGAN,
4x_foolhardy_Remacri...) each frame is upscaled by the model in chunks,
then resized to the exact target; without one it is a clean Lanczos
resize. Aspect is kept and sizes stay even for video codecs. Bypass the
node (or its group) to deliver at render size.

| input | type | default | notes |
|---|---|---|---|
| `frames` | IMAGE |  | The clip to deliver. |
| `target` | choice (2K (2048 long edge), 1080p (1920 long edge), 1440p (2560 long edge), 4K (3840 long edge), x1.5, x2, ...) | 2K (2048 long edge) | Final size. 2K = 2048 on the long edge; xN = multiply. |
| `method` | choice (lanczos, bicubic, bilinear, area) | lanczos | Resize filter after the model. lanczos is sharpest. |
| `sharpen` | FLOAT | 0.15 | Light unsharp after resizing. |
| `chunk` | INT | 16 | Frames per model pass; lower it if VRAM runs out. |
| `upscale_model` *(optional)* | UPSCALE_MODEL |  | Load Upscale Model (RealESRGAN, 4x-UltraSharp, Remacri...). Empty = resize only. |

**Outputs:** `frames` (IMAGE), `info` (STRING)
