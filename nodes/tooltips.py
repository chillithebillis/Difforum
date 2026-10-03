"""Hover help for every Difforum input, kept in one place.

ComfyUI shows an input's `tooltip` when the mouse rests on it. Writing them
here (instead of scattered through INPUT_TYPES) keeps the wording consistent
and lets the docs reuse it. `apply()` merges them into each node's
INPUT_TYPES at registration; an entry here wins over an inline tooltip.
"""

from __future__ import annotations

import functools

# Wording shared by many nodes (used when a node has no entry of its own).
COMMON = {
    "params": "From Setup: frame count, fps, width and height. Plug the same Setup everywhere.",
    "direction": "The Director's wire: camera, energy curve, scene prompts, keys and look in one cable.",
    "camera": "A camera track from Camera (keys), Camera (expressions) or Camera Import. "
              "Overrides the Director's camera.",
    "model": "The image model (SD1.5 / SDXL / Flux / turbo). MODEL output of your checkpoint or LoRA chain.",
    "vae": "The VAE that matches the model.",
    "positive": "Positive conditioning (CLIP Text Encode).",
    "negative": "Negative conditioning (CLIP Text Encode).",
    "steps": "Sampler steps for a full denoise.",
    "cfg": "Prompt strength. Turbo / LCM / DMD2 models want 1-2; regular SDXL 4-7.",
    "sampler_name": "Sampler. euler / dpmpp_2m for regular models, euler_ancestral or lcm for turbo.",
    "scheduler": "Noise schedule. karras / normal for regular models, sgm_uniform for turbo.",
    "seed": "Noise seed. Same seed = same result.",
    "audio": "From Audio Analyzer: per-frame loudness, bands and beats.",
    "depth": "A depth map (white = near), e.g. Depth Anything V2 on the first frame. Needed for real "
             "3D parallax; without it 3D moves fall back to pseudo-3D.",
    "init_image": "The first frame. Everything grows from this image.",
    "frames": "A batch of frames (a clip).",
    "image": "An image or a batch of frames.",
    "prompts": "Prompt travel from Prompt Travel (or the Director with CLIP connected).",
    "fps": "Frames per second.",
    "key_images": "Keyframe Images: pictures pinned to moments of the clip.",
    "key_indices": "Frame numbers of those pictures (Keyframe Images 'indices').",
    "keyframes": "Keyframe pictures (from Keyframes or Keyframe Images).",
    "options": "Difforum · Render Options: manual colour, detail, symmetry, depth and chunk settings. "
               "Without it the Director's look decides.",
    "strength": "A schedule for denoise per frame; replaces the Director's energy curve.",
    "indices": "Frame number of each keyframe, comma separated, e.g. 0,48,96,123.",
}

TIPS = {
    "Difforum_Setup": {
        "target": "The model you will render with. Snaps length and size to its grid: H3 17k+5 frames "
                  "@ 24 fps and 32 px, LTX 8k+1 and 32 px, Wan 4k+1 and 16 px, feedback 8 px.",
        "duration_mode": "Give the length in seconds or in frames.",
        "duration": "Clip length (seconds or frames). The target grid may round it slightly.",
        "fps": "Frames per second. MiniMax H3 is always 24.",
        "aspect": "Frame shape. 'custom' uses custom_aspect_w / h.",
        "long_edge": "Pixels on the longer side before snapping. H3 two-stage templates render at 640 "
                     "and upscale the latent 2x; the feedback look is fine at 768-1024.",
        "seed": "Master seed for every sampler that reads Setup.",
        "custom_aspect_w": "Width part of a custom aspect (e.g. 2.39).",
        "custom_aspect_h": "Height part of a custom aspect (e.g. 1).",
        "max_megapixels": "Hard cap on frame area. 0 = no cap. ~0.35 on a 16 GB Mac, ~0.65 on 24 GB.",
    },
    "Difforum_Director": {
        "params": "From Setup: the timeline length and fps.",
        "timeline": "Written by the timeline editor below (scenes, camera, keys, energy). Edit it there.",
        "timeline_in": "A timeline from outside: Shot Script, an LLM node, a text loader. Plain shot lines "
                       "('0s | calm | dolly_in slow | prompt'), CSV or Director JSON. The editor shows the "
                       "result after each run; disconnect to edit by hand.",
        "images": "Pictures pinned to moments of the clip, in order (a batch: Keyframe Assets, Scene Stills, "
                  "Batch Images). They travel on the direction wire: the Feedback Sampler passes through "
                  "them, H3 / LTX Guides anchor them, the Animatic shows them.",
        **{f"image_{i}": f"Picture {i}, any size (cropped to the canvas). Pinned in order, after `images`."
           for i in range(1, 7)},
        "images_at": "Where the pictures land: the start of each scene, the Keys markers, or spread evenly "
                     "from the first to the last frame.",
        "external": "How timeline_in meets the drawn timeline. replace: all from outside. text only: scenes "
                    "and keys from outside, your drawn camera stays. camera only: the reverse. add to drawn: "
                    "both, outside wins on the same frame.",
        "camera_mode": "2d: flat moves (zoom, pan, roll). 3d: real parallax when a depth map reaches the "
                       "renderer; 3D moves without depth become pseudo-3D.",
        "look": "One aesthetic for every renderer. Feedback Sampler: colour lock, detail, grain and extra "
                "energy. H3 / LTX: a style sentence at the head of the prompt.",
        "transition": "How much of each camera block eases into the next. 1 = one continuous crane move, "
                      "0 = hard cuts between blocks.",
        "camera_scale": "Master multiplier on every move. 0.5 = everything half as far, 2 = twice.",
        "energy_bias": "Moves the whole energy (denoise) curve up or down. +0.1 = more re-imagining "
                       "everywhere.",
        "variation": "0 = exactly as drawn. Above 0, each block gets small seeded changes in speed and "
                     "amount - useful for variations of the same shot.",
        "variation_seed": "Seed for those variations: change it for another take.",
        "clip": "Connect the CLIP of your image model to encode the scene prompts: the Feedback Sampler "
                "and Restyle then travel through them.",
        "audio": "From Audio Analyzer: enables the per-block audio reactions (beat pulse, bass speed...).",
    },
    "Difforum_Camera": {
        "keys": "One key per line: frame: move [speed] [intensity] [lens°] [ease]. "
                "e.g. '48: orbit_right 1.2 0.8 35 ease_in_out'. Frames can be '2.5s'.",
        "camera_mode": "2d = flat moves; 3d = parallax with a depth map.",
        "transition": "1 = moves blend into each other; 0 = hard cuts at each key.",
        "speed": "Master speed on every key.",
        "loop_mode": "harmonic / zero_mean make the path periodic over cycle_frames, so the clip loops "
                     "with no crossfade (installations). off = free path.",
        "cycle_frames": "Frames per loop. 0 = the whole clip is one cycle. Feeds Loop 'keep settled lap'.",
        "harmonics": "harmonic loop: how many sine terms describe the path. More = closer to the keys, "
                     "fewer = smoother loop.",
    },
    "Difforum_CameraExpr": {
        "mode": "2d = flat moves; 3d = parallax with a depth map.",
        "fov": "Lens field of view in degrees. 20-30 long lens, 40 normal, 60+ wide.",
        "translation_x": "Horizontal move per frame. Deforum syntax: '0:(0), 60:(2*sin(t/10))'; t = frame.",
        "translation_y": "Vertical move per frame, same syntax.",
        "translation_z": "Forward (+) / back (-) per frame. 3D only.",
        "rotation_3d_x": "Tilt, degrees per frame. 3D only.",
        "rotation_3d_y": "Pan (turn), degrees per frame. 3D only.",
        "rotation_3d_z": "Roll, degrees per frame.",
        "zoom": "Scale per frame. 1.0 = none, 1.01 = 1% closer each frame.",
        "translation_x_curve": "A schedule (Audio Curve, Schedule) that replaces the text for this axis.",
        "translation_y_curve": "A schedule that replaces the text for this axis.",
        "translation_z_curve": "A schedule that replaces the text for this axis.",
        "rotation_3d_x_curve": "A schedule that replaces the text for this axis.",
        "rotation_3d_y_curve": "A schedule that replaces the text for this axis.",
        "rotation_3d_z_curve": "A schedule that replaces the text for this axis.",
        "zoom_curve": "A schedule that replaces the text for this axis.",
        "audio": "Audio curves usable in expressions: amp, low, mid, high, onset, beat.",
    },
    "Difforum_Storyboard": {
        "init_image": "Any still: the camera move is played over it.",
        "every_nth": "One contact-sheet cell every N frames.",
        "columns": "Cells per row on the contact sheet.",
        "preview_scale": "Working size vs. Setup. 0.5 = half size, much faster.",
        "symmetry": "Preview an in-loop symmetry (kaleidoscope...) as the Feedback Sampler would apply it.",
    },
    "Difforum_KeyframeImages": {
        "images": "One picture per key, in time order (Batch Images). These pictures set the look and the "
                  "content the render passes through, so make them good: same style, same resolution.",
        "times": "Override the Director's Keys: '0, 4s, 9.5s' or frame numbers. Empty = use the Keys "
                 "track; no keys = spread evenly.",
    },
    "Difforum_TravelConditioning": {
        "direction": "A Director with a CLIP connected: its scene prompts become one prompt per frame.",
        "prompts": "Or the output of a Prompt Travel node.",
        "images": "Optional: the frames that will be sampled; the travel is stretched to their count.",
        "latent": "Optional: the latent batch that will be sampled; the travel is stretched to its count.",
    },
    "Difforum_ShotScript": {
        "script": "One beat per line: TIME | MOOD | CAMERA | PROMPT. TIME 0s, 4.5s, 00:09 or f96. CAMERA a "
                  "move (dolly_in, orbit_left, crane_up...) plus slow / fast, small / large, 35mm, an easing. "
                  "'4s | key: label' adds a key. Lines without a time are spread evenly.",
        "file": "A .txt / .csv / .json in ComfyUI/input (e.g. shots/scene01.txt), or a shipped example: "
                "examples/01_infinite_zoom.txt. Re-read whenever it changes. Overrides the box.",
        "params": "From Setup: converts seconds to frames and flags beats past the end.",
        "script_in": "Text from another node (an LLM, a text file loader, a spreadsheet export). Overrides "
                     "the box and the file. Feed the llm_instructions output to the LLM as its prompt.",
    },
    "Difforum_KeyframeAssets": {
        "folder": "A folder inside ComfyUI/input holding the stills (png / jpg / webp), sorted by name.",
        "timing": "Director keys: the pictures land on the Keys markers in order. filename: '0s_x.png', "
                  "'4.5s_x.png', 'f096.png' or '0096_x.png'. spread evenly: first to last over the clip.",
        "fit": "cover crops to the canvas, contain pads with gray (the masks output lets Fill Reveal paint "
               "the bars), stretch distorts.",
        "direction": "Gives the Keys track and the canvas size.",
        "params": "Canvas size and fps when no direction is connected.",
        "images": "A batch from other nodes instead of the folder (one per key, in order).",
        "times": "Override: '0, 4s, 9.5s' or frame numbers.",
    },
    "Difforum_SceneStills": {
        "direction": "The Director timeline: one still per scene that has a prompt.",
        "model": "An image model (SDXL, SD1.5, Flux...).",
        "style": "The look shared by every still, written once: medium, light, lens, palette. It goes in "
                 "front of each scene prompt.",
        "negative": "What no still should show.",
        "steps": "Sampling steps per still. 24 for a base model, 4-8 with a turbo / DMD2 model.",
        "cfg": "Prompt strength. 5-6 for a base model, 1-2 for turbo / DMD2.",
        "seed": "Same seed for every still, so they share composition habits and texture.",
        "continuity": "How much each still is painted over the one before. 0 = every still from scratch "
                      "(free composition, the look can jump). 0.5 = palette and layout carry over. 1 = small "
                      "changes only.",
        "long_edge": "Size the stills are made at, in the canvas aspect. Use the image model's own size "
                     "(1024 SDXL, 768 SD1.5): the video model scales them down itself.",
        "first_image": "Your own picture as the first still; the rest are painted after it.",
    },
    "Difforum_Animatic": {
        "direction": "The Director to previz.",
        "preview_scale": "Size vs. Setup. 0.4 = small and instant; 1.0 = full size.",
        "overlay": "Burn in timecode, camera move, scene prompt, energy bar and key markers.",
        "init_image": "First frame to move over. Empty = a grid plate.",
        "depth": "Depth of the first frame, for real parallax in 3d mode.",
        "key_images": "Keyframe Images: shown at their moments.",
        "key_indices": "Their frame numbers.",
    },
    "Difforum_Schedule": {
        "schedule": "Values over time: 'frame:(value)', e.g. '0:(0.45), 60:(0.6)'. Frames can be '2s'. "
                    "Expressions allowed: t = frame, amp/low/beat with audio.",
        "easing": "How values move between keys.",
    },
    "Difforum_SchedulePlot": {
        "schedule": "Any schedule or curve to draw.",
        "width": "Plot width in pixels.",
        "height": "Plot height in pixels.",
    },
    "Difforum_AudioAnalyzer": {
        "audio": "Your track (Load Audio).",
        "smoothing": "0 = every transient, 0.5+ = slow envelope. 0.2-0.3 reads well on camera.",
        "beat_sensitivity": "Higher = fewer, stronger beats detected.",
        "offset_seconds": "Start reading the track here, to sync with an edit.",
    },
    "Difforum_AudioCurve": {
        "audio_curves": "From Audio Analyzer.",
        "source": "Which band drives the curve: amp (loudness), low / mid / high bands, onset "
                  "(attacks), beat (pulses).",
        "combine": "How the audio is applied to base: add, subtract or multiply.",
        "base": "Value when the band is silent (e.g. 0.45 denoise).",
        "amount": "How far the band pushes the value at its loudest.",
        "smoothing": "Extra smoothing of the band before it is used.",
    },
    "Difforum_PromptTravel": {
        "clip": "CLIP of the image model.",
        "prompts": "One prompt per line: 'frame: text' or '2.5s: text'. Consecutive prompts blend.",
        "easing": "How each prompt blends into the next.",
    },
    "Difforum_FeedbackSampler": {
        "positive": "Look / style prompt. Used for every frame when no scene prompts reach the sampler.",
        "init_image": "The first frame. Everything grows from this image.",
        "steps": "Steps for a full denoise. With step scaling (default) a frame at energy 0.5 runs half "
                 "of them, like Deforum.",
        "cadence": "Diffuse every Nth frame; the frames in between ride the camera and crossfade. 2 is "
                   "almost invisible and twice as fast.",
        "strength": "A schedule that replaces the Director's energy curve (denoise per frame).",
        "cfg_curve": "A schedule for cfg per frame.",
        "control_net": "Optional ControlNet (depth, canny...) applied on every diffused frame.",
        "control_image": "Per-frame control images; empty = the frame itself.",
        "control_strength": "ControlNet strength.",
        "energy": "Denoise per frame when no Director or strength curve is connected. It is how much "
                  "each frame is re-imagined: 0.3 = steady, 0.5 = classic Deforum morph, 0.7+ = the "
                  "image changes every few frames.",
        "key_images": "Keyframe Images: the travel steers toward each picture and lands on it at its "
                      "frame. This is multikeyframing.",
        "key_indices": "Frame numbers of the key pictures (Keyframe Images 'indices').",
        "key_pull": "How strongly the frame is pulled toward the next key picture before diffusion. "
                    "1 = arrives exactly on it, 0.6 = strong resemblance, 0.3 = a hint.",
        "key_approach": "Frames before a key over which the pull ramps up. Short (6-12) = sudden arrival, "
                        "long (24-48) = the image slowly becomes the key. Long shots want long approaches.",
    },
    "Difforum_LiveSampler": {
        "run_frames": "How many frames to run (the loop wraps the camera).",
        "steps": "Steps per frame. Turbo models: 1-4.",
        "cadence": "Diffuse every Nth frame for speed.",
        "target_fps": "Cap the speed. 0 = as fast as possible.",
        "energy": "Denoise per frame when no Director is connected.",
        "loop_camera": "Repeat the camera when the run is longer than the clip.",
        "live_source": "'' = off, '0' = webcam, or a video path: mixed in before each frame.",
        "source_blend": "How much of the live source enters each frame.",
        "stream_dir": "Folder to write live PNG frames to (for another app).",
        "spout_name": "Spout sender name (Windows) for Resolume / TouchDesigner / OBS.",
        "keep_frames": "Frames kept in memory for the output (ring buffer).",
        "live_preview": "Show the frames inside the node while running.",
    },
    "Difforum_RenderOptions": {
        "color_mode": "How colour is matched to the anchor: lab (perceptual), rgb, or none.",
        "sharpen": "Sharpening after each diffused frame; keeps feedback from going soft.",
        "noise": "Grain added before each frame is diffused; feeds new detail.",
        "symmetry": "Symmetry applied inside the loop (kaleidoscope for mandalas).",
        "symmetry_segments": "Kaleidoscope segments.",
        "border": "What fills the edges the camera exposes: reflection (mirror), border (stretch), zeros.",
        "near": "Depth calibration: distance of the nearest depth value.",
        "far": "Depth calibration: distance of the farthest depth value.",
    },
    "Difforum_GuideFrames": {
        "anchor_image": "The image the camera travels over (usually your first frame).",
        "mask_convention": "Which value the mask uses for 'generate': 1 = generate (VACE / LTX / H3 / Fill "
                           "Reveal) or 1 = keep.",
        "hole_fill": "What goes into the area the camera uncovers before AI fills it: gray (neutral, "
                     "best for Fill Reveal and H3), stretched edge, or black.",
        "depth": "Depth of the anchor, for real parallax in 3d mode.",
        "translation_scale": "Strength of 3D translation vs. the depth range.",
    },
    "Difforum_Keyframes": {
        "frames": "The clip to pick keyframes from (Guide Frames, Storyboard, a Feedback render...). "
                  "Keyframes carry its look into H3 / LTX: better frames here = better video.",
        "grid": "Snap keyframes to the video model's latent grid (H3 17k+5, LTX 8k+1, Wan 4k+1) so each "
                "one lands on a real latent frame.",
        "every_seconds": "How many keyframes: one every N seconds. 1 = many, strong guidance; 2 = fewer, "
                         "more freedom; 0 = first and last only. H3 Guides then keeps up to max_guides.",
        "fps": "Frames per second of the clip (from Setup).",
        "indices": "Exact frames instead of spacing, e.g. '0, 48, 96, -1' (-1 = last).",
        "masks": "Guide Frames masks, so Fill Reveal can repaint what the camera uncovered.",
    },
    "Difforum_CameraPrompt": {
        "prefix": "What is on screen. sentence / suffix: put before the camera text. H3 structured: the "
                  "opening composition ([Shot 1]); in reference mode, what <Picture 1> shows.",
    },
    "Difforum_LTXGuides": {
        "positive": "LTX positive conditioning.",
        "negative": "LTX negative conditioning.",
        "vae": "LTX video VAE.",
        "latent": "Empty LTX latent (length and size from Keyframes / Setup).",
        "strength": "Guide strength of the keyframes in between. 0.6-0.8 = follows the path, 1 = locked.",
        "first_strength": "Frame 0 usually locks the look: keep it high.",
        "last_strength": "Strength of the last keyframe. High = the shot must end on it.",
    },
    "Difforum_H3Shot": {
        "frames": "The clip whose first and last frame H3 should connect (Guide Frames, Storyboard, "
                  "Feedback render).",
        "segment_length": "Frames per H3 generation. Clips longer than this are split into segments.",
        "segment": "Which segment to render: 0, then 1 with segment 0's last frame as its first frame...",
        "shot_description": "What is on screen at the start: subject, place, light. Goes into the prompt.",
        "include_look": "Add the Director's look as the style of the shot.",
    },
    "Difforum_H3Guides": {
        "positive": "Positive conditioning from MiniMax H3 Reference to Video.",
        "keyframes": "Keyframe pictures (Keyframes, Keyframe Images or Fill Reveal).",
        "max_guides": "Most keyframes anchored inside the generation. The first and last are kept, the "
                      "rest evenly spaced. More = H3 follows your frames closely; fewer = more of H3's "
                      "own motion and invention. 4 for a camera path, 6-8 for a look pass.",
    },
    "Difforum_H3RefineGuides": {
        "positive": "The same conditioning the first H3 pass used (with its guides / first-last frames).",
        "latent": "The upscaled AV latent (after the H3 Latent Upscaler), so the guides match its size.",
        "mode": "re-encode: guides rebuilt at the refine size (from the original pixels when Difforum "
                "H3 Guides added them). drop guides: the refine follows the upscaled render alone.",
    },
    "Difforum_FillReveal": {
        "images": "Frames with holes where the camera uncovered new area (gray from Guide Frames).",
        "model": "An image model; an inpainting checkpoint (SDXL inpainting, Flux Fill) gives the "
                 "cleanest seams.",
        "frames": "Which frames to paint: all keyframes, or only the first / last (FL2VA).",
        "steps": "Inpainting steps.",
        "feather": "Soft edge, in pixels, where the painted area meets the original.",
        "seed": "Fill seed; change it for another fill.",
    },
    "Difforum_CameraExport": {
        "formats": "Which files to write to output/difforum: After Effects .jsx, Blender .py, JSON.",
        "filename_prefix": "File name prefix.",
    },
    "Difforum_CameraImport": {
        "file": "A camera .json in ComfyUI/input: from Camera Export, the Blender exporter or the After "
                "Effects script in tools/.",
        "retime": "match frames = stretch the move to Setup's length; keep source length = frame for "
                  "frame.",
    },
    "Difforum_Loop": {
        "method": "keep settled lap: render several laps, keep the last (perfect loop). flow crossfade: "
                  "blend the end into the start along motion. ping-pong: forward then back.",
    },
    "Difforum_LookMix": {
        "video": "The video-model render (H3, LTX...). Its motion and light are kept.",
        "blend": "detail transfer: the look pass's fine texture on top. colour + detail: also its "
                 "palette. flicker cuts: cut to the look pass every few frames. crossfade: plain mix.",
        "amount": "How much of the look pass comes through.",
        "flicker_every": "flicker cuts: roughly one cut every N frames.",
        "fps": "Clip fps, for step_fps.",
        "seed": "flicker cuts: which frames flicker.",
        "look_pass": "Frames in the look (Feedback Sampler or Restyle). Resampled to the video's length.",
    },
    "Difforum_Symmetry": {
        "symmetry": "Mirror or kaleidoscope the image.",
        "segments": "Kaleidoscope segments.",
        "mix": "How much of the symmetric image is used.",
        "flip": "Mirror the other way.",
        "center_x": "Symmetry centre, 0-1 across.",
        "center_y": "Symmetry centre, 0-1 down.",
        "angle": "Rotation of the symmetry axes, degrees.",
    },
    "Difforum_EchoTrails": {
        "decay": "How long trails last. 0.9 = long exposure.",
        "mix": "How much of the trail is laid over the frame.",
    },
    "Difforum_FlowStabilize": {
        "strength": "How much history is blended in along motion. 0.3-0.5 calms boiling texture; 0 = off.",
        "flow_scale": "Resolution of the motion estimate. 0.5 = fast, 1.0 = precise.",
        "error_gate": "Where the motion estimate is unsure (occlusions, new content), blending turns off. "
                      "Lower = safer, higher = stronger smoothing.",
    },
    "Difforum_DetailGuard": {
        "sharpen": "Unsharp amount.",
        "contrast": "Contrast multiplier.",
        "grain": "Film grain amount.",
        "grain_mode": "gaussian = fine film grain, plasma = soft organic noise.",
    },
    "Difforum_Restyle": {
        "model": "An image model in the look you want. Turbo SDXL / DMD2 at 4-6 steps is fast and good.",
        "style": "deforum morph: strong feedback, smears along motion. animatediff boil: texture re-rolls "
                 "each frame. disco flicker: high energy, loose colour. clean restyle: steady painted "
                 "version. custom: uses feedback / seed_mode / color_hold below.",
        "cfg": "Prompt strength. Turbo models: 1-2.",
        "seed": "Noise seed.",
        "prompts": "Prompt travel; otherwise the Director's scene prompts, otherwise positive.",
        "flow_scale": "Resolution of the motion estimate that carries the previous frame. 0.5 = fast.",
        "control_net": "A ControlNet that holds the clip's structure while the look is re-painted, so "
                       "denoise can go up to 0.6-0.8 (the real Deforum / AnimateDiff range) without losing "
                       "the shapes. Depth or canny with control_image; tile with the clip itself.",
        "control_image": "Per-frame control maps of the clip (Depth Anything 3, Canny...). Empty = the clip "
                         "frames themselves (tile / union ControlNets).",
        "control_strength": "How hard the structure is held. 0.4 loose, 0.6 balanced, 0.9 locked.",
    },
    "Difforum_Upscale": {
        "frames": "The clip to deliver.",
        "target": "Final size. 2K = 2048 on the long edge; xN = multiply.",
        "method": "Resize filter after the model. lanczos is sharpest.",
        "sharpen": "Light unsharp after resizing.",
        "upscale_model": "Load Upscale Model (RealESRGAN, 4x-UltraSharp, Remacri...). Empty = resize only.",
        "model_use": "auto: the model runs only when the clip grows 2x or more (640 -> 2K); a clip that is "
                     "already large (1280 -> 2K) is resized and sharpened, many times faster. always / never "
                     "force it.",
    },
}


def tooltip(node: str, name: str) -> str | None:
    return TIPS.get(node, {}).get(name) or COMMON.get(name)


def _with_tips(node: str, types: dict) -> dict:
    out = {}
    for section, inputs in types.items():
        if not isinstance(inputs, dict):
            out[section] = inputs
            continue
        sec = {}
        for name, spec in inputs.items():
            tip = TIPS.get(node, {}).get(name)
            if isinstance(spec, tuple) and spec:
                opts = dict(spec[1]) if len(spec) > 1 and isinstance(spec[1], dict) else {}
                if tip or (not opts.get("tooltip") and COMMON.get(name)):
                    opts["tooltip"] = tip or COMMON[name]
                spec = (spec[0], opts, *spec[2:]) if opts else spec
            sec[name] = spec
        out[section] = sec
    return out


def apply(mappings: dict) -> None:
    """Wrap each Difforum node's INPUT_TYPES so the tooltips above are merged in."""
    for node, cls in mappings.items():
        if not node.startswith("Difforum_") or getattr(cls, "_difforum_tips", False):
            continue
        original = cls.INPUT_TYPES

        def wrapped(klass, _orig=original, _node=node):
            return _with_tips(_node, _orig())

        cls.INPUT_TYPES = classmethod(functools.wraps(original)(wrapped))
        cls._difforum_tips = True
