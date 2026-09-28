# Getting quality out of Difforum

Every input has hover help inside ComfyUI (rest the mouse on it). This page
explains how the pieces work together: which dials matter, where the image
quality really comes from, and the paths that give the best results.

## The dials that matter

| Dial | Where | What it does | Start at |
|---|---|---|---|
| **Energy** | Director (curve), Feedback Sampler | Denoise per frame: how much each frame is re-imagined. 0.3 steady, 0.5 classic Deforum morph, 0.7+ the image changes every few frames. The moods set it automatically; draw the curve to override. | 0.40-0.55 |
| **energy_bias** | Director | Moves the whole curve up or down. | 0 |
| **cadence** | Feedback Sampler, Restyle | Diffuse every Nth frame; the rest ride the camera and crossfade. | 2 |
| **key_pull** | Feedback Sampler | How strongly a frame is pulled toward the next key picture before it is diffused. 1 = lands exactly on it, 0.6 = strong resemblance, 0.3 = a hint. | 0.65 |
| **key_approach** | Feedback Sampler | Frames before a key over which the pull ramps up. 6-12 = sudden arrival, 24-48 = the image slowly becomes the key. | 12 (24+ on long shots) |
| **transition** | Director | 1 = camera blocks ease into each other (one continuous move), 0 = hard cuts. | 1 |
| **camera_scale** | Director | Master multiplier on every move. | 1 |
| **every_seconds** | Keyframes | Spacing of the keyframes taken from the clip. | 1-2 |
| **max_guides** | H3 Guides | How many of those keyframes are anchored inside H3. | 4 (6-8 for a look pass) |
| **denoise** | Restyle | How much each H3 frame is re-painted in the look. | 0.35-0.5 |

## How many keyframes, and where

Three places decide the keyframes a video model sees:

1. **Keys track** on the Director: exact moments you mark. Keyframe Images pins
   your own pictures to them.
2. **Keyframes** node: takes frames from a clip every `every_seconds` (or at
   explicit `indices` such as `0, 48, 96, -1`) and snaps them to the model's
   latent grid (H3 17k+5, LTX 8k+1).
3. **H3 Guides** `max_guides`: of those, keeps the first, the last and the best
   spread in time.

More guides = H3 follows your frames closely (and inherits their flaws). Fewer =
more of H3's own motion and invention.

## Keyframe quality is output quality

H3 copies the keyframes it is given: their sharpness, palette and texture. The
guide keyframes come from warping one image along the camera, so they soften the
further the camera travels, and the areas the camera uncovers are empty. The H3
templates fix both before the keyframes reach H3:

1. **Depth (Depth Anything 3)**: real parallax, so the warp moves near and far
   things correctly instead of sliding the picture.
2. **Fill Reveal (AI)**: an image model paints the uncovered areas (an inpainting
   checkpoint gives the cleanest seams).
3. **Keyframe Polish**: Restyle in `clean restyle` at denoise 0.35 re-paints each
   keyframe with the image model: detail comes back, composition stays.

Further gains, in order of effect:

- **Start from a great first frame.** It is the reference and frame 0; render it
  at the size you will guide at, in the final look.
- **Polish with the model and prompt of the final look** (the Fill prompt). A
  turbo SDXL at 6 steps is enough.
- **Use your own pictures** as keys (Keyframe Images) for the moments that
  matter: hand-made stills beat warped frames.
- **Keep keyframes consistent**: same model, same prompt and seed for all of
  them, or H3 will interpolate between styles.

## Prompt dynamics from the Director

Camera → Prompt (format **H3 structured**) and H3 Shot (prompt_style **H3
structured**) write the whole timeline in MiniMax H3's native prompt format:

- the **look** opens `[Shot 1]` as the style,
- each **scene** prompt becomes what is on screen at that point (a
  transformation in one continuous take, or a new `[Shot N]` at its cut time with
  `cuts` on),
- each **camera block** becomes H3 camera vocabulary with amplitude and speed
  (push in, truck, arc, pedestal...),
- each **key label** becomes the event at that moment ("the doors open"),
- `soundscape` and `music` fill `overall_soundscape` and `non_diegetic_music`.

`h3_mode` adds the exact alignment line (I2VA, FL2VA) or the six full-reference
sections for ref2va with H3 Guides. Write scene prompts and key labels as things
that can be seen or heard ("glowing moss on the roots", not "a sense of wonder").

## Depth

- Set the Director to **3d** and connect a depth map to Guide Frames / the
  Feedback Sampler: near things move faster than far ones.
- Core **Depth Anything 3** (`depth_anything_3_mono_large` in
  `models/geometry_estimation`) is wired in templates 03, 08 and 10; its
  `v2_style` render is near = white, as Difforum expects.
- The Feedback Sampler outputs the **tracked depth per frame**: use it for
  relighting, fog or DOF in comp.
- Too much parallax: lower `translation_scale` (Render Options / Guide Frames).

## Cameras from After Effects and Blender

- **Out:** Camera Export writes an After Effects `.jsx` (a 3D camera, or a null in
  2d mode) and a Blender `.py`, so titles and 3D lock to the render.
- **In:** block the move in AE or Blender (`tools/aftereffects/Difforum Export
  Camera.jsx`, `tools/blender/difforum_export_camera.py`), put the `.json` in
  `ComfyUI/input`, and Camera Import turns it into a Difforum camera for Guide
  Frames, the Storyboard or the Feedback Sampler. A tracked plate from AE's 3D
  camera tracker works the same way: the AI shot follows the live-action move.

## Speed and memory

- H3 templates render in **two stages**: 640 px, then the H3 Latent Upscaler x2
  and a 4-step refine. Much faster than rendering at full size.
- Live Preview (TAEH3) shows the video forming; switch it off for the last few
  percent of speed.
- Feedback renders: launch ComfyUI **without** `--lowvram` /
  `--disable-smart-memory` (they reload the model every frame); keep those flags
  for H3 / LTX.
- Previz first (Animatic), render second: the Switches panel turns Render off.

## Known gaps

What Difforum does not do yet, and the workaround:

| Gap | Today |
|---|---|
| The energy curve does not reach H3 (it is a denoise for image models) | Use Restyle with `follow_energy` on: the look pass breathes with the curve |
| Depth is estimated from the first frame only | Depth follows the image in the Feedback Sampler; for H3, re-run depth on a polished keyframe for long moves |
| Clips longer than one H3 generation (~20 s) are chained by hand (H3 Shot `segment`) | Render segment 0, feed its last frame as the next first frame |
| Depth Anything 3 multiview can estimate camera poses from a video | Not wired yet: export the camera from AE's tracker and use Camera Import |
| Restyle has no temporal model (AnimateDiff motion module) | The flow-carried feedback gives most of the coherence; Flow Stabilize after it smooths the rest |
