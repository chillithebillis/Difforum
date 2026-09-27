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

## MiniMax H3 (first / last frame)

Template: `08_h3_first_last`.

1. **Setup** target `MiniMax H3` locks 24 fps and the 17k+5 length grid.
2. **H3 Shot** takes the frames (Guide Frames, a Feedback render or a Storyboard)
   and returns, for the chosen `segment`:
   `first_frame`, `last_frame`, `length`, `width`, `height` (32 px grid) and a
   `prompt` built from your shot description plus the camera move of that segment.
3. Wire those into core **MiniMax H3 Image to Video** (mode fl2va). The last frame is
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
