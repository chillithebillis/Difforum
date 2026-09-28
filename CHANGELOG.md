# Changelog

## 1.0.0

Difforum becomes a direction layer: a visual timeline drives any renderer.

### New
- **Restyle + ControlNet**: optional `control_net` / `control_image` / `control_strength`
  hold the source structure so Restyle can run at higher denoise (template 13 uses DA3
  depth + a union SDXL ControlNet).
- **Template 14 · AnimateDiff look on any video**: AnimateDiff-Evolved vid2vid with
  sliding 16-frame context, depth + canny ControlNets, 2K upscale and source audio.
- **Template 15 · AnimateDiff LCM (fast, hi-res)**: AnimateLCM motion module + LoRA
  in 8 steps, then a switchable x1.5 hi-res pass in 6 steps.
- Templates 14 and 15 use Advanced-ControlNet nodes, which work inside AnimateDiff's
  sliding context window (the core ControlNet nodes raise an error there).
- **Director (timeline)**: a multi-track editor with Scenes (prompt + mood), Camera
  (visual move picker, speed, amount, lens, easing, per-block audio reaction) and
  Energy (a drawable denoise curve), plus an animated camera preview computed by the
  render engine. A single `direction` wire feeds the renderers.
- **Setup** with target model grids: LTX 8k+1, MiniMax H3 17k+5 @ 24 fps, Wan 4k+1.
- Video model bridges: **Guide Frames**, **Keyframes**, **Camera → Prompt**,
  **LTX Guides** (core `LTXVAddGuide`), **H3 Guides** (core `MiniMaxH3AddGuide`, multi-keyframe)
  and **H3 Shot** (first/last frame, segments).
- **Fill Reveal (AI)**: inpaints what the camera uncovers (core `InpaintModelConditioning`,
  any image model), on the frames a video model will see.
- **Looks** on the Director (cinematic, documentary, deforum_morph, animatediff_dream,
  psychedelic, music_video, stop_motion, hand_drawn): one choice sets the feedback pass and the
  video-model prompt.
- **Previz everywhere**: the **Animatic** node renders the whole shot in seconds with timecode,
  prompt, move, energy and keys burnt in; every template has one, and the Director's
  **Previz only** button mutes the render outputs so Queue runs only the previz.
- **Keys track** on the Director (point markers at exact frames), timeline **zoom and scroll**
  for long shots, and **Keyframe Images** for multikeyframing: the Feedback Sampler travels
  through your pictures (`key_pull`, `key_approach`), H3 / LTX Guides anchor them.
- **Look Mix**: detail transfer, colour, flicker cuts or crossfade from a feedback pass onto
  an H3 / LTX render. New looks: disco_diffusion, vqgan_clip, flicker_experimental.
- Template **12 · long shot with key moments** (30 s, installations).
- **Hover help on every input** (one registry, `nodes/tooltips.py`, also in docs/NODES.md).
- **H3 structured prompts**: Camera → Prompt and H3 Shot write the whole Director timeline (look,
  scenes, camera in H3 vocabulary, key events, soundscape, music) in MiniMax H3's native format,
  with the I2VA / FL2VA alignment lines or the full-reference sections.
- H3 templates: **Keyframe Polish** (low-denoise re-paint of the warped keyframes) and **Depth**
  (core Depth Anything 3) blocks. Template 03 uses core Depth Anything 3 too.
- **Restyle**: gives any H3 / LTX / live-action clip the Deforum / AnimateDiff / Disco look with
  an image model, feedback carried along the clip's optical flow. Template **13 · restyle any video**.
- **Upscale (2K / 4K)**: upscale model + exact resize, chunked for long clips.
- MiniMax H3 templates render in two stages: a switchable **H3 Latent Upscale (x2)** block
  (Minimax H3 Latent Upscaler 3D + a short H3 refine at full size), then Upscale 2K in pixels.
- **Workflow Switches**: one switch per group (render groups muted, pass-through groups bypassed,
  dependent outputs muted with them) and ⌖ to jump to a group.
- Every template rebuilt in named blocks (Control, Direction, Previz, Models, Render, Restyle,
  Upscale 2K, Output); H3 templates add a **Live Preview (TAEH3)** block (KJNodes Model Preview
  Override).
- **Camera Export** to After Effects (.jsx), Blender (.py) and JSON; **Camera Import**
  from them, with helper exporters in `tools/`.
- 7 new moves: tilt up/down, crane up, handheld, drift, breathe, vortex.
- Feedback Sampler outputs the tracked depth per frame and a run report; its fine-tuning
  moved to a separate **Render Options** node, so the sampler stays small.
- 13 templates in ComfyUI's template browser, generated from the node definitions.

### Improved
- One shared feedback engine behind the Feedback Sampler, Live Sampler and Storyboard.
- Depth follows the image in 3D, and pseudo-3D replaces the silent freeze without depth.
- Steps scale with the energy, like Deforum (about 2x faster); the run report shows
  seconds per frame and warns when low-VRAM launch flags make ComfyUI reload the model
  every frame.
- Cadence crossfades between keys. Revealed areas are repainted with extra noise.
- Colour anchoring per scene. Lazy prompt travel. Ring buffer in the Live Sampler.
- Deterministic 3D z-buffer on CUDA / MPS / CPU.

### Fixed
- H3 two-stage refine (templates 08, 10, 11): guides and first / last frames were
  reused at the first-pass size and the refine stopped with a shape mismatch. The new
  **H3 Refine Guides** node re-encodes them at the upscaled size.
- Widgets named `mode` collided with the node's own mode in the ComfyUI frontend (mute / bypass
  stopped working on those nodes). Renamed: Camera (keys) `camera_mode`, Look Mix `blend`,
  Symmetry `symmetry`, Audio Curve `combine`. Saved workflows keep their values.
- The pack no longer imports a top-level `core` module (it collided with other packs).
- The 0.x templates `audio_reactive_video`, `deluxe` and `mesmerize` had camera values
  in the wrong fields. The v1 templates are generated, so this cannot recur.
- `feedback_classic` and `hybrid_wan_guides` rendered a static camera (3D move, no depth).

### Removed / deprecated
- 0.x nodes are kept under Difforum/legacy (hidden, deprecated) until 2.0. See
  docs/MIGRATION.md.
- Model Profile / Model Catalog → docs/MODELS.md. Load/Save Video → ComfyUI core
  nodes. Look, Grade, Glow, Glitch, Datamosh → legacy only.
- Media moved to `docs/media/` as small WebP files. The published package is ~40 MB lighter.
