# Changelog

## 1.0.0

Difforum becomes a direction layer: a visual timeline drives any renderer.

### New
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
- **Camera Export** to After Effects (.jsx), Blender (.py) and JSON; **Camera Import**
  from them, with helper exporters in `tools/`.
- 7 new moves: tilt up/down, crane up, handheld, drift, breathe, vortex.
- Feedback Sampler outputs the tracked depth per frame and a run report; its fine-tuning
  moved to a separate **Render Options** node, so the sampler stays small.
- 12 templates in ComfyUI's template browser, generated from the node definitions.

### Improved
- One shared feedback engine behind the Feedback Sampler, Live Sampler and Storyboard.
- Depth follows the image in 3D, and pseudo-3D replaces the silent freeze without depth.
- Steps scale with the energy, like Deforum (about 2x faster); the run report shows
  seconds per frame and warns when ComfyUI was launched with `--lowvram` /
  `--disable-smart-memory`, which reload the model every frame.
- Cadence crossfades between keys. Revealed areas are repainted with extra noise.
- Colour anchoring per scene. Lazy prompt travel. Ring buffer in the Live Sampler.
- Deterministic 3D z-buffer on CUDA / MPS / CPU.

### Fixed
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
