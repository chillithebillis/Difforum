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
  **LTX Guides** (core `LTXVAddGuide`) and **H3 Shot** (first/last frame, segments).
- **Camera Export** to After Effects (.jsx), Blender (.py) and JSON; **Camera Import**
  from them, with helper exporters in `tools/`.
- 7 new moves: tilt up/down, crane up, handheld, drift, breathe, vortex.
- Feedback Sampler outputs the tracked depth per frame and a run report; its fine-tuning
  moved to a separate **Render Options** node, so the sampler stays small.
- 9 templates in ComfyUI's template browser, generated from the node definitions.

### Improved
- One shared feedback engine behind the Feedback Sampler, Live Sampler and Storyboard.
- Depth follows the image in 3D, and pseudo-3D replaces the silent freeze without depth.
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
