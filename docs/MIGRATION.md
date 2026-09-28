# Migrating from Difforum 0.x

Difforum 1.0 goes from 40 nodes to 33 and makes the Director timeline the main
way to drive a shot. Nothing breaks on load: every 0.x node is still
registered, hidden from the node search and moved to **Difforum/legacy**, so
old workflows open and run. Legacy nodes will be removed in 2.0.

## Node map

| 0.x | 1.0 |
|---|---|
| Anim Setup, Anim Setup+ | **Setup** (seconds or frames, aspect, target model grid) |
| Film Director | **Director (timeline)**: separate Scenes / Camera / Energy tracks. It outputs the camera, energy and prompts directly (one `direction` wire) instead of text for other nodes to parse |
| Camera Keys | **Camera (keys)**, same syntax, and `loop_mode` included |
| Camera Shots | **Camera (keys)** with `transition 0` (hard cuts) |
| Camera Move (presets) | **Camera (keys)** with a single key, or one Director block |
| Seamless Camera | **Camera (keys)** with `loop_mode` |
| Camera (advanced) | **Camera (expressions)**, where each axis also accepts a schedule socket (e.g. an Audio Curve) |
| Camera Path Preview, Warp (2D/3D), Storyboard | **Storyboard** (sheet + frames + camera path, same engine as the sampler), or **Animatic** for a previz video |
| Schedule / Sample Schedule | **Schedule** |
| Schedule Info / Schedule Plot | **Schedule Plot** (image + text) |
| Audio Analyzer | **Audio Analyzer** (+ `offset_seconds`, a bands plot) |
| Audio Schedule (reactive) | **Audio Curve**, or a per-block audio reaction in the Director |
| Prompt Schedule / Prompt Scenes | **Prompt Travel** (`2.5s:` timestamps allowed) or the Director's Scenes track |
| Prompt Batch (→ AnimateDiff) | **Prompt Travel** with `build_batched` |
| Feedback Sampler | **Feedback Sampler** (see "behaviour changes" below) |
| Live Sampler, Live Step | **Live Sampler** |
| Guide Builder (Wan VACE) | **Guide Frames** (mask convention and hole fill selectable) |
| Loop Take, Loop Blend, Ping-Pong | **Loop** (`method`) |
| Symmetry, Echo Trails, Detail Guard, Flow Stabilize | same names, under Post |
| Model Profile, Model Catalog | [MODELS.md](MODELS.md): recommendations that go stale belong in docs, not in nodes |
| Load Video, Save Video | ComfyUI core **Load Video** / **Create Video** + **Save Video** (these keep audio) |
| VJ Look, Colour Grade, Glow, Glitch, Datamosh | a dedicated grading / glitch pack; out of scope for a direction tool |

## Behaviour changes in the Feedback Sampler

- **3D without depth** used to freeze dolly / orbit moves silently. It now runs a
  pseudo-3D fallback and reports it.
- **3D lateral moves** are measured in pixels at the reference depth, the same unit
  as 2D pans. A 0.x 3D graph with `translation_x` will pan less; raise
  `translation_scale` if you relied on the old magnitude.
- **The depth map follows the image** (`depth_tracking = follow`). Set `static` for
  the 0.x behaviour.
- **Cadence** in-betweens are crossfaded between keys instead of being plain warps.
- **Steps scale with the energy** (`step_scaling = by energy`): a frame at denoise 0.5
  runs half the steps, like Deforum. Set `fixed` on Render Options for the 0.x cost.
- **Colour anchor** defaults to `scene`. Set `anchor_mode = first` for 0.x.
- **The camera pose** accumulates in view order (`P_f = d_f · P_{f-1}`). Poses from
  combined rotation + translation moves differ slightly from 0.x; per-frame
  deltas are unchanged.
- **Inputs:** `strength_schedule` → `strength`, `cfg_schedule` → `cfg_curve`,
  `positive_schedule` → `prompts`. Colour, detail, symmetry, depth calibration
  and chunking moved to **Render Options**; without it the Director's style sets
  the look.
