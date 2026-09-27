# Architecture

```
Setup ──params──► Director ──direction──► Feedback / Live Sampler ──frames──► Post / Save
                     │                     Storyboard
                     │                     Guide Frames ─► Keyframes ─► LTX Guides / H3 Shot
                     └──────────────────► Camera → Prompt, Camera Export
```

## Layers

| Layer | Where | Depends on |
|---|---|---|
| Engine | `core/` | numpy, torch. No ComfyUI. |
| Nodes | `nodes/` | the engine, plus ComfyUI imported lazily inside functions |
| UI | `js/director_timeline.js` | the `/difforum/catalog` and `/difforum/preview` routes (`nodes/routes.py`) |
| Legacy | `legacy/` | the engine; 0.x nodes flagged `DEPRECATED` |

Every module imports the engine *relatively* (`from ..core import ...`). The
0.x pack put itself on `sys.path` and imported a top-level `core`, which
collided with any other custom node that did the same.

## Core modules

- `schedule.py`, `expr.py`: the Deforum keyframe syntax, evaluated by a
  whitelisted AST walker (no `eval`).
- `camera.py`: per-frame 4x4 deltas and accumulated poses. A delta acts on the
  previous view: `P_f = d_f · P_{f-1}`.
- `camera_presets.py`: the 25 moves as per-axis expressions, plus UI metadata.
- `camera_keys.py`: blends moves between keyframes (easing, lens channel).
- `direction.py`: the timeline model (scenes / camera / energy / guidance),
  audio reactions, and camera → words. `build_direction()` has no ComfyUI or torch
  dependency, which is why it can back the live preview endpoint.
- `engine.py`: `FeedbackEngine` + `iter_feedback()`, the one loop behind the
  Feedback Sampler, the Live Sampler and the Storyboard. Diffusion is a callback.
- `warp.py`: the 2D affine and 3D depth warps, with a deterministic z-buffer
  (`scatter_reduce`), depth re-projection, affine composition and pseudo-3D.
- `export.py`: camera ↔ JSON / After Effects / Blender.
- `prompt.py`: prompt travel. `PromptTrack` blends lazily and indexes like a list.
- `color.py`, `detail.py`, `symmetry.py`, `flow.py`, `effects.py`, `loop.py`,
  `audio.py`, `plot.py`, `storyboard.py`: the pixel and signal tools.

## Units and conventions

- Images are `[B,H,W,C]` in 0..1 (ComfyUI `IMAGE`). Depth maps: white = near.
- 2D camera: pixels per frame, degrees per frame, and zoom as a per-frame scale.
- 3D camera: rotations in degrees per frame. Lateral translation is in pixels at the
  reference depth (so 2D and 3D pans agree), and z translation is in depth units
  (`near`..`far`, default 1..100).
- Exported cameras use OpenCV axes (x right, y down, z forward), with world = the
  first camera.

## Keeping things in sync

- `tools/build_workflows.py` builds `example_workflows/` from the node
  definitions, so widget order can never drift. CI runs it with `--check`.
- `tools/build_docs.py` builds `docs/NODES.md` from docstrings and tooltips.
  CI runs `--check` on it as well.
