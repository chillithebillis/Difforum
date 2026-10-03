# Recipes

Ten short projects, each a template plus a shot script from `example_scripts/`.
Load the template, put a **Shot Script** node on the Director's `timeline_in`
(template 16 has one wired) and type the script name in `file`, for example
`examples/01_infinite_zoom.txt`. Or open the `.txt`, read it and draw the same
beats by hand. Every script is plain text: copy one, change the words, queue.

| # | recipe | template | script | what it shows |
|---|---|---|---|---|
| 1 | Infinite zoom | 02 / 05 | `01_infinite_zoom` | the Deforum classic: scenes morph as the camera falls in |
| 2 | Music video on the beat | 04 | `02_music_video_beat` | camera blocks that react to the track (pulse, shake, bass speed) |
| 3 | Seamless loop for VJ / LED | 06 | `03_seamless_loop_vj` | camera and prompt close on themselves |
| 4 | A still with real depth | 03 | `04_parallax_still` | 2.5D move on a photo or key visual, low energy keeps the picture |
| 5 | Cinematic move on H3 | 08 | `05_cinematic_push_in` | one frame, a drawn camera, H3 does the motion |
| 6 | Storyboard to shot | 16 | `06_storyboard_to_shot` | your stills as keyframes, no look pass |
| 7 | AnimateDiff look on footage | 15 | `07_animatediff_dance` | AnimateLCM over a clip you shot; the script is a list of styles to try |
| 8 | Deforum morph over any clip | 13 | `08_deforum_morph_restyle` | the look changes over time, the motion stays the clip's own |
| 9 | One-minute installation shot | 12 | `09_installation_long_shot` | a long take with key moments that must land |
| 10 | Previz and camera hand-off | 01, 09 | `10_previz_camera_export` | block, show and export to After Effects / Blender with no model |

## Writing the script with an LLM

Shot Script's `llm_instructions` output is a prompt that asks for this format
with the clip length and the valid camera moves. Feed it, plus one line about
the film, to any text node, and wire the answer into `script_in`:

```
LLM node (prompt = llm_instructions + "A storm arrives at a lighthouse.")
   -> Shot Script.script_in -> Director.timeline_in
```

Set the Director's `external` to `text only` to keep a camera you drew and take
only the words, or `replace` to let the script direct everything.

## Fast first, quality last

A long render is almost always one of four things. Reference figures from a
12 GB card: about 2 s per diffused frame (SDXL turbo, 4 steps, 640 px) and about
14 s per MiniMax H3 step (640 px, 124 frames).

| what costs | fast setting | what it saves |
|---|---|---|
| H3 at 20 steps | Turbo LoRA at 4 steps (on in the templates) | 5x on the H3 pass |
| H3 Latent Upscale (refine at 2x) | off while exploring, on for the final | about half the H3 time |
| Restyle on every frame | off while exploring; `cadence` 2 when on | one image-model pass per frame |
| a 4x upscale model on large frames | Upscale `model_use = auto` skips it under 2x | most of the upscale time |

Rough budgets with those settings:

| shot | route | diffusions | order of time |
|---|---|---|---|
| 10 s Deforum look, 24 fps | template 02 / 05, turbo model, cadence 2 | 120 | ~5 min |
| 60 s Deforum look, 12 fps | template 12, turbo model, cadence 3 | 240 | ~10 min |
| 5 s H3 shot | template 08 / 16, Turbo LoRA | 4-8 H3 steps | ~1-2 min + model loading |
| 5 s H3 with the look | template 11 as shipped (look pass cadence 2) | 62 + 4 H3 steps | ~4 min + model loading |

For long pieces:

- **Feedback look**: render at 12 fps and 640 px with cadence 2-3, then Upscale
  2K; interpolate to 24 fps with any frame-interpolation node if the piece
  needs it.
- **H3**: the model is trained on 5-15 s clips. Split a long shot into segments
  with H3 Shot (`segment`), each starting on the last frame of the one before.
- **Launch flags**: `--cache-none` makes every Queue run the whole graph again,
  including text encoding and passes that did not change; without it, changing
  only the Restyle re-runs only the Restyle. `--disable-smart-memory` unloads
  the image model between the VAE and the sampler; Difforum suspends it while a
  frame is made so both stay loaded (set `DIFFORUM_RESPECT_MEMORY_FLAGS=1` to
  opt out).
- **Previz first**: the Animatic costs seconds. Switch Render off (or press
  *Previz only* on the Director) until the timing is right.
