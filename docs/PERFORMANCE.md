# Quality at few steps, and Apple Silicon

Two problems that look separate and are not. A feedback render is hundreds of
sampler calls in a row, so anything that costs a little per frame costs a lot
per clip - and on unified memory the ceiling arrives long before the clock does.
The same handful of levers fixes both.

Measured on the 0.x Film Director graph (the v1 engine adds cadence crossfades, so the same settings are no slower), 768x432, 120
frames, SDXL + DMD2, M-series with 24 GB unified:

| stage | time | share |
|---|---|---|
| Feedback Sampler | 344 s | 90% |
| KSampler (frame 0) | 24 s | 6% |
| everything else | 16 s | 4% |

Only the first row is worth optimising. The rest is noise.

---

## Before anything: launch flags and step scaling

- **Launch flags.** `--lowvram`, `--novram` and `--disable-smart-memory` (useful for
  MiniMax H3 / LTX on 24 GB) make ComfyUI re-stage the image model for *every*
  frame of a feedback render. SDXL fits easily on a 24 GB card: start ComfyUI
  without these flags for Feedback / Live renders. The run report warns when they
  are on.
- **Steps scale with the energy** (`step_scaling = by energy`, the default): a
  frame at denoise 0.5 runs 10 of 20 steps, as in Deforum. `fixed` runs all steps.

## The cost model

```
render time  ~  frames / cadence  x  steps  x  (pixels / throughput)
```

Four terms, and all four are yours to set. The order below is by ratio of time
saved to quality lost - work down it and stop when the clip looks right.

### 1. Cadence - the free one

`cadence = 2` diffuses every second frame; the in-between frames ride the camera
warp alone. Half the sampler calls, and the motion often reads *smoother*
because the warp interpolates cleanly where the sampler would have re-rolled
texture.

- `1` every frame - only worth it for very fast camera moves
- `2` the default, near-invisible cost
- `3` fine for slow moves; watch for the warp softening
- `4+` the tween frames start to look like a slideshow of warps

Above `3`, raise `sharpen` a little to compensate for the extra warp softening.

### 2. Few-step models - the big one

A 4-step distill at 5 steps beats a base model at 26 steps *for this workload*,
because feedback rewards consistency between frames far more than it rewards
per-frame detail. The base model's extra steps mostly re-roll texture the next
frame will overwrite anyway.

| recipe | steps | cfg | sampler / scheduler |
|---|---|---|---|
| SDXL + DMD2 4-step LoRA | 4-6 | 1.0-1.5 | `lcm` / `sgm_uniform` |
| SDXL Lightning 4/8-step | 4-8 | 1.0-2.0 | `euler` / `sgm_uniform` |
| SD-Turbo / SDXL-Turbo | 1-4 | 1.0 | `euler_ancestral` / `sgm_uniform` |
| SD1.5 + LCM-LoRA | 4-8 | 1.0-2.0 | `lcm` / `sgm_uniform` |

Two mistakes to avoid: cfg above ~2 on a distilled model burns contrast and
compounds every frame; and a scheduler other than `sgm_uniform` wastes steps at
the noise levels a distill was trained for.

**Render frame 0 on the base model, loop on the distill.** Frame 0 sets the
composition every later frame inherits, so it is worth 26 steps once. The
templates already wire it this way - the `LoraLoader` feeds the Feedback
Sampler, while the frame-0 `KSampler` takes the checkpoint directly.

### 3. Resolution - quadratic, so it bites

Attention memory grows with the square of the pixel count. Dropping 768x432 to
640x360 is 30% fewer pixels; 512x288 is 55% fewer.

Feedback renders survive low resolution better than single images do, because
the loop keeps re-imprinting structure. Render at 512-640 wide and upscale the
finished batch - an upscale pass over N frames costs far less than diffusing
those frames larger.

Keep both dimensions divisible by 8. For any Wan 2.2 path, 16.

### 4. Strength - the one that also fixes flicker

Every 0.1 of denoise is real sampler work. Lower strength is both faster and
more coherent; too low and the loop stops evolving and mushes.

- `0.40-0.50` calm, coherent, cheap - the range most clips want
- `0.55-0.65` active transformation, for climax beats
- `> 0.7` the frame stops being a continuation of the last one

The Director's mood presets already sit in this band. `energy_bias`
shifts the whole clip at once - the fastest single knob for a speed/quality
trade.

---

## Apple Silicon

### fp8 does not work, and the error is confusing

MPS has no fp8 kernels. An fp8 checkpoint fails deep inside dequantisation:

```
TypeError: Trying to convert Float8_e4m3fn to the MPS backend
```

That is not a Difforum error and no setting fixes it. Use **bf16/fp16** or
**GGUF**. A bf16 SDXL is ~5 GB and fits comfortably; the same model in fp8 will
not run at all.

### Launch flags

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 \
python main.py --use-pytorch-cross-attention
```

The watermark ratio removes PyTorch's allocation ceiling so the whole unified
pool is reachable. Without it a 24 GB machine refuses allocations while several
GB are still free.

### Budget your memory honestly

Unified memory is shared with the OS and every open app. On 24 GB, plan for
about **16-18 GB of usable headroom**, and remember the model is resident *plus*
the frame batch is accumulating in RAM as the render proceeds.

| model | weights | realistic max resolution |
|---|---|---|
| SD1.5 fp16 | ~2 GB | 1024x576 |
| SDXL bf16 | ~5 GB | 768x432 |
| SDXL bf16 + ControlNet | ~7 GB | 640x360 |
| Flux fp16 | ~24 GB | does not fit - use GGUF Q4/Q5 |

If a render dies partway with `MPS backend out of memory`, the batch grew past
the ceiling. Use the Feedback Sampler's `start_frame` / `end_frame` to render in
chunks - the schedules stay absolutely indexed, so chunks line up exactly.

### fp16 saves memory, not time

M-series ALUs run fp16 and fp32 at similar rates, so half precision buys
headroom rather than speed. The real lever on a Mac is **step count**, which is
why the Turbo and DMD2 templates are the ones to start from.

### Rough throughput, 768x432 SDXL

| setup | per diffused frame | 120 frames at cadence 2 |
|---|---|---|
| base SDXL, 26 steps | ~14 s | ~14 min |
| DMD2, 5 steps | ~2.9 s | ~3 min |
| DMD2, 5 steps, 512x288 | ~1.3 s | ~1.3 min |

---

## Spend the render time on the right take

The cheapest optimisation is not rendering the wrong clip. Two nodes exist for
that:

- **Storyboard (no diffusion)** runs the whole camera chain with the sampler
  removed and returns a contact sheet plus a drift readout. Under a second for a
  120-frame clip. Fix the movement here.
- **Camera Path Preview** draws the trajectory. In `2d` mode the top-down view
  is nearly empty by design - 2D motion is zoom and roll, not travel - so read
  it in `3d`, or read the Storyboard instead.

A clip re-rendered three times because the camera was wrong costs more than
every setting on this page combined.

---

## Long-running installations

For a piece that projects for hours, render once at the highest quality the
machine allows and loop it properly:

- **Difforum · Camera (keys)** with `loop_mode = harmonic` makes the path periodic, so the
  last frame flows into the first with the velocity matched.
- Render **3 laps** of the cycle and keep the last with **Loop Take**. The early
  laps let the feedback image settle onto its cycle.
- Nothing is blended, so there is no dissolve for the eye to find on repeat.

`example_workflows/06_seamless_loop.json` is wired this way. Three laps
means three times the render for a clip that plays forever - the right trade for
an installation, the wrong one while you are still exploring.
