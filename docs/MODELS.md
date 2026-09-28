# Models that work well

Difforum takes a plain `MODEL` / `VAE` / `CONDITIONING`, so any image model
ComfyUI loads can drive the Feedback Sampler. These are good starting points.
Check each model's page for current versions and licences.

## Feedback Sampler (the Deforum look)

| Goal | Model | Settings |
|---|---|---|
| Quality | SDXL finetunes (e.g. Juggernaut XL, RealVis XL) | 20-28 steps, dpmpp_2m karras, cfg 5-7, energy 0.4-0.6 |
| Speed | SDXL + DMD2 4-step LoRA, or Lightning | 4-8 steps, lcm / euler, cfg 1-1.5, cadence 2 |
| Flux | Flux.1 dev / schnell (fp16 / GGUF on Mac) | 8-20 steps, cfg 1, use FluxGuidance on the positive prompt |
| Stylised / light | SD1.5 finetunes (DreamShaper, etc.) | 20 steps, cfg 7. On some builds, launch with `--force-fp32` if frames come out black |
| Live | SDXL-Turbo, SD-Turbo, LCM LoRA | 1-2 steps, cfg ~1, 512 px |

A depth model for 3D parallax: **Depth Anything 3**, built into ComfyUI (Load Depth
Anything 3 → Run → Render, `depth_anything_3_mono_large.safetensors` in
`models/geometry_estimation`). Its `v2_style` render is near = white, Difforum's
convention. Depth Anything V2 ([Kijai's pack](https://github.com/kijai/ComfyUI-DepthAnythingV2))
still works on any `depth` input.

## Video models (via the bridges)

| Model | Bridge | Notes |
|---|---|---|
| LTX-2 / 2.5 | **LTX Guides** | Setup target LTX. Runs on Apple Silicon with bf16 / GGUF (no fp8 on MPS). |
| MiniMax H3 | **H3 Shot** | 24 fps, 17k+5 frames, up to 20 s per generation. |
| Wan 2.x VACE | **Guide Frames** → VACE control video + masks | Setup target Wan (4k+1, 16 px). |

## Apple Silicon

- fp8 checkpoints do not run on MPS. Use fp16 / bf16 or GGUF.
- Launch with `--use-pytorch-cross-attention`, and
  `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` to unlock unified memory.
- Few-step models are the real speed lever. Setup's `max_megapixels` keeps long
  renders inside memory (about 0.35 MP on 16 GB, 0.65 on 24 GB, 1.0 on 32 GB).
