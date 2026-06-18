"""
Model/VRAM profile resolver for Difforum.

Picks sane render settings (model variant, quantization, resolution cap,
segment length, steps, offload) for the user's GPU so the same workflow runs
from a 12 GB card up to 32 GB+. This is the "integrate the options to run
Difforum" layer: downstream render nodes read a profile instead of forcing the
user to hand-tune a dozen knobs.

Pure data + logic, no torch import (the node passes vram_gb in). Numbers are
conservative defaults meant to *fit and run*, not to max out quality.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

FAMILIES = ("wan22", "sd15_animatediff", "ltxv", "sdxl")
QUALITIES = ("fast", "balanced", "quality")


@dataclass
class RenderProfile:
    family: str
    label: str                # human name of the chosen model variant
    repo_hint: str            # where to get it / suggested filename
    quant: str                # gguf quant or precision
    width: int
    height: int
    segment_frames: int       # frames generated per video segment
    overlap: int              # frames shared between segments (for chaining)
    steps: int
    use_lightning_lora: bool  # 4-step distill LoRA for Wan/LTXV
    cfg: float
    offload: str              # "none" | "model" | "sequential"
    vae_tiling: bool
    attention: str            # "sage" | "sdpa"
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


# tier table keyed by minimum VRAM (GB). Picked by largest floor <= vram.
# Each entry is a dict of overrides merged onto the family base.
_WAN22_TIERS = {
    12: dict(label="Wan 2.2 5B (GGUF Q5_K_M)", quant="Q5_K_M",
             repo_hint="QuantStack/Wan2.2-TI2V-5B-GGUF (loader: ComfyUI-GGUF)",
             width=640, height=640, segment_frames=49, overlap=8,
             offload="sequential", vae_tiling=True,
             notes="5B fits comfortably; pair lightx2v/Wan2.2-Lightning 4-step LoRA."),
    16: dict(label="Wan 2.2 14B (GGUF Q4_K_M)", quant="Q4_K_M",
             repo_hint="QuantStack/Wan2.2-I2V-A14B-GGUF (loader: ComfyUI-GGUF)",
             width=832, height=480, segment_frames=65, overlap=12,
             offload="model", vae_tiling=True,
             notes="14B is MoE (high+low noise); Q4 fits w/ offload. 5B is the safer alt."),
    24: dict(label="Wan 2.2 14B (GGUF Q5_K_M)", quant="Q5_K_M",
             repo_hint="QuantStack/Wan2.2-I2V-A14B-GGUF (loader: ComfyUI-GGUF)",
             width=1280, height=720, segment_frames=81, overlap=16,
             offload="model", vae_tiling=True,
             notes="720p comfortable; load both high+low noise experts; bump steps for quality."),
    32: dict(label="Wan 2.2 14B (fp8)", quant="fp8_e4m3fn",
             repo_hint="Comfy-Org/Wan_2.2 repackaged fp8",
             width=1280, height=720, segment_frames=81, overlap=16,
             offload="none", vae_tiling=False,
             notes="fp8 native, no offload; best quality/speed balance."),
}

_SD15_TIERS = {
    12: dict(label="SD1.5 + AnimateLCM", quant="fp16",
             repo_hint="SD1.5 checkpoint + AnimateLCM motion module + LCM LoRA",
             width=512, height=512, segment_frames=16, overlap=4,
             offload="none", vae_tiling=False,
             notes="The lightweight AnimateDiff look; 4-8 step LCM."),
    24: dict(label="SD1.5 + AnimateDiff v3", quant="fp16",
             repo_hint="SD1.5 checkpoint + mm_sd15_v3 motion module",
             width=768, height=768, segment_frames=32, overlap=8,
             offload="none", vae_tiling=False,
             notes="Longer context windows, higher res."),
}

_LTXV_TIERS = {
    12: dict(label="LTX-Video 0.9 (GGUF)", quant="Q5_K_M",
             repo_hint="Lightricks LTX-Video + ComfyUI-LTXVideo",
             width=768, height=512, segment_frames=65, overlap=8,
             offload="model", vae_tiling=True,
             notes="Fastest previews; great for iterating schedules."),
    24: dict(label="LTX-Video 0.9 (fp16)", quant="fp16",
             repo_hint="Lightricks LTX-Video",
             width=1216, height=704, segment_frames=97, overlap=16,
             offload="none", vae_tiling=False, notes="High-res, fast."),
}

_SDXL_TIERS = {
    12: dict(label="SDXL (feedback Classic+)", quant="fp16",
             repo_hint="any SDXL checkpoint (+ SDXL-Turbo/LCM LoRA for fast)",
             width=768, height=768, segment_frames=1, overlap=0,
             offload="model", vae_tiling=True,
             notes="Classic deforum feedback loop, per-frame img2img. 'fast' needs a Turbo/LCM LoRA."),
    24: dict(label="SDXL (feedback Classic+)", quant="fp16",
             repo_hint="any SDXL checkpoint (+ SDXL-Turbo/LCM LoRA for fast)",
             width=1024, height=1024, segment_frames=1, overlap=0,
             offload="none", vae_tiling=False, notes="Higher-res feedback. 'fast' needs a Turbo/LCM LoRA."),
}

_TABLES = {
    "wan22": _WAN22_TIERS,
    "sd15_animatediff": _SD15_TIERS,
    "ltxv": _LTXV_TIERS,
    "sdxl": _SDXL_TIERS,
}

# steps/cfg/lightning per quality, applied on top of the tier
_QUALITY = {
    "fast":     dict(steps=4,  cfg=1.0, use_lightning_lora=True),
    "balanced": dict(steps=6,  cfg=1.5, use_lightning_lora=True),
    "quality":  dict(steps=20, cfg=4.0, use_lightning_lora=False),
}


def _pick_tier(table: dict[int, dict], vram_gb: float) -> dict:
    floors = sorted(table)
    chosen = floors[0]
    for f in floors:
        if vram_gb >= f:
            chosen = f
    return table[chosen]


def resolve_profile(
    vram_gb: float,
    family: str = "wan22",
    quality: str = "balanced",
    attention: str = "sdpa",
) -> RenderProfile:
    """Resolve a RenderProfile for the given GPU and model family."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}, pick from {FAMILIES}")
    if quality not in QUALITIES:
        raise ValueError(f"unknown quality {quality!r}, pick from {QUALITIES}")

    tier = _pick_tier(_TABLES[family], float(vram_gb))
    q = _QUALITY[quality]

    # SD1.5/SDXL don't use the Wan/LTXV lightning distill LoRA
    use_ll = q["use_lightning_lora"] and family in ("wan22", "ltxv")
    # quality mode needs more steps on classic diffusion families
    steps = q["steps"]
    if not use_ll and family in ("wan22", "ltxv") and quality != "quality":
        steps = max(steps, 8)
    # plain SDXL has no built-in few-step distill -> 4 steps would be garbage
    # unless the user adds an SDXL-Turbo/LCM LoRA. Keep a usable floor.
    if family == "sdxl" and quality == "fast":
        steps = max(steps, 8)

    return RenderProfile(
        family=family,
        label=tier["label"],
        repo_hint=tier["repo_hint"],
        quant=tier["quant"],
        width=tier["width"],
        height=tier["height"],
        segment_frames=tier["segment_frames"],
        overlap=tier["overlap"],
        steps=steps,
        use_lightning_lora=use_ll,
        cfg=q["cfg"],
        offload=tier["offload"],
        vae_tiling=tier["vae_tiling"],
        attention=attention,
        notes=tier.get("notes", ""),
    )


def summarize(p: RenderProfile) -> str:
    ll = "yes" if p.use_lightning_lora else "no"
    return (
        f"{p.label}  [{p.quant}]\n"
        f"  res {p.width}x{p.height}  segment {p.segment_frames}f overlap {p.overlap}f\n"
        f"  steps {p.steps}  cfg {p.cfg}  lightning {ll}\n"
        f"  offload {p.offload}  vae_tiling {p.vae_tiling}  attn {p.attention}\n"
        f"  get: {p.repo_hint}\n"
        f"  {p.notes}"
    )
