"""
Curated model catalog for Difforum.

Maps the classic, widely-used models (the ones people already have on disk and
know how to train) to Difforum's hybrid modes, with download sources and LoRA
training tooling. Goal: integration = "pick a recipe", not "go hunt files".

Philosophy:
- SD1.5 is the most trainable ecosystem on the planet (huge Civitai LoRA scene,
  8-12GB LoRA training) -> the easy-to-train classic path.
- AnimateDiff motion modules give the recognizable animated look on top of any
  SD1.5 checkpoint -> reuse, don't reinvent.
- Wan 2.2 (GGUF) is the modern high-quality path; trainable via musubi-tuner /
  diffusion-pipe / ai-toolkit, or hosted on fal.ai.

Sources are either a verified URL or a Civitai/HF *search* hint (we avoid
inventing exact model IDs we aren't sure of). Pure data, no torch. Tested.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ROLES = ("checkpoint", "motion_module", "lora", "vae", "video_model", "controlnet")


@dataclass(frozen=True)
class ModelEntry:
    key: str
    name: str
    role: str
    family: str            # sd15 | sdxl | wan22 | ltxv
    source: str            # verified URL or search hint
    trainable: str         # how to train, or "no (use LoRA on base)"
    notes: str
    tags: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return asdict(self)


# --- catalog ----------------------------------------------------------------
# Verified IDs: DreamShaper=4384, epiCRealism XL=277058 (from Civitai search).
# Others use a Civitai/HF search URL on purpose to stay accurate.
_CATALOG: tuple[ModelEntry, ...] = (
    # ---- SD1.5 checkpoints (the classic, most-trainable base) ----
    ModelEntry(
        "dreamshaper8", "DreamShaper 8", "checkpoint", "sd15",
        "https://civitai.com/models/4384/dreamshaper",
        "LoRA: kohya sd-scripts / OneTrainer (~8GB)",
        "13.7M downloads. Versatile illustration/semi-real/fantasy. Great AnimateDiff base.",
        ("versatile", "classic", "popular"),
    ),
    ModelEntry(
        "realisticvision", "Realistic Vision V6", "checkpoint", "sd15",
        "https://civitai.com/search/models?query=Realistic%20Vision",
        "LoRA: kohya sd-scripts / OneTrainer (~8GB)",
        "Best-known SD1.5 photoreal base. Clip skip 2.",
        ("photoreal", "classic", "popular"),
    ),
    ModelEntry(
        "toonyou", "ToonYou", "checkpoint", "sd15",
        "https://civitai.com/search/models?query=ToonYou",
        "LoRA: kohya sd-scripts / OneTrainer (~8GB)",
        "Cartoon/anime look. Classic AnimateDiff pairing (clip skip 2, cfg 8).",
        ("toon", "anime", "animatediff"),
    ),
    ModelEntry(
        "epicrealism", "epiCRealism (Natural Sin)", "checkpoint", "sd15",
        "https://civitai.com/search/models?query=epiCRealism",
        "LoRA: kohya sd-scripts / OneTrainer (~8GB)",
        "Popular photoreal SD1.5; soft natural lighting.",
        ("photoreal", "popular"),
    ),
    # ---- SD1.5 motion modules (the AnimateDiff look) ----
    ModelEntry(
        "mm_sd15_v3", "AnimateDiff v3 (mm_sd15_v3)", "motion_module", "sd15",
        "https://huggingface.co/guoyww/animatediff",
        "no (motion module; train SparseCtrl/LoRA on top)",
        "Best general SD1.5 motion module. Works with AnimateDiff-Evolved.",
        ("animatediff", "motion", "recommended"),
    ),
    ModelEntry(
        "animatelcm", "AnimateLCM (sd15_t2v)", "motion_module", "sd15",
        "https://huggingface.co/wangfuyun/AnimateLCM",
        "no (use with LCM LoRA)",
        "4-8 step fast motion. Pair with LCM LoRA + LCM sampler/beta schedule. Ideal 12GB.",
        ("animatediff", "fast", "lcm", "recommended"),
    ),
    ModelEntry(
        "animatediff_lightning", "AnimateDiff-Lightning", "motion_module", "sd15",
        "https://huggingface.co/ByteDance/AnimateDiff-Lightning",
        "no",
        "Distilled 1-8 step motion from ByteDance. Very fast previews.",
        ("animatediff", "fast", "distill"),
    ),
    # ---- Wan 2.2 (modern high quality, GGUF for low VRAM) ----
    ModelEntry(
        "wan22_5b_gguf", "Wan 2.2 TI2V 5B (GGUF)", "video_model", "wan22",
        "https://huggingface.co/QuantStack/Wan2.2-TI2V-5B-GGUF",
        "LoRA: musubi-tuner / diffusion-pipe (~16GB) or fal.ai hosted",
        "Fits 12GB with Q5_K_M. Flagship low-VRAM hybrid engine. Load via ComfyUI-GGUF.",
        ("wan", "gguf", "lowvram", "recommended"),
    ),
    ModelEntry(
        "wan22_14b_gguf", "Wan 2.2 I2V A14B (GGUF)", "video_model", "wan22",
        "https://huggingface.co/QuantStack/Wan2.2-I2V-A14B-GGUF",
        "LoRA: musubi-tuner / diffusion-pipe (~24GB) or fal.ai hosted",
        "Higher quality; MoE high+low noise. Q4 fits 16GB w/ offload, Q5 great at 24GB.",
        ("wan", "gguf", "quality"),
    ),
    ModelEntry(
        "wan22_lightning_lora", "Wan 2.2 Lightning LoRA (4-step)", "lora", "wan22",
        "https://huggingface.co/lightx2v/Wan2.2-Lightning",
        "n/a (distill LoRA)",
        "Cuts Wan to ~4 steps, no CFG. Separate high/low-noise LoRAs (rank64). Essential for fast renders.",
        ("wan", "fast", "distill", "recommended"),
    ),
    # ---- LTX-Video (fast preview / iteration) ----
    ModelEntry(
        "ltxv_09", "LTX-Video 0.9", "video_model", "ltxv",
        "https://huggingface.co/Lightricks/LTX-Video",
        "LoRA: diffusion-pipe",
        "Fastest open video; perfect for iterating schedules before final Wan render.",
        ("ltxv", "fast", "preview"),
    ),
    # ---- SDXL (modern quality, big LoRA ecosystem, works in feedback now) ----
    ModelEntry(
        "juggernaut_xl", "Juggernaut XL", "checkpoint", "sdxl",
        "https://civitai.com/search/models?query=Juggernaut%20XL",
        "LoRA: kohya sd-scripts / OneTrainer (~12GB)",
        "Top general SDXL finetune. Great Classic+ base at 768-1024px.",
        ("sdxl", "photoreal", "popular"),
    ),
    ModelEntry(
        "realvisxl", "RealVisXL V5", "checkpoint", "sdxl",
        "https://civitai.com/search/models?query=RealVisXL",
        "LoRA: kohya sd-scripts / OneTrainer (~12GB)",
        "Photoreal SDXL; crisp detail.",
        ("sdxl", "photoreal"),
    ),
    ModelEntry(
        "sdxl_lightning", "SDXL-Lightning (2/4/8-step LoRA)", "lora", "sdxl",
        "https://huggingface.co/ByteDance/SDXL-Lightning",
        "n/a (distill LoRA)",
        "2-8 step distill for SDXL -> fast feedback / near-realtime. cfg 1-2.",
        ("sdxl", "fast", "distill", "recommended"),
    ),
    ModelEntry(
        "dmd2", "DMD2 (SDXL 1-4 step)", "lora", "sdxl",
        "https://huggingface.co/tianweiy/DMD2",
        "n/a (distill LoRA)",
        "Newer distill, strong quality at 4 steps. Great for live SDXL.",
        ("sdxl", "fast", "distill"),
    ),
    # ---- Flux (best open quality; fits 24GB in fp8/GGUF) ----
    ModelEntry(
        "flux_dev", "FLUX.1-dev (fp8)", "checkpoint", "flux",
        "https://huggingface.co/Comfy-Org/flux1-dev (fp8) or black-forest-labs/FLUX.1-dev",
        "LoRA: ai-toolkit / SimpleTuner / kohya (~24GB or hosted)",
        "Best open image quality. fp8 fits a 4090. Heavier per step.",
        ("flux", "quality", "recommended"),
    ),
    ModelEntry(
        "flux_schnell", "FLUX.1-schnell (4-step)", "checkpoint", "flux",
        "https://huggingface.co/black-forest-labs/FLUX.1-schnell",
        "LoRA: ai-toolkit / SimpleTuner",
        "Distilled Flux, 4 steps, Apache-2.0. Quality + speed for feedback.",
        ("flux", "fast", "distill", "recommended"),
    ),
    ModelEntry(
        "flux_gguf", "FLUX.1-dev (GGUF)", "checkpoint", "flux",
        "https://huggingface.co/city96/FLUX.1-dev-gguf",
        "LoRA: ai-toolkit / SimpleTuner",
        "Quantized Flux for lower VRAM (Q4-Q8). Load via ComfyUI-GGUF.",
        ("flux", "gguf", "quality", "lowvram"),
    ),
    # ---- ControlNets for structure / illusions (great for VJ feedback) ----
    ModelEntry(
        "qr_monster_sd15", "QR Code Monster (SD1.5)", "controlnet", "sd15",
        "https://huggingface.co/monster-labs/control_v1p_sd15_qrcode_monster",
        "n/a (ControlNet)",
        "Turns ANY grayscale pattern (spiral/mandala/logo/mask) into structure -> "
        "hidden-image illusions. Feed a fixed pattern to the Feedback Sampler's "
        "control_image to lock the illusion while the scene morphs. strength ~1.0-1.4.",
        ("sd15", "controlnet", "illusion", "vj", "recommended"),
    ),
    ModelEntry(
        "qr_monster_sdxl", "QR Code Monster (SDXL)", "controlnet", "sdxl",
        "https://huggingface.co/monster-labs/control_v1p_sdxl_qrcode_monster",
        "n/a (ControlNet)",
        "SDXL version of the illusion/pattern ControlNet. Pair with an SDXL base.",
        ("sdxl", "controlnet", "illusion", "vj"),
    ),
    # ---- SD 3.5 ----
    ModelEntry(
        "sd35_large", "Stable Diffusion 3.5 Large", "checkpoint", "sd35",
        "https://huggingface.co/stabilityai/stable-diffusion-3.5-large",
        "LoRA: kohya sd-scripts / SimpleTuner",
        "Stability's modern model; strong prompt adherence. fp8 on 24GB.",
        ("sd35", "quality"),
    ),
)

_BY_KEY = {m.key: m for m in _CATALOG}


# --- curated recipes (a ready stack per use-case) ---------------------------
@dataclass(frozen=True)
class Recipe:
    key: str
    title: str
    family: str
    parts: tuple[str, ...]   # ModelEntry keys
    why: str

    def as_dict(self) -> dict:
        d = asdict(self)
        d["models"] = [_BY_KEY[k].as_dict() for k in self.parts]
        return d


_RECIPES: tuple[Recipe, ...] = (
    Recipe("sd15_fast_anim", "SD1.5 fast animation (AnimateLCM)", "sd15",
           ("dreamshaper8", "animatelcm"),
           "Lightest path, runs on 12GB, 4-8 steps. The recognizable AnimateDiff look."),
    Recipe("sd15_toon_anim", "SD1.5 toon animation", "sd15",
           ("toonyou", "mm_sd15_v3"),
           "Cartoon/anime motion; classic ToonYou + AnimateDiff v3."),
    Recipe("sd15_real_anim", "SD1.5 photoreal animation", "sd15",
           ("realisticvision", "mm_sd15_v3"),
           "Photoreal frames with smooth v3 motion."),
    Recipe("wan22_12gb", "Wan 2.2 hybrid (12GB)", "wan22",
           ("wan22_5b_gguf", "wan22_lightning_lora"),
           "Modern quality on a 12GB card; 5B GGUF + 4-step Lightning."),
    Recipe("wan22_24gb", "Wan 2.2 hybrid (24GB)", "wan22",
           ("wan22_14b_gguf", "wan22_lightning_lora"),
           "720p 14B GGUF; best quality/perf on prosumer cards."),
    Recipe("ltxv_preview", "LTX-Video preview", "ltxv",
           ("ltxv_09",),
           "Ultra-fast iteration of camera/audio schedules before the final render."),
    Recipe("sdxl_quality", "SDXL quality (Classic+)", "sdxl",
           ("juggernaut_xl",),
           "Modern SDXL feedback at 768-1024px. Big step up from SD1.5."),
    Recipe("sdxl_fast", "SDXL fast / near-realtime", "sdxl",
           ("juggernaut_xl", "sdxl_lightning"),
           "SDXL + Lightning = 2-8 step feedback, great quality/speed. cfg 1-2."),
    Recipe("flux_quality", "Flux quality (4090)", "flux",
           ("flux_dev",),
           "Best open image quality in the feedback loop. fp8 fits 24GB."),
    Recipe("flux_fast", "Flux fast (schnell)", "flux",
           ("flux_schnell",),
           "4-step Flux: high quality with low step count for animation."),
    Recipe("sd35_quality", "SD 3.5 Large", "sd35",
           ("sd35_large",),
           "Stability's modern model; strong prompt adherence."),
    Recipe("sd15_illusion", "SD1.5 illusion / hidden-pattern (QR Monster)", "sd15",
           ("dreamshaper8", "qr_monster_sd15"),
           "Lock a spiral/logo/mask in the scene via QR Monster ControlNet while "
           "the feedback loop morphs around it - the psychedelic illusion look."),
)

_RECIPE_BY_KEY = {r.key: r for r in _RECIPES}


# --- training tooling per family --------------------------------------------
TRAINING = {
    "sd15": {
        "tools": ["kohya_ss/sd-scripts", "OneTrainer"],
        "vram_gb": 8,
        "kind": "LoRA / DreamBooth",
        "notes": "Easiest training in the ecosystem. 10-30 images for a style/char LoRA.",
    },
    "sdxl": {
        "tools": ["kohya_ss/sd-scripts", "OneTrainer"],
        "vram_gb": 12,
        "kind": "LoRA / DreamBooth",
        "notes": "Same toolchain as SD1.5, a bit heavier.",
    },
    "wan22": {
        "tools": ["musubi-tuner (kohya)", "diffusion-pipe", "ai-toolkit", "fal.ai (hosted)"],
        "vram_gb": 16,
        "kind": "video/image LoRA",
        "notes": "musubi-tuner & diffusion-pipe train Wan2.1/2.2 LoRA locally; fal.ai if no GPU.",
    },
    "ltxv": {
        "tools": ["diffusion-pipe"],
        "vram_gb": 16,
        "kind": "LoRA",
        "notes": "Lighter than Wan; fast to train.",
    },
    "flux": {
        "tools": ["ai-toolkit", "SimpleTuner", "kohya_ss/sd-scripts", "fal.ai (hosted)"],
        "vram_gb": 24,
        "kind": "LoRA",
        "notes": "ai-toolkit is the easiest Flux LoRA path; ~24GB local or fal.ai hosted.",
    },
    "sd35": {
        "tools": ["kohya_ss/sd-scripts", "SimpleTuner"],
        "vram_gb": 24,
        "kind": "LoRA",
        "notes": "SD3.5 LoRA via kohya/SimpleTuner.",
    },
}


# --- query helpers ----------------------------------------------------------
def all_entries() -> tuple[ModelEntry, ...]:
    return _CATALOG


def get(key: str) -> ModelEntry:
    return _BY_KEY[key]


def by_family(family: str) -> list[ModelEntry]:
    return [m for m in _CATALOG if m.family == family]


def recipes(family: str | None = None) -> list[Recipe]:
    if family is None:
        return list(_RECIPES)
    return [r for r in _RECIPES if r.family == family]


def get_recipe(key: str) -> Recipe:
    return _RECIPE_BY_KEY[key]


def recipe_keys() -> list[str]:
    return [r.key for r in _RECIPES]


def summarize_recipe(key: str) -> str:
    r = get_recipe(key)
    lines = [f"{r.title}  [{r.family}]", f"  {r.why}", "  models:"]
    for k in r.parts:
        m = _BY_KEY[k]
        lines.append(f"   - {m.name} ({m.role})")
        lines.append(f"       get:   {m.source}")
        lines.append(f"       train: {m.trainable}")
    t = TRAINING.get(r.family)
    if t:
        lines.append(f"  train this family: {', '.join(t['tools'])} (~{t['vram_gb']}GB) - {t['notes']}")
    return "\n".join(lines)
