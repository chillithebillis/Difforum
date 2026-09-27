"""
Difforum 0.x nodes, kept so existing workflows still open and run.

Every class here is flagged ``DEPRECATED`` (hidden from the node search in
current ComfyUI frontends) and moved to the ``Difforum/legacy`` category. They
will be removed in Difforum 2.0. See docs/MIGRATION.md for the v1 replacement
of each one.
"""

from __future__ import annotations

import importlib
import logging

log = logging.getLogger("difforum")

_MODULES = (
    "schedule_nodes", "audio_nodes", "hybrid_nodes", "catalog_nodes",
    "warp_nodes", "sampler_nodes", "guide_nodes", "prompt_nodes",
    "live_nodes", "effects_nodes", "look_nodes", "video_nodes",
    "glitch_nodes", "director_nodes", "storyboard_nodes", "loop_nodes",
    "setup_nodes",
)

# old class id -> the v1 node that replaces it (shown in the node description)
REPLACED_BY = {
    "DifforumAnimSetup": "Difforum · Setup",
    "DifforumAnimSetupPlus": "Difforum · Setup",
    "DifforumSchedule": "Difforum · Schedule",
    "DifforumSampleSchedule": "Difforum · Schedule (values output)",
    "DifforumScheduleInfo": "Difforum · Schedule Plot",
    "DifforumSchedulePlot": "Difforum · Schedule Plot",
    "DifforumAudioAnalyzer": "Difforum · Audio Analyzer",
    "DifforumAudioSchedule": "Difforum · Audio Curve",
    "DifforumCamera": "Difforum · Camera (expressions)",
    "DifforumCameraMove": "Difforum · Camera (one key)",
    "DifforumCameraShots": "Difforum · Camera (blend=0)",
    "DifforumCameraKeys": "Difforum · Camera",
    "DifforumSeamlessCamera": "Difforum · Camera (loop_mode)",
    "DifforumCameraPreview": "Difforum · Storyboard",
    "DifforumWarp": "Difforum · Storyboard",
    "DifforumStoryboard": "Difforum · Storyboard",
    "DifforumFilmDirector": "Difforum · Director",
    "DifforumPromptSchedule": "Difforum · Prompt Travel",
    "DifforumPromptScenes": "Difforum · Director / Prompt Travel",
    "DifforumPromptBatch": "Difforum · Prompt Travel (batched output)",
    "DifforumFeedbackSampler": "Difforum · Feedback Sampler",
    "DifforumLiveSampler": "Difforum · Live Sampler",
    "DifforumLiveStep": "Difforum · Live Sampler",
    "DifforumGuideBuilder": "Difforum · Guide Frames",
    "DifforumLoopTake": "Difforum · Loop",
    "DifforumLoopBlend": "Difforum · Loop",
    "DifforumPingPong": "Difforum · Loop",
    "DifforumSymmetry": "Difforum · Symmetry",
    "DifforumEchoTrails": "Difforum · Echo Trails",
    "DifforumDetailGuard": "Difforum · Detail Guard",
    "DifforumFlowStabilize": "Difforum · Flow Stabilize",
    "DifforumModelProfile": "docs/MODELS.md",
    "DifforumModelCatalog": "docs/MODELS.md",
    "DifforumLoadVideo": "core Load Video / VHS",
    "DifforumSaveVideo": "core Save Video (keeps audio)",
    "DifforumLook": "a colour-grading pack",
    "DifforumColorGrade": "a colour-grading pack",
    "DifforumGlow": "a colour-grading pack",
    "DifforumGlitch": "a glitch pack",
    "DifforumDatamosh": "a glitch pack",
}

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

for _name in _MODULES:
    try:
        _mod = importlib.import_module(f".{_name}", __name__)
    except Exception as exc:  # a broken legacy module must never block v1
        log.warning("[Difforum] legacy module %s not loaded: %s", _name, exc)
        continue
    for _cid, _cls in _mod.NODE_CLASS_MAPPINGS.items():
        _cls.DEPRECATED = True
        _cls.CATEGORY = "Difforum/legacy"
        _new = REPLACED_BY.get(_cid, "a v1 node")
        _cls.DESCRIPTION = (
            f"Difforum 0.x node, kept so old workflows open. Replaced by {_new}. "
            "Removed in Difforum 2.0."
        )
        NODE_CLASS_MAPPINGS[_cid] = _cls
        _disp = _mod.NODE_DISPLAY_NAME_MAPPINGS.get(_cid, _cid)
        NODE_DISPLAY_NAME_MAPPINGS[_cid] = f"{_disp} [legacy]"
