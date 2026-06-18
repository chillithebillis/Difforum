"""
Difforum model catalog node.

Surfaces curated, classic-and-trainable model stacks (recipes) with download
sources and LoRA training tooling, so picking a hybrid setup is a dropdown
choice instead of a research project. Outputs a DIFFORUM_RECIPE dict that a
future loader node can consume to auto-wire the right models.
"""

from __future__ import annotations

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core import models as M  # noqa: E402

CATEGORY = "Difforum/models"


class DifforumModelCatalog:
    """Pick a curated model recipe; get its download + training guide."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "recipe": (M.recipe_keys(), {"default": "wan22_12gb"}),
            }
        }

    RETURN_TYPES = ("DIFFORUM_RECIPE", "STRING")
    RETURN_NAMES = ("recipe", "guide")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, recipe):
        guide = M.summarize_recipe(recipe)
        bundle = M.get_recipe(recipe).as_dict()
        return {"ui": {"text": [guide]}, "result": (bundle, guide)}


NODE_CLASS_MAPPINGS = {
    "DifforumModelCatalog": DifforumModelCatalog,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumModelCatalog": "Difforum · Model Catalog",
}
