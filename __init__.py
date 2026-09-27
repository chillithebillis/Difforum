"""
Difforum - camera direction and timeline orchestration for ComfyUI.

Direct a shot on a visual timeline (scenes, camera moves, energy), render it
with the Deforum-style feedback sampler on any image model, or hand the same
direction to LTX-2 / MiniMax H3 and export the camera to After Effects and
Blender. See README.md.
"""

import logging

from .nodes import NODE_CLASS_MAPPINGS as _V1, NODE_DISPLAY_NAME_MAPPINGS as _V1_NAMES

log = logging.getLogger("difforum")

NODE_CLASS_MAPPINGS = dict(_V1)
NODE_DISPLAY_NAME_MAPPINGS = dict(_V1_NAMES)

try:  # 0.x nodes, hidden and deprecated, so old workflows still open
    from .legacy import NODE_CLASS_MAPPINGS as _OLD, NODE_DISPLAY_NAME_MAPPINGS as _OLD_NAMES
    for _k, _v in _OLD.items():
        NODE_CLASS_MAPPINGS.setdefault(_k, _v)
        NODE_DISPLAY_NAME_MAPPINGS.setdefault(_k, _OLD_NAMES[_k])
except Exception as exc:  # never let legacy code block the v1 pack
    log.warning("[Difforum] legacy nodes unavailable: %s", exc)

try:
    from .nodes.routes import register_routes
    register_routes()
except Exception as exc:
    log.warning("[Difforum] timeline preview endpoints unavailable: %s", exc)

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

print(f"[Difforum] {len(_V1)} nodes ready (+{len(NODE_CLASS_MAPPINGS) - len(_V1)} legacy)")
