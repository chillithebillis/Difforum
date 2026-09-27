"""Shared plumbing for the Difforum v1 nodes."""

from __future__ import annotations

import logging

log = logging.getLogger("difforum")

# Socket types. Kept identical to 0.x where the payload is compatible, so a
# legacy node can still feed a v1 node during migration.
PARAMS = "DIFFORUM_PARAMS"
SCHEDULE = "DIFFORUM_SCHEDULE"
CAMERA = "DIFFORUM_CAMERA"
AUDIO = "DIFFORUM_AUDIO"
PROMPT = "DIFFORUM_PROMPT"

ROOT = "Difforum"
CAT_SETUP = f"{ROOT}/1 · Setup"
CAT_DIRECT = f"{ROOT}/2 · Direction"
CAT_CURVES = f"{ROOT}/3 · Curves & Prompts"
CAT_RENDER = f"{ROOT}/4 · Render"
CAT_BRIDGE = f"{ROOT}/5 · Video model bridges"
CAT_EXPORT = f"{ROOT}/6 · Export"
CAT_POST = f"{ROOT}/7 · Post"


def audio_vars(audio) -> dict | None:
    """Per-frame audio curves for expressions, or None."""
    return audio.get("curves") if isinstance(audio, dict) else None


def call_comfy_node(class_id: str, **kwargs):
    """Call another ComfyUI node by class id, v1 or v3 schema, and return its
    outputs as a tuple. Used for bridges onto core nodes (e.g. LTXVAddGuide) so
    Difforum never imports another pack's internals by path."""
    import nodes as comfy_nodes

    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(class_id)
    if cls is None:
        raise RuntimeError(
            f"ComfyUI node {class_id!r} is not available - update ComfyUI "
            "(it ships with core) or install the pack that provides it."
        )
    execute = getattr(cls, "execute", None)
    if callable(execute) and not hasattr(cls, "FUNCTION"):
        out = execute(**kwargs)             # v3 schema (io.ComfyNode)
    else:
        out = getattr(cls(), cls.FUNCTION)(**kwargs)
    res = getattr(out, "result", out)
    if isinstance(res, dict):
        res = res.get("result", ())
    return tuple(res)


def progress_bar(total: int):
    try:
        import comfy.utils
        return comfy.utils.ProgressBar(int(total))
    except Exception:  # outside ComfyUI (tests)
        class _Null:
            def update(self, *_a, **_k):
                pass

            def update_absolute(self, *_a, **_k):
                pass
        return _Null()


def check_interrupt():
    try:
        import comfy.model_management as mm
        mm.throw_exception_if_processing_interrupted()
    except ImportError:
        pass
