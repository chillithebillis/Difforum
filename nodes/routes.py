"""HTTP endpoints for the Director timeline widget.

GET  /difforum/catalog  -> moves, moods, easings, reactions (one source of truth)
POST /difforum/preview  -> the camera path the renderer will actually use,
                           as per-frame 2x3 affines for the animated preview
"""

from __future__ import annotations

import logging

log = logging.getLogger("difforum")


def compute_preview(body: dict) -> dict:
    import torch

    from ..core.direction import build_direction
    from ..core.engine import EngineConfig, FeedbackEngine

    frames = max(1, min(int(body.get("frames", 120)), 20000))
    fps = float(body.get("fps", 24.0))
    mode = body.get("mode", "2d") if body.get("mode") in ("2d", "3d") else "2d"
    w = max(64, min(int(body.get("width", 768)), 8192))
    h = max(64, min(int(body.get("height", 432)), 8192))
    d = build_direction(body.get("timeline", ""), frames, fps, mode=mode,
                        camera_scale=float(body.get("camera_scale", 1.0)),
                        strength_bias=float(body.get("energy_bias", 0.0)),
                        blend=float(body.get("transition", 1.0)),
                        variation=float(body.get("variation", 0.0)),
                        variation_seed=int(body.get("variation_seed", 0)))
    from .direction import _track
    cam = _track(d.axes, d.lens, frames, mode, [b["move"] for b in d.camera_blocks])
    eng = FeedbackEngine(cam, EngineConfig(width=w, height=h))
    acc = torch.eye(3, dtype=torch.float64)
    step = max(1, frames // 480)
    affines = []
    for f in range(frames):
        if f > 0:
            acc = eng._affine_for(f) @ acc
        if f % step == 0 or f == frames - 1:
            a = acc[:2].flatten().tolist()
            affines.append([f] + [round(x, 5) for x in a])
    return {
        "frames": frames, "fps": fps, "width": w, "height": h,
        "affines": affines,
        "strength": [round(v, 4) for v in d.strength[::step]],
        "strength_step": step,
        "warnings": d.warnings,
        "camera_text": d.camera_text,
    }


def convert_script(body: dict) -> dict:
    """{timeline} -> {text};  {text} -> {timeline, notes}."""
    from ..core.script import script_to_timeline, timeline_to_script
    fps = float(body.get("fps", 24.0) or 24.0)
    frames = int(body.get("frames", 0) or 0) or None
    if "text" in body:
        tl, notes = script_to_timeline(str(body["text"]), fps, frames)
        return {"timeline": tl, "notes": notes}
    return {"text": timeline_to_script(body.get("timeline") or {}, frames)}


def register_routes():
    try:
        from aiohttp import web
        from server import PromptServer
    except Exception:  # outside ComfyUI
        return
    routes = PromptServer.instance.routes

    @routes.get("/difforum/catalog")
    async def _catalog(_request):
        from ..core.direction import ui_catalog
        return web.json_response(ui_catalog())

    @routes.post("/difforum/script")
    async def _script(request):
        """Timeline <-> text for the Director's Script panel."""
        try:
            body = await request.json()
            return web.json_response(convert_script(body))
        except Exception as exc:
            return web.json_response({"error": str(exc)}, status=400)

    @routes.post("/difforum/preview")
    async def _preview(request):
        try:
            body = await request.json()
            return web.json_response(compute_preview(body))
        except Exception as exc:  # the widget shows the message
            log.warning("[Difforum] preview failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=400)
