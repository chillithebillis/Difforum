"""Camera export / import: After Effects, Blender and JSON."""

from __future__ import annotations

import json
import os

from ..core import export as X
from ._common import CAMERA, CAT_EXPORT, PARAMS
from .direction import DIRECTION


def _output_dir() -> str:
    try:
        import folder_paths
        base = folder_paths.get_output_directory()
    except Exception:
        base = "."
    d = os.path.join(base, "difforum")
    os.makedirs(d, exist_ok=True)
    return d


def _unique(path: str) -> str:
    root, ext = os.path.splitext(path)
    i, out = 1, path
    while os.path.exists(out):
        out = f"{root}_{i:03d}{ext}"
        i += 1
    return out


class DifforumCameraExport:
    """Send the Difforum camera to After Effects, Blender or any tool (JSON).

    Writes to `output/difforum/`:
      * `.jsx` - run in After Effects: a 3D camera (3d mode) or a transform
        null that reproduces the 2D move exactly (2d mode).
      * `.py` - run in Blender's Text Editor: an animated camera, fps,
        resolution and frame range matched.
      * `.json` - camera-to-world matrices + focal length per frame.
    Composite titles, 3D elements or a relight pass on top of the AI render
    with the exact same camera.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "formats": (["all", "after effects", "blender", "json"], {"default": "all"}),
                "filename_prefix": ("STRING", {"default": "difforum_camera"}),
                "blender_unit_scale": ("FLOAT", {"default": 0.1, "min": 0.0001, "max": 100.0, "step": 0.01,
                                       "tooltip": "Metres per Difforum scene unit."}),
            },
            "optional": {
                "direction": (DIRECTION,),
                "params": (PARAMS,),
                "camera": (CAMERA,),
                "translation_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.05,
                                      "tooltip": "Match the sampler's value so the exported move matches the render."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("paths", "json")
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CAT_EXPORT

    def run(self, formats, filename_prefix, blender_unit_scale, direction=None, params=None,
            camera=None, translation_scale=1.0):
        if direction is not None:
            params = params or direction.params
            camera = camera or direction.camera
        if params is None or camera is None:
            raise ValueError("Connect a Director `direction`, or params + camera.")
        if abs(translation_scale - 1.0) > 1e-9:
            camera = _scaled(camera, translation_scale)
        w, h, fps = int(params["width"]), int(params["height"]), float(params["fps"])
        prefix = os.path.basename(str(filename_prefix)) or "difforum_camera"
        out = _output_dir()
        paths = []
        data = X.to_json(camera, w, h, fps)
        text = json.dumps(data)
        if formats in ("all", "json"):
            p = _unique(os.path.join(out, f"{prefix}.json"))
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(text)
            paths.append(p)
        if formats in ("all", "after effects"):
            p = _unique(os.path.join(out, f"{prefix}.jsx"))
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(X.to_ae_jsx(camera, w, h, fps))
            paths.append(p)
        if formats in ("all", "blender"):
            p = _unique(os.path.join(out, f"{prefix}_blender.py"))
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(X.to_blender_script(camera, w, h, fps, unit_scale=blender_unit_scale))
            paths.append(p)
        listing = "\n".join(paths)
        return {"ui": {"text": [listing]}, "result": (listing, text)}


def _scaled(camera, k):
    import numpy as np

    from ..core.camera import CameraTrack
    deltas = []
    for d in camera.deltas:
        d = np.array(d, dtype=np.float64).copy()
        d[:3, 3] *= k
        deltas.append(d)
    return CameraTrack(deltas=deltas, poses=camera.poses, zoom=camera.zoom, fov=camera.fov,
                       mode=camera.mode)


class DifforumCameraImport:
    """Bring a camera in from Blender, After Effects or JSON.

    Accepts the files written by Camera Export and by the helper scripts in
    `tools/` (export the active Blender camera, or the selected After Effects
    camera). The result drives the Feedback Sampler, Storyboard or Guide
    Frames like any Difforum camera, so a move blocked in 3D becomes the AI
    shot.
    """

    DESCRIPTION = __doc__

    @classmethod
    def INPUT_TYPES(cls):
        files = []
        try:
            import folder_paths
            d = folder_paths.get_input_directory()
            files = sorted(f for f in os.listdir(d) if f.lower().endswith(".json"))
        except Exception:
            pass
        return {
            "required": {
                "file": (files or ["(put a camera .json in ComfyUI/input)"],),
                "params": (PARAMS,),
                "retime": (["match frames", "keep source length"], {"default": "match frames"}),
            },
            "optional": {"json_text": ("STRING", {"multiline": True, "default": "",
                         "tooltip": "Paste JSON here instead of choosing a file."})},
        }

    RETURN_TYPES = (CAMERA, "STRING")
    RETURN_NAMES = ("camera", "info")
    FUNCTION = "run"
    CATEGORY = CAT_EXPORT

    def run(self, file, params, retime, json_text=""):
        if json_text.strip():
            data = json_text
        else:
            import folder_paths

            from ..core.video import safe_input_path
            path = safe_input_path(folder_paths.get_input_directory(), file)
            with open(path, encoding="utf-8") as fh:
                data = fh.read()
        c2w, focals, w, h, fps = X.from_json(data)
        n_target = int(params["max_frames"])
        if retime == "match frames" and len(c2w) != n_target and len(c2w) > 1:
            idx = [round(i * (len(c2w) - 1) / max(1, n_target - 1)) for i in range(n_target)]
            c2w = [c2w[i] for i in idx]
            focals = [focals[i] for i in idx]
        # focal lengths are relative to the source width; rescale to the render width
        k = int(params["width"]) / float(w)
        cam = X.world_to_camera_track(c2w, [fl * k for fl in focals], int(params["width"]))
        return (cam, f"imported {len(c2w)} frames from a {w}x{h} @ {fps:g} fps camera")


NODE_CLASS_MAPPINGS = {
    "Difforum_CameraExport": DifforumCameraExport,
    "Difforum_CameraImport": DifforumCameraImport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Difforum_CameraExport": "Difforum · Camera Export (AE / Blender / JSON)",
    "Difforum_CameraImport": "Difforum · Camera Import",
}
