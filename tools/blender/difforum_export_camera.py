"""Export the active Blender camera to Difforum.

Run from Blender's Text Editor (Alt+P) with the camera selected or set as the
scene camera. Writes `<blend name>_difforum_camera.json` next to the .blend
(or to your home folder for an unsaved file). Load it in ComfyUI with
Difforum · Camera Import (put the file in ComfyUI/input).

UNIT_SCALE must match Camera Export's `blender_unit_scale` if the move came
from Difforum originally (default 0.1 m per Difforum unit).
"""

import json
import os

import bpy

UNIT_SCALE = 0.1

scn = bpy.context.scene
cam = bpy.context.active_object if (bpy.context.active_object and
                                    bpy.context.active_object.type == "CAMERA") else scn.camera
if cam is None:
    raise RuntimeError("Select a camera (or set a scene camera) first.")

W = scn.render.resolution_x * scn.render.resolution_percentage // 100
H = scn.render.resolution_y * scn.render.resolution_percentage // 100
fps = scn.render.fps / scn.render.fps_base
frames = []
current = scn.frame_current
for f in range(scn.frame_start, scn.frame_end + 1):
    scn.frame_set(f)
    data = cam.data
    sensor = data.sensor_width if data.sensor_fit != "VERTICAL" else data.sensor_height * W / H
    focal_px = data.lens / sensor * W
    frames.append({
        "f": f - scn.frame_start,
        "matrix": [list(row) for row in cam.matrix_world],
        "focal_px": focal_px,
    })
scn.frame_set(current)

out = {"schema": "difforum.camera/1", "convention": "blender", "unit_scale": UNIT_SCALE,
       "fps": fps, "width": W, "height": H, "frames": frames}
base = bpy.path.abspath("//") or os.path.expanduser("~")
name = (bpy.path.basename(bpy.data.filepath).rsplit(".", 1)[0] or "untitled") + "_difforum_camera.json"
path = os.path.join(base, name)
with open(path, "w", encoding="utf-8") as fh:
    json.dump(out, fh)
print(f"Difforum: wrote {len(frames)} frames -> {path}")
