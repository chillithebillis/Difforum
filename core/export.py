"""
Camera interchange: Difforum <-> After Effects / Blender / JSON.

Difforum's camera is a per-frame *scene* transform (P_f = d_f @ P_{f-1}) in an
OpenCV-style frame: x right, y down, z forward, world = the first camera. Here
it becomes a real camera-to-world track with a focal length per frame:

* 3D mode: the deltas as-is (lateral units measured in pixels at the
  reference depth, exactly as the renderer uses them), plus the zoom channel
  as a focal-length change.
* 2D mode: a nodal camera. Pans become yaw/pitch (tan(angle) = shift / f),
  roll stays roll and zoom becomes focal length - the move you would do on a
  tripod to get the same 2D image motion.

Units: one scene unit = the depth-map unit of the renderer (near..far, default
1..100). `unit_scale` converts to Blender metres; After Effects uses pixels, so
the reference depth is placed on the z = 0 layer plane.

All conversions are plain numpy so they are testable and round-trip.
"""

from __future__ import annotations

import json
import math

import numpy as np

SCHEMA = "difforum.camera/1"

# OpenCV world (x right, y down, z fwd) -> Blender world (x right, y fwd, z up)
_CV_TO_BL_WORLD = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=np.float64)
# OpenCV camera axes -> Blender camera axes (camera looks down -Z, Y up)
_CV_TO_BL_CAM = np.diag([1.0, -1.0, -1.0, 1.0])


def focal_px(width: int, fov_deg: float) -> float:
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def fov_from_focal(width: int, f: float) -> float:
    return math.degrees(2.0 * math.atan(0.5 * width / max(f, 1e-6)))


def _rot(rx, ry, rz):
    from .camera import euler_to_matrix
    return euler_to_matrix(rx, ry, rz)


def reference_z(near: float = 1.0, far: float = 100.0, reference_depth: float = 0.5) -> float:
    return near + (1.0 - reference_depth) * (far - near)


def camera_track_to_world(camera, width: int, height: int, near: float = 1.0,
                          far: float = 100.0, reference_depth: float = 0.5):
    """-> (list of 4x4 camera-to-world, focal_px per frame). World = frame 0 camera."""
    n = len(camera.deltas)
    z_ref = reference_z(near, far, reference_depth)
    pose = np.eye(4)
    zoom_cum = 1.0
    c2w, focals = [], []
    for f in range(n):
        fov = float(camera.fov[f])
        f0 = focal_px(width, fov)
        if f > 0:
            d = np.array(camera.deltas[f], dtype=np.float64)
            if camera.mode == "3d":
                d = d.copy()
                lat = z_ref / f0
                d[0, 3] *= lat
                d[1, 3] *= lat
            else:
                tx, ty = float(d[0, 3]), float(d[1, 3])
                roll = math.degrees(math.atan2(d[1, 0], d[0, 0]))
                yaw = math.degrees(math.atan(tx / f0))
                pitch = -math.degrees(math.atan(ty / f0))
                d = np.eye(4)
                d[:3, :3] = _rot(pitch, yaw, roll)
            pose = d @ pose
            zoom_cum *= float(camera.zoom[f])
        c2w.append(np.linalg.inv(pose))
        focals.append(f0 * zoom_cum)
    return c2w, focals


def world_to_camera_track(c2w: list, focals: list, width: int, mode: str = "3d",
                          near: float = 1.0, far: float = 100.0, reference_depth: float = 0.5):
    """Inverse of `camera_track_to_world` (3D): camera-to-world matrices in the
    OpenCV frame -> a Difforum CameraTrack whose deltas re-create the move."""
    from .camera import CameraTrack

    z_ref = reference_z(near, far, reference_depth)
    base = np.array(c2w[0], dtype=np.float64)
    poses, deltas, fovs = [], [], []
    prev = np.eye(4)
    for f, (m, fl) in enumerate(zip(c2w, focals)):
        pose = np.linalg.inv(np.array(m, dtype=np.float64)) @ base    # world0 -> cam f
        d = pose @ np.linalg.inv(prev) if f > 0 else np.eye(4)
        fov = fov_from_focal(width, float(fl))
        lat = z_ref / focal_px(width, fov)
        d = d.copy()
        d[0, 3] /= lat
        d[1, 3] /= lat
        deltas.append(d)
        poses.append(pose)
        fovs.append(fov)
        prev = pose
    return CameraTrack(deltas=deltas, poses=poses, zoom=[1.0] * len(deltas), fov=fovs, mode=mode)


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def to_json(camera, width, height, fps, near=1.0, far=100.0, reference_depth=0.5) -> dict:
    c2w, focals = camera_track_to_world(camera, width, height, near, far, reference_depth)
    return {
        "schema": SCHEMA,
        "convention": "opencv",
        "note": "camera-to-world, x right / y down / z forward, world = first camera, "
                "units = Difforum scene units (depth-map units)",
        "fps": float(fps), "width": int(width), "height": int(height),
        "reference_z": reference_z(near, far, reference_depth),
        "frames": [
            {"f": i, "matrix": np.round(m, 8).tolist(), "focal_px": round(float(fl), 5),
             "fov_deg": round(fov_from_focal(width, fl), 5)}
            for i, (m, fl) in enumerate(zip(c2w, focals))
        ],
    }


def _ae_matrix(rx, ry, rz):
    return _rot(rx, ry, rz)


def from_json(data) -> tuple:
    """Parse a camera JSON (Difforum, or the Blender / After Effects export
    tools in tools/) -> (c2w list in OpenCV frame, focals, width, height, fps)."""
    if isinstance(data, str):
        data = json.loads(data)
    conv = data.get("convention", "opencv")
    w, h = int(data["width"]), int(data["height"])
    fps = float(data.get("fps", 24.0))
    c2w, focals = [], []
    for fr in data["frames"]:
        if conv == "ae":
            # position in px (y down, z forward), rotations in degrees, zoom = focal px
            fl = float(fr["zoom"])
            pos = np.array(fr["position"], dtype=np.float64)
            m = np.eye(4)
            m[:3, :3] = _ae_matrix(*fr.get("rotation", [0.0, 0.0, 0.0]))
            origin = np.array([w / 2.0, h / 2.0, -float(data.get("zoom0", fl))])
            scale = float(data.get("reference_z", 50.5)) / float(data.get("zoom0", fl))
            m[:3, 3] = (pos - origin) * scale
        else:
            m = np.array(fr["matrix"], dtype=np.float64)
            if conv == "blender":
                m = np.linalg.inv(_CV_TO_BL_WORLD) @ m @ np.linalg.inv(_CV_TO_BL_CAM)
                m[:3, 3] /= float(data.get("unit_scale", 0.1))
            fl = float(fr.get("focal_px") or focal_px(w, float(fr.get("fov_deg", 40.0))))
        c2w.append(m)
        focals.append(fl)
    if c2w and conv == "blender":            # re-base so the first camera is the origin
        base_inv = np.linalg.inv(c2w[0])
        c2w = [base_inv @ m for m in c2w]
    return c2w, focals, w, h, fps


# ---------------------------------------------------------------------------
# Blender
# ---------------------------------------------------------------------------

def to_blender_script(camera, width, height, fps, unit_scale=0.1, name="DifforumCam",
                      near=1.0, far=100.0, reference_depth=0.5) -> str:
    c2w, focals = camera_track_to_world(camera, width, height, near, far, reference_depth)
    mats = []
    for m in c2w:
        mb = _CV_TO_BL_WORLD @ m @ _CV_TO_BL_CAM
        mb[:3, 3] *= unit_scale
        mats.append(np.round(mb, 6).tolist())
    lens = [round(fl / width * 36.0, 5) for fl in focals]
    return f'''# Difforum camera for Blender - run from the Text Editor (Alt+P).
# Creates/updates the camera "{name}", matches fps, resolution and frame range.
import bpy
from mathutils import Matrix

MATS = {json.dumps(mats)}
LENS = {json.dumps(lens)}
FPS, W, H = {float(fps)!r}, {int(width)}, {int(height)}

scn = bpy.context.scene
scn.render.fps = int(round(FPS))
scn.render.fps_base = scn.render.fps / FPS
scn.render.resolution_x, scn.render.resolution_y = W, H
scn.frame_start, scn.frame_end = 1, len(MATS)

cam_data = bpy.data.cameras.get("{name}") or bpy.data.cameras.new("{name}")
cam_data.sensor_fit = "HORIZONTAL"
cam_data.sensor_width = 36.0
cam = bpy.data.objects.get("{name}") or bpy.data.objects.new("{name}", cam_data)
if cam.name not in scn.collection.all_objects:
    scn.collection.objects.link(cam)
cam.animation_data_clear()
cam_data.animation_data_clear()
for i, (m, lens) in enumerate(zip(MATS, LENS)):
    f = i + 1
    cam.matrix_world = Matrix(m)
    cam.keyframe_insert("location", frame=f)
    cam.keyframe_insert("rotation_euler", frame=f)
    cam_data.lens = lens
    cam_data.keyframe_insert("lens", frame=f)
scn.camera = cam
print("Difforum: {{}} camera keys on {name}".format(len(MATS)))
'''


# ---------------------------------------------------------------------------
# After Effects
# ---------------------------------------------------------------------------

def _decompose_xyz(r: np.ndarray) -> tuple[float, float, float]:
    """Inverse of euler_to_matrix (R = Rz @ Ry @ Rx) -> (rx, ry, rz) degrees."""
    sy = -r[2, 0]
    sy = max(-1.0, min(1.0, sy))
    ry = math.asin(sy)
    if abs(math.cos(ry)) > 1e-6:
        rx = math.atan2(r[2, 1], r[2, 2])
        rz = math.atan2(r[1, 0], r[0, 0])
    else:
        rx = math.atan2(-r[1, 2], r[1, 1])
        rz = 0.0
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


def to_ae_jsx(camera, width, height, fps, name="Difforum Camera",
              near=1.0, far=100.0, reference_depth=0.5) -> str:
    n = len(camera.deltas)
    duration = n / float(fps)
    if camera.mode == "2d":
        # exact 2D: a null whose transform reproduces the image motion
        from .warp import affine_2d
        acc = np.eye(3)
        pos, scl, rot = [], [], []
        for f in range(n):
            if f > 0:
                d = camera.deltas[f]
                ang = math.degrees(math.atan2(float(d[1][0]), float(d[0][0])))
                a = affine_2d(float(d[0][3]), float(d[1][3]), ang, float(camera.zoom[f]),
                              width, height).numpy()
                acc = a @ acc
            c = np.array([(width - 1) / 2.0, (height - 1) / 2.0, 1.0])
            p = acc @ c
            s = math.hypot(acc[0, 0], acc[1, 0])
            pos.append([round(float(p[0]), 4), round(float(p[1]), 4)])
            scl.append(round(s * 100.0, 5))
            rot.append(round(math.degrees(math.atan2(acc[1, 0], acc[0, 0])), 5))
        body = f'''var POS = {json.dumps(pos)};
var SCL = {json.dumps(scl)};
var ROT = {json.dumps(rot)};
var nul = comp.layers.addNull(comp.duration);
nul.name = "{name} (2D)";
nul.anchorPoint.setValue([(comp.width - 1) / 2, (comp.height - 1) / 2]);
for (var i = 0; i < POS.length; i++) {{
    var t = i / comp.frameRate;
    nul.position.setValueAtTime(t, POS[i]);
    nul.scale.setValueAtTime(t, [SCL[i], SCL[i]]);
    nul.rotation.setValueAtTime(t, ROT[i]);
}}
alert("Difforum: 2D move on a null. Parent your footage to it.");'''
    else:
        c2w, focals = camera_track_to_world(camera, width, height, near, far, reference_depth)
        z_ref = reference_z(near, far, reference_depth)
        f0 = focals[0]
        s = f0 / z_ref
        pos, rot, zoom = [], [], []
        for m, fl in zip(c2w, focals):
            p = m[:3, 3] * s + np.array([width / 2.0, height / 2.0, -f0])
            pos.append([round(float(v), 4) for v in p])
            rot.append([round(v, 5) for v in _decompose_xyz(m[:3, :3])])
            zoom.append(round(float(fl), 4))
        body = f'''var POS = {json.dumps(pos)};
var ROT = {json.dumps(rot)};
var ZOOM = {json.dumps(zoom)};
var cam = comp.layers.addCamera("{name}", [comp.width / 2, comp.height / 2]);
cam.autoOrient = AutoOrientType.NO_AUTO_ORIENT;
for (var i = 0; i < POS.length; i++) {{
    var t = i / comp.frameRate;
    cam.position.setValueAtTime(t, POS[i]);
    cam.xRotation.setValueAtTime(t, ROT[i][0]);
    cam.yRotation.setValueAtTime(t, ROT[i][1]);
    cam.zRotation.setValueAtTime(t, ROT[i][2]);
    cam.zoom.setValueAtTime(t, ZOOM[i]);
}}
alert("Difforum: " + POS.length + " camera keys. Layers at z=0 sit at the reference depth.");'''
    return f'''// Difforum camera for After Effects - File > Scripts > Run Script File...
// Uses the active comp, or creates one that matches the render.
(function () {{
app.beginUndoGroup("Difforum camera");
var comp = app.project.activeItem;
if (!(comp && comp instanceof CompItem)) {{
    comp = app.project.items.addComp("Difforum", {int(width)}, {int(height)}, 1, {duration:.6f}, {float(fps)!r});
    comp.openInViewer();
}}
{body}
app.endUndoGroup();
}})();
'''
