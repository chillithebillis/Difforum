"""
Camera engine for Difforum - the Deforum soul of the hybrid pipeline.

Turns per-frame schedules (translation / rotation / zoom) into camera
transforms used to warp guide frames for the video model. This is what gives
the classic Deforum feel: math-driven, audio-reactive camera motion in 2D/3D.

Representation per frame:
    delta  4x4 relative transform applied THIS frame (Deforum-style, motion
           values are per-frame increments)
    pose   4x4 accumulated camera-to-world transform up to this frame

Both are returned so a warp node can either step the previous frame by `delta`
(feedback warp) or project an anchor image along the full `pose` path (VACE
guides). Pure numpy - no torch, no ComfyUI, fully testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CAMERA_MODES = ("2d", "3d")


def euler_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """Euler angles in DEGREES -> 3x3 rotation matrix (intrinsic Rz@Ry@Rx)."""
    ax, ay, az = math.radians(rx), math.radians(ry), math.radians(rz)
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    rot_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    rot_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return rot_z @ rot_y @ rot_x


def _transform(tx, ty, tz, rx, ry, rz) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = euler_to_matrix(rx, ry, rz)
    m[:3, 3] = (tx, ty, tz)
    return m


@dataclass
class CameraTrack:
    """Per-frame camera transforms plus optics, ready for a warp node."""

    deltas: list[np.ndarray]   # 4x4 relative transform per frame
    poses: list[np.ndarray]    # 4x4 accumulated pose per frame
    zoom: list[float]          # per-frame 2D scale factor
    fov: list[float]           # per-frame field of view (degrees)
    mode: str = "3d"

    def __len__(self) -> int:
        return len(self.deltas)


def build_camera(
    schedules: dict[str, list[float]],
    max_frames: int,
    mode: str = "3d",
    fov: float = 40.0,
) -> CameraTrack:
    """
    Build a CameraTrack from dense per-frame schedule lists.

    Expected keys (any missing one defaults to no motion):
        translation_x, translation_y, translation_z
        rotation_3d_x, rotation_3d_y, rotation_3d_z
        zoom           (2D scale; 1.0 = no zoom)
        fov            (overrides the scalar `fov` arg if present)

    In 2d mode the z translation and 3d rotations are ignored; `angle` (via
    rotation_3d_z) and zoom drive the motion.
    """
    if mode not in CAMERA_MODES:
        raise ValueError(f"unknown camera mode {mode!r}, pick from {CAMERA_MODES}")

    def col(name: str, default: float) -> list[float]:
        s = schedules.get(name)
        if not s:
            return [default] * max_frames
        return [s[min(i, len(s) - 1)] for i in range(max_frames)]

    tx = col("translation_x", 0.0)
    ty = col("translation_y", 0.0)
    tz = col("translation_z", 0.0)
    rx = col("rotation_3d_x", 0.0)
    ry = col("rotation_3d_y", 0.0)
    rz = col("rotation_3d_z", 0.0)
    zoom = col("zoom", 1.0)
    fov_list = col("fov", fov)

    deltas: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    acc = np.eye(4, dtype=np.float64)
    for f in range(max_frames):
        if mode == "2d":
            d = _transform(tx[f], ty[f], 0.0, 0.0, 0.0, rz[f])
        else:
            d = _transform(tx[f], ty[f], tz[f], rx[f], ry[f], rz[f])
        acc = acc @ d
        deltas.append(d)
        poses.append(acc.copy())

    return CameraTrack(
        deltas=deltas, poses=poses, zoom=zoom, fov=fov_list, mode=mode
    )
