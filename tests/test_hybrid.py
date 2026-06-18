"""Standalone tests for the camera engine and model profile resolver."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.camera import build_camera, euler_to_matrix  # noqa: E402
from core.camera_presets import CAMERA_PRESETS, preset_schedules  # noqa: E402
from core.profiles import FAMILIES, resolve_profile, summarize  # noqa: E402
from core.schedule import build_schedule  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


def _raises(fn):
    try:
        fn()
        return False
    except Exception:  # noqa: BLE001
        return True


# --- camera engine ----------------------------------------------------------
print("camera engine:")

# identity rotation
assert np.allclose(euler_to_matrix(0, 0, 0), np.eye(3))
check("euler identity", True)

# 90deg about Z maps x-axis -> y-axis
r = euler_to_matrix(0, 0, 90)
check("rotate z 90", np.allclose(r @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9),
      str((r @ np.array([1, 0, 0])).round(3)))

# no motion -> all deltas identity, poses identity
cam = build_camera({}, max_frames=10, mode="3d")
check("track length", len(cam) == 10)
check("no-motion deltas identity", all(np.allclose(d, np.eye(4)) for d in cam.deltas))
check("no-motion poses identity", all(np.allclose(p, np.eye(4)) for p in cam.poses))
check("default zoom 1.0", all(abs(z - 1.0) < 1e-9 for z in cam.zoom))

# constant forward dolly accumulates linearly in z
cam = build_camera({"translation_z": [2.0] * 5}, max_frames=5, mode="3d")
zs = [p[2, 3] for p in cam.poses]
check("dolly accumulates", np.allclose(zs, [2, 4, 6, 8, 10]), str(zs))

# 2d mode ignores z translation
cam = build_camera({"translation_z": [5.0] * 3}, max_frames=3, mode="2d")
check("2d ignores z", all(abs(p[2, 3]) < 1e-9 for p in cam.poses))

# rotation accumulates: 10deg/frame about z over 9 frames -> 90deg total
cam = build_camera({"rotation_3d_z": [10.0] * 9}, max_frames=9, mode="3d")
total = cam.poses[-1][:3, :3]
check("rotation accumulates to 90", np.allclose(total @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-6),
      str((total @ np.array([1, 0, 0])).round(3)))

# fov override from schedule
cam = build_camera({"fov": [60.0] * 4}, max_frames=4)
check("fov from schedule", all(abs(v - 60.0) < 1e-9 for v in cam.fov))

# --- camera presets ---------------------------------------------------------
print("camera presets:")


def _cam_from_preset(preset, mode="2d", n=10):
    strs = preset_schedules(preset, speed=1.0, intensity=1.0)
    sch = {ax: build_schedule(e, n, 24).as_list() for ax, e in strs.items()}
    return build_camera(sch, n, mode)


still = _cam_from_preset("still")
check("still is identity", all(np.allclose(p, np.eye(4)) for p in still.poses))

# every preset expands and builds without error
ok_all = True
for p in CAMERA_PRESETS:
    try:
        _cam_from_preset(p)
    except Exception as e:  # noqa: BLE001
        ok_all = False
        print("    preset failed:", p, e)
check("all presets build", ok_all)

# non-still presets actually move (poses or zoom change)
zoom_cam = _cam_from_preset("zoom_in")
check("zoom_in changes zoom", zoom_cam.zoom[-1] > 1.0, str(zoom_cam.zoom[-1]))
pan_cam = _cam_from_preset("pan_right")
check("pan_right moves x", abs(pan_cam.poses[-1][0, 3]) > 1.0, str(pan_cam.poses[-1][0, 3]))
orbit = _cam_from_preset("orbit_left", mode="3d")
check("orbit rotates", not np.allclose(orbit.poses[-1][:3, :3], np.eye(3)))
check("unknown preset raises", _raises(lambda: preset_schedules("nope")))

# --- model profiles ---------------------------------------------------------
print("model profiles:")

# tier selection by VRAM floor
p12 = resolve_profile(12, "wan22", "fast")
p16 = resolve_profile(16, "wan22", "balanced")
p24 = resolve_profile(24, "wan22", "quality")
p32 = resolve_profile(32, "wan22", "quality")
check("12GB -> 5B", "5B" in p12.label, p12.label)
check("16GB -> 14B Q4", "Q4" in p12.label or "Q4" in p16.label, p16.label)
check("24GB -> 720p", p24.width == 1280 and p24.height == 720, f"{p24.width}x{p24.height}")
check("32GB -> fp8 no offload", p32.quant.startswith("fp8") and p32.offload == "none", p32.quant)

# resolution and segment grow with VRAM
check("res grows with vram", p12.width <= p24.width)
check("segment grows with vram", p12.segment_frames <= p24.segment_frames)

# fast uses lightning + few steps; quality drops it
check("fast lightning", p12.use_lightning_lora and p12.steps <= 6)
check("quality no lightning", not p24.use_lightning_lora and p24.steps >= 20)

# below-min vram clamps to lowest tier instead of crashing
p8 = resolve_profile(8, "wan22", "fast")
check("8GB clamps to lowest", "5B" in p8.label, p8.label)

# every family resolves and summarizes without error
for fam in FAMILIES:
    prof = resolve_profile(16, fam, "balanced")
    s = summarize(prof)
    check(f"family {fam} resolves", isinstance(s, str) and prof.label != "")

# sd15 never uses the wan lightning lora
psd = resolve_profile(12, "sd15_animatediff", "fast")
check("sd15 no wan-lightning", not psd.use_lightning_lora)

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL HYBRID TESTS PASSED")
