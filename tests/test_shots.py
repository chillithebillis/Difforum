"""Tests for the camera shot list (director) + camera path preview render."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.camera import build_camera  # noqa: E402
from core.plot import render_camera_path  # noqa: E402
from core.shots import parse_shots, shots_to_axis_values  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


print("parse:")
shots = parse_shots("0: dolly_in 1.0 1.0\n48: orbit_right 1.2 0.8\n96: spiral")
check("three shots", len(shots) == 3)
check("sorted + fields", shots[0] == (0, "dolly_in", 1.0, 1.0) and shots[1][1] == "orbit_right")
check("defaults speed/intensity", parse_shots("0: spiral")[0][2:] == (1.0, 1.0))
check("comments + blank lines ok", len(parse_shots("# intro\n0: still\n\n24: sway  # b\n")) == 2)
auto = parse_shots("24: spiral")
check("auto still at 0 when list starts later", auto[0] == (0, "still", 1.0, 1.0))
for bad in ("dolly_in", "0: notapreset", "x: spiral"):
    try:
        parse_shots(bad)
        ok = False
    except ValueError:
        ok = True
    check(f"rejects {bad!r}", ok)

print("axis values:")
vals, summary = shots_to_axis_values("0: pan_right 1.0 1.0\n10: pan_left 1.0 1.0", max_frames=20)
check("axes complete", set(vals) == {"translation_x", "translation_y", "translation_z",
                                     "rotation_3d_x", "rotation_3d_y", "rotation_3d_z", "zoom"})
check("exact frame count", all(len(v) == 20 for v in vals.values()))
tx = vals["translation_x"]
check("segment 1 pans right (+)", all(v > 0 for v in tx[:10]), f"{tx[:3]}")
check("segment 2 pans left (-)", all(v < 0 for v in tx[10:]), f"{tx[10:13]}")
check("summary lists both", "pan_right" in summary and "pan_left" in summary)
vals2, _ = shots_to_axis_values("0: zoom_in", max_frames=8)
check("zoom axis > 1", all(v > 1.0 for v in vals2["zoom"]))

print("camera + preview:")
cam = build_camera(vals, max_frames=20, mode="2d", fov=40.0)
check("camera has 20 frames", len(cam) == 20)
img = render_camera_path(cam.poses, cam.zoom, mode=cam.mode, width=256, height=256)
check("preview shape", img.shape == (256, 256, 3))
check("preview in range", float(img.min()) >= 0.0 and float(img.max()) <= 1.0)
check("preview draws something", float(img.std()) > 0.01)
# still camera: degenerate path should not crash
still = build_camera(shots_to_axis_values("0: still", max_frames=5)[0], max_frames=5, mode="2d")
img2 = render_camera_path(still.poses, still.zoom, mode="2d", width=128, height=128)
check("degenerate path safe", img2.shape == (128, 128, 3) and np.isfinite(img2).all())

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL SHOTS TESTS PASSED")
