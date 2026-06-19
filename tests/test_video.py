"""Round-trip test for video IO (write -> read). Skips cleanly if the optional
video deps (opencv-python / imageio-ffmpeg) are not installed."""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import cv2  # noqa: F401
    import imageio_ffmpeg  # noqa: F401
except Exception as e:
    print(f"SKIP video tests (optional deps missing: {e})")
    print("ALL VIDEO TESTS PASSED")
    sys.exit(0)

from core.video import read_video, write_video  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


torch.manual_seed(0)
N, H, W = 16, 180, 320
frames = torch.zeros(N, H, W, 3)
for i in range(N):
    x = torch.linspace(0, 1, W).view(1, W, 1)
    frames[i] = (0.5 + 0.5 * torch.sin(6 * x + i * 0.3)) * torch.tensor([1.0, 0.4, 0.8])
frames = frames.clamp(0, 1)

tmp = Path(tempfile.gettempdir()) / "difforum_video_test.mp4"
print("write/read round-trip:")
out = write_video(tmp, frames, fps=24)
check("file written", Path(out).exists() and Path(out).stat().st_size > 0)

back, fps = read_video(tmp)
check("read shape matches", back.shape[1:] == (H, W, 3), f"{tuple(back.shape)}")
check("frame count preserved", abs(back.shape[0] - N) <= 1, f"{back.shape[0]} vs {N}")
check("fps recovered", abs(fps - 24.0) < 2.0, f"{fps}")
check("values in range", float(back.min()) >= 0.0 and float(back.max()) <= 1.0)
check("not blank", float(back.mean()) > 0.05)

print("options:")
half = read_video(tmp, frame_skip=2)[0]
check("frame_skip reduces frames", half.shape[0] <= (N // 2) + 1)
small = read_video(tmp, resize_to=160)[0]
check("resize_to caps longest side", max(small.shape[1], small.shape[2]) <= 160)
capped = read_video(tmp, max_frames=5)[0]
check("max_frames caps count", capped.shape[0] == 5)

try:
    tmp.unlink()
except Exception:
    pass

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL VIDEO TESTS PASSED")
