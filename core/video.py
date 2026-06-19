"""
Video IO for Difforum: read a video file into an IMAGE batch and write an IMAGE
batch back to MP4, with no external custom nodes. Reading uses OpenCV, writing
uses the ffmpeg that ships with imageio-ffmpeg (both common in ComfyUI installs).
Imports are lazy so the package still loads without them.
"""

from __future__ import annotations

import numpy as np
import torch

VIDEO_EXTS = (".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif")


def read_video(path, max_frames=0, frame_skip=1, resize_to=0):
    """Return (frames[N,H,W,3] float 0..1, effective_fps). frame_skip keeps every
    Nth frame; resize_to caps the longest side (0 = native)."""
    try:
        import cv2
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"OpenCV needed to read video ({e}). pip install opencv-python")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    skip = max(1, int(frame_skip))
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % skip == 0:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            if resize_to and resize_to > 0:
                h, w = fr.shape[:2]
                s = resize_to / float(max(h, w))
                if s < 1.0:
                    nw, nh = (int(round(w * s)) // 2) * 2, (int(round(h * s)) // 2) * 2
                    fr = cv2.resize(fr, (max(2, nw), max(2, nh)), interpolation=cv2.INTER_AREA)
            frames.append(fr)
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames read from {path!r}")
    arr = np.stack(frames).astype(np.float32) / 255.0
    return torch.from_numpy(arr), float(fps) / skip


def write_video(path, frames, fps=24.0, quality=8):
    """Write an IMAGE batch [N,H,W,3] 0..1 to an MP4 (h264, yuv420p). Returns path."""
    try:
        import imageio_ffmpeg
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"imageio-ffmpeg needed to write video ({e}). pip install imageio-ffmpeg")
    if frames.dim() == 3:
        frames = frames.unsqueeze(0)
    arr = (frames[..., :3].clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)
    n, h, w, _ = arr.shape
    h2, w2 = h - (h % 2), w - (w % 2)            # yuv420p needs even dims
    if (h2, w2) != (h, w):
        arr = arr[:, :h2, :w2, :]
    writer = imageio_ffmpeg.write_frames(
        str(path), (w2, h2), fps=float(fps), codec="libx264",
        quality=int(quality), macro_block_size=1, pix_fmt_out="yuv420p",
    )
    writer.send(None)
    for frame in arr:
        writer.send(frame.tobytes())
    writer.close()
    return str(path)
