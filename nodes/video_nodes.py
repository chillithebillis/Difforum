"""
Difforum video IO nodes: Load Video (file -> IMAGE batch) and Save Video
(IMAGE batch -> MP4), so a footage workflow is self-contained with no external
video nodes. Reading uses OpenCV, writing uses imageio-ffmpeg.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from core.video import VIDEO_EXTS, read_video, write_video  # noqa: E402

CATEGORY = "Difforum/video"


def _input_videos():
    try:
        import folder_paths
        d = folder_paths.get_input_directory()
        return sorted(f for f in os.listdir(d) if f.lower().endswith(VIDEO_EXTS))
    except Exception:
        return []


class DifforumLoadVideo:
    """Load a video file (from the input folder) into an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        vids = _input_videos()
        video = (vids,) if vids else ("STRING", {"default": "footage.mp4"})
        return {
            "required": {
                "video": video,
                "max_frames": ("INT", {"default": 0, "min": 0, "max": 100000}),
                "frame_skip": ("INT", {"default": 1, "min": 1, "max": 30}),
                "resize_to": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "FLOAT")
    RETURN_NAMES = ("frames", "frame_count", "fps")
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, video, max_frames, frame_skip, resize_to):
        path = video
        if not os.path.isabs(path):
            try:
                import folder_paths
                path = os.path.join(folder_paths.get_input_directory(), video)
            except Exception:
                pass
        frames, fps = read_video(path, max_frames=int(max_frames),
                                 frame_skip=int(frame_skip), resize_to=int(resize_to))
        return (frames, int(frames.shape[0]), float(fps))


class DifforumSaveVideo:
    """Write an IMAGE batch to an MP4 in the output folder."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "Difforum_vj"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "quality": ("INT", {"default": 8, "min": 1, "max": 10}),
            },
            "optional": {
                "fps_in": ("FLOAT", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def run(self, frames, filename_prefix, fps, quality, fps_in=None):
        try:
            import folder_paths
            out_dir = folder_paths.get_output_directory()
        except Exception:
            out_dir = "."
        os.makedirs(out_dir, exist_ok=True)
        i = 0
        while True:
            name = f"{filename_prefix}_{i:05d}.mp4"
            path = os.path.join(out_dir, name)
            if not os.path.exists(path):
                break
            i += 1
        use_fps = float(fps_in) if fps_in else float(fps)
        write_video(path, frames, fps=use_fps, quality=int(quality))
        print(f"[Difforum] saved video -> {path}")
        return {"ui": {"text": [path]}, "result": (path,)}


NODE_CLASS_MAPPINGS = {
    "DifforumLoadVideo": DifforumLoadVideo,
    "DifforumSaveVideo": DifforumSaveVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DifforumLoadVideo": "Difforum · Load Video",
    "DifforumSaveVideo": "Difforum · Save Video (MP4)",
}
