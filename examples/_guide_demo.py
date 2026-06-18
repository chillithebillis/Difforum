"""
Offline validation of the hybrid bridge: run the real Difforum nodes
(AnimSetup -> Camera -> GuideBuilder) on a real anchor image and save the guide
sequence + a GIF. No Wan model needed - this proves the guide batch the Wan
VACE graph would receive is coherent (camera-driven warp of the anchor).
"""

import glob
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from difforum.nodes.guide_nodes import DifforumGuideBuilder  # noqa: E402
from difforum.nodes.hybrid_nodes import DifforumCamera  # noqa: E402
from difforum.nodes.schedule_nodes import DifforumAnimSetup  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "_render_out"

# pick any rendered frame as the anchor
cands = sorted(glob.glob(str(OUT / "Difforum_pt_*.png"))) or sorted(glob.glob(str(OUT / "*.png")))
if not cands:
    raise SystemExit("no anchor image found in _render_out")
anchor_path = cands[-1]
img = Image.open(anchor_path).convert("RGB")
anchor = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0)
print(f"anchor: {anchor_path}  shape={tuple(anchor.shape)}")

params = DifforumAnimSetup().build(img.width, img.height, 16, 24, 0)[0]
cam = DifforumCamera().run(
    params, "2d", 40.0,
    translation_x="0:(0)", translation_y="0:(0)", translation_z="0:(0)",
    rotation_3d_x="0:(0)", rotation_3d_y="0:(0)", rotation_3d_z="0:(0.8)",
    zoom="0:(1.03)",
)[0]

guides, masks, info = DifforumGuideBuilder().run(anchor, cam, params, "force_2d")
print(info)
print(f"guides shape={tuple(guides.shape)} masks shape={tuple(masks.shape)}")

frames = [Image.fromarray((guides[i].clamp(0, 1).numpy() * 255).astype(np.uint8)) for i in range(guides.shape[0])]
frames = [f.resize((256, 256)) for f in frames]
gif = HERE / "difforum_guide_demo.gif"
frames[0].save(gif, save_all=True, append_images=frames[1:] + frames[::-1], duration=120, loop=0, optimize=True)
print(f"GIF: {gif} ({gif.stat().st_size // 1024} KB, {len(frames)} frames)")
