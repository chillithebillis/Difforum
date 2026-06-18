"""
End-to-end orchestration test: wires every Difforum node together and runs the
full pipeline, exactly as the workflow does, but with stub MODEL/VAE so it runs
without a GPU or checkpoint.

What it proves:
  AnimSetup -> AudioAnalyzer -> Camera(+audio) -> Schedule(+audio)
            -> FeedbackSampler (warp -> [stub diffuse] -> colour-match loop)
            -> N frames of the right shape
  and the Hybrid path: Camera -> GuideBuilder -> Wan-ready guide batch.

The only thing swapped for the real run is the stub diffusion -> a real
checkpoint inside ComfyUI. Everything else here is the production code path.
"""

import sys
import types
from pathlib import Path

import torch

# import Difforum as a package (difforum.nodes.*), like the real ComfyUI runtime,
# so the stub top-level `nodes` module below does not collide with difforum's own
# internal `nodes` subpackage.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# --- inject stub comfy + nodes BEFORE the sampler's lazy imports run --------
_comfy = types.ModuleType("comfy")
_comfy_utils = types.ModuleType("comfy.utils")


class _PBar:
    def __init__(self, total):
        self.total = total

    def update(self, n):
        pass


_comfy_utils.ProgressBar = _PBar
_comfy.utils = _comfy_utils
sys.modules["comfy"] = _comfy
sys.modules["comfy.utils"] = _comfy_utils

_nodes = types.ModuleType("nodes")


def _common_ksampler(model, seed, steps, cfg, sampler_name, scheduler,
                     positive, negative, latent, denoise=1.0):
    # identity "diffusion": hand the latent back unchanged
    return (latent,)


_nodes.common_ksampler = _common_ksampler


class _CNApply:
    def apply_controlnet(self, positive, negative, control_net, image, strength,
                         start, end, vae=None, extra_concat=[]):
        return (positive, negative)  # identity stub


_nodes.ControlNetApplyAdvanced = _CNApply
sys.modules["nodes"] = _nodes


class FakeVAE:
    """Pretends the latent is the image (encode/decode are identity)."""

    def encode(self, pixels):
        return pixels.clone()

    def decode(self, samples):
        return samples.clone()


# --- import the real Difforum nodes ------------------------------------------
from difforum.nodes.audio_nodes import DifforumAudioAnalyzer  # noqa: E402
from difforum.nodes.guide_nodes import DifforumGuideBuilder  # noqa: E402
from difforum.nodes.hybrid_nodes import DifforumCamera  # noqa: E402
from difforum.nodes.sampler_nodes import DifforumFeedbackSampler  # noqa: E402
from difforum.nodes.live_nodes import DifforumLiveStep  # noqa: E402
from difforum.nodes.schedule_nodes import DifforumAnimSetup, DifforumSchedule  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


W = H = 64
FRAMES = 8
FPS = 24

print("orchestrated pipeline:")

# 1) global params
params = DifforumAnimSetup().build(W, H, FPS, FRAMES, seed=1)[0]
check("params built", params["max_frames"] == FRAMES and params["width"] == W)

# 2) synthetic audio -> reactive curves
sr = 44100
t = torch.linspace(0, FRAMES / FPS, int(sr * FRAMES / FPS))
wave = (torch.sin(2 * torch.pi * 110 * t) * 0.6).reshape(1, 1, -1).expand(1, 2, -1)
audio = {"waveform": wave, "sample_rate": sr}
audio_bundle = DifforumAudioAnalyzer().run(params, audio, 0.25, True, 1.5)[0]
check("audio curves built", "curves" in audio_bundle and len(audio_bundle["curves"]["amp"]) == FRAMES)

# 3) camera driven by schedules (+ audio reactive zoom)
cam = DifforumCamera().run(
    params, "2d", 40.0, audio=audio_bundle,
    translation_x="0:(2)", translation_y="0:(0)", translation_z="0:(0)",
    rotation_3d_x="0:(0)", rotation_3d_y="0:(0)", rotation_3d_z="0:(1.0)",
    zoom="0:(1.0 + 0.1*amp)",
)[0]
check("camera track length", len(cam) == FRAMES)

# 4) strength (denoise) schedule
strength = DifforumSchedule().run(params, "0:(0.5), 7:(0.5)", "linear")[0]
check("strength schedule length", len(strength) == FRAMES)

# 5) feedback sampler (the full Classic+ loop, stub diffusion)
init = torch.rand(1, H, W, 3)
frames = DifforumFeedbackSampler().run(
    model=object(), positive=[], negative=[], vae=FakeVAE(),
    params=params, camera=cam, init_image=init, strength_schedule=strength,
    steps=4, cfg=7.0, sampler_name="euler", scheduler="normal",
    color_coherence=0.8,
)[0]
check("feedback output shape", tuple(frames.shape) == (FRAMES, H, W, 3), str(tuple(frames.shape)))
check("feedback values in range", float(frames.min()) >= 0.0 and float(frames.max()) <= 1.0)
# camera motion means frame 1 should differ from frame 0
check("frames evolve (motion applied)", not torch.allclose(frames[0], frames[1], atol=1e-4))
check("no NaNs", not torch.isnan(frames).any())

# 5b) prompt travel: per-frame conditioning list drives the sampler
from difforum.core.prompt import blend_conditioning, parse_prompt_schedule, plan_blend  # noqa: E402


def fake_cond(v):
    return [[torch.full((1, 77, 768), float(v)), {"pooled_output": torch.full((1, 768), float(v))}]]


kfs = parse_prompt_schedule("0: forest\n6: ocean\n11: galaxy")
plan = plan_blend([f for f, _ in kfs], FRAMES, "linear")
enc = [fake_cond(0.0), fake_cond(0.5), fake_cond(1.0)]
pos_sched = []
for (li, ri, w) in plan:
    if li == ri or w <= 0.0:
        pos_sched.append(enc[li])
    elif w >= 1.0:
        pos_sched.append(enc[ri])
    else:
        pos_sched.append(blend_conditioning(enc[ri], enc[li], w))
check("prompt schedule length", len(pos_sched) == FRAMES)
frames_pt = DifforumFeedbackSampler().run(
    model=object(), positive=fake_cond(0.0), negative=[], vae=FakeVAE(),
    params=params, camera=cam, init_image=init, strength_schedule=strength,
    steps=4, cfg=7.0, sampler_name="euler", scheduler="normal",
    color_coherence=0.5, positive_schedule=pos_sched,
)[0]
check("prompt-travel feedback shape", tuple(frames_pt.shape) == (FRAMES, H, W, 3), str(tuple(frames_pt.shape)))

# 5c) ControlNet path runs (stub control_net)
frames_cn = DifforumFeedbackSampler().run(
    model=object(), positive=fake_cond(0.0), negative=fake_cond(0.0), vae=FakeVAE(),
    params=params, camera=cam, init_image=init, strength_schedule=strength,
    steps=4, cfg=7.0, sampler_name="euler", scheduler="normal",
    color_coherence=0.5, control_net=object(), control_strength=0.6,
)[0]
check("controlnet feedback shape", tuple(frames_cn.shape) == (FRAMES, H, W, 3), str(tuple(frames_cn.shape)))

# 5d) Live Step: realtime single-frame primitive
live = DifforumLiveStep()
# frame 0 = seed passthrough
f0, nxt0 = live.run(
    model=object(), positive=fake_cond(0.0), negative=fake_cond(0.0), vae=FakeVAE(),
    params=params, camera=cam, prev_frame=init, frame_index=0, strength=0.5,
    steps=2, cfg=1.2, sampler_name="lcm", scheduler="sgm_uniform", color_coherence=0.5,
)
check("live frame0 passthrough", torch.allclose(f0[:1], init) and nxt0 == 1)
# frame 3 = warp + diffuse, returns one frame, advances index
f3, nxt3 = live.run(
    model=object(), positive=fake_cond(0.0), negative=fake_cond(0.0), vae=FakeVAE(),
    params=params, camera=cam, prev_frame=f0, frame_index=3, strength=0.5,
    steps=2, cfg=1.2, sampler_name="lcm", scheduler="sgm_uniform", color_coherence=0.5,
    control_net=object(), control_strength=0.6,
)
check("live step shape", tuple(f3.shape) == (1, H, W, 3), str(tuple(f3.shape)))
check("live index advances", nxt3 == 4)
# camera loops past max_frames without error
fL, nxtL = live.run(
    model=object(), positive=fake_cond(0.0), negative=fake_cond(0.0), vae=FakeVAE(),
    params=params, camera=cam, prev_frame=f0, frame_index=FRAMES + 2, strength=0.5,
    steps=2, cfg=1.2, sampler_name="lcm", scheduler="sgm_uniform", color_coherence=0.5,
    loop_camera=True,
)
check("live camera loops", tuple(fL.shape) == (1, H, W, 3))

# 6) hybrid path: guide builder produces a Wan-ready batch
anchor = torch.rand(1, H, W, 3)
guides, masks, info = DifforumGuideBuilder().run(anchor, cam, params, "force_2d")
check("guide batch shape", tuple(guides.shape) == (FRAMES, H, W, 3), str(tuple(guides.shape)))
check("mask batch shape", tuple(masks.shape) == (FRAMES, H, W), str(tuple(masks.shape)))
check("guide info text", isinstance(info, str) and "Wan VACE" in info)

# 7) 3D feedback path with a depth map also runs
cam3d = DifforumCamera().run(
    params, "3d", 40.0,
    translation_x="0:(0)", translation_y="0:(0)", translation_z="0:(0.5)",
    rotation_3d_x="0:(0)", rotation_3d_y="0:(0)", rotation_3d_z="0:(0)",
    zoom="0:(1.0)",
)[0]
depth = torch.rand(1, H, W, 1)
frames3d = DifforumFeedbackSampler().run(
    model=object(), positive=[], negative=[], vae=FakeVAE(),
    params=params, camera=cam3d, init_image=init, strength_schedule=strength,
    steps=4, cfg=7.0, sampler_name="euler", scheduler="normal",
    color_coherence=0.5, depth=depth, near=1.0, far=50.0,
)[0]
check("3d feedback shape", tuple(frames3d.shape) == (FRAMES, H, W, 3), str(tuple(frames3d.shape)))

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL INTEGRATION TESTS PASSED")
