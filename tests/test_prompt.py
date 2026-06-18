"""Tests for prompt-travel parsing, blend planning, and conditioning blend."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.prompt import (  # noqa: E402
    blend_conditioning, parse_prompt_schedule, plan_blend, scenes_to_keyframes,
)

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


print("prompt parsing:")
kfs = parse_prompt_schedule("0: a forest\n30: an ocean\n59: a galaxy")
check("parse 3 keyframes", [f for f, _ in kfs] == [0, 30, 59])
check("parse text", kfs[1][1] == "an ocean", kfs[1][1])

# tolerate (), quotes, trailing commas, blank lines, out-of-order
kfs = parse_prompt_schedule('\n30:("storm"),\n0: (calm)\n')
check("tolerant parse order", [f for f, _ in kfs] == [0, 30])
check("strip parens", kfs[0][1] == "calm", kfs[0][1])
check("strip quotes+comma", kfs[1][1] == "storm", kfs[1][1])

try:
    parse_prompt_schedule("no frames here")
    check("reject empty schedule", False)
except ValueError:
    check("reject empty schedule", True)

print("scenes:")
kf = scenes_to_keyframes(["a", "b", "c"], 11)
check("3 scenes spaced", [f for f, _ in kf] == [0, 5, 10], str(kf))
check("scene texts", [t for _, t in kf] == ["a", "b", "c"])
kf = scenes_to_keyframes(["only", "", "  "], 10)
check("empty scenes skipped", kf == [(0, "only")], str(kf))
kf = scenes_to_keyframes(["x", "", "y", ""], 21)
check("gaps collapse + spread", [f for f, _ in kf] == [0, 20] and [t for _, t in kf] == ["x", "y"], str(kf))
try:
    scenes_to_keyframes(["", "  ", ""], 10)
    check("all-empty raises", False)
except ValueError:
    check("all-empty raises", True)

print("blend plan:")
plan = plan_blend([0, 10], max_frames=11, easing="linear")
check("plan length", len(plan) == 11)
check("frame 0 fully left", plan[0] == (0, 1, 0.0), str(plan[0]))
check("frame 5 halfway", abs(plan[5][2] - 0.5) < 1e-6, str(plan[5]))
# at the last keyframe it resolves to that keyframe fully (left==right, w=0)
check("frame 10 = last kf fully", plan[10][0] == 1 and plan[10][2] == 0.0, str(plan[10]))
# just before the end it is mostly toward the right keyframe
check("frame 9 mostly right", plan[9][1] == 1 and plan[9][2] > 0.8, str(plan[9]))

# single keyframe -> always weight 0 (no travel)
plan1 = plan_blend([0], max_frames=4)
check("single kf no travel", all(w == 0.0 for _, _, w in plan1))

print("conditioning blend:")
# fake conditioning: [[tensor[1,77,768], {"pooled_output": tensor[1,768]}]]
condA = [[torch.zeros(1, 77, 768), {"pooled_output": torch.zeros(1, 768)}]]
condB = [[torch.ones(1, 77, 768), {"pooled_output": torch.ones(1, 768)}]]

mid = blend_conditioning(condB, condA, 0.5)  # to=B(ones), from=A(zeros)
check("blend midpoint = 0.5", abs(float(mid[0][0].mean()) - 0.5) < 1e-6, str(float(mid[0][0].mean())))
check("blend pooled midpoint", abs(float(mid[0][1]["pooled_output"].mean()) - 0.5) < 1e-6)

full = blend_conditioning(condB, condA, 1.0)
check("strength 1 = to", float(full[0][0].mean()) == 1.0)
zero = blend_conditioning(condB, condA, 0.0)
check("strength 0 = from", float(zero[0][0].mean()) == 0.0)

# mismatched token lengths reconcile without crashing
condShort = [[torch.ones(1, 20, 768), {"pooled_output": torch.ones(1, 768)}]]
out = blend_conditioning(condB, condShort, 0.5)
check("mismatched tokens ok", out[0][0].shape == (1, 77, 768), str(out[0][0].shape))

print("prompt batch (AnimateDiff bridge):")
# simulate a per-frame conditioning list, then batch it
per_frame = [
    [[torch.full((1, 77, 768), float(i)), {"pooled_output": torch.full((1, 768), float(i))}]]
    for i in range(8)
]
# inline the node logic (import the class)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from difforum.nodes.prompt_nodes import DifforumPromptBatch  # noqa: E402

out = DifforumPromptBatch().run(per_frame)[0]
check("batch single entry", len(out) == 1 and len(out[0]) == 2)
check("batch cond shape [N,T,D]", tuple(out[0][0].shape) == (8, 77, 768), str(tuple(out[0][0].shape)))
check("batch pooled [N,D]", tuple(out[0][1]["pooled_output"].shape) == (8, 768))
check("frame i keeps prompt i", float(out[0][0][3].mean()) == 3.0)

# mismatched token lengths pad to max
mixed = [
    [[torch.ones(1, 20, 768), {"pooled_output": torch.ones(1, 768)}]],
    [[torch.ones(1, 77, 768), {"pooled_output": torch.ones(1, 768)}]],
]
out = DifforumPromptBatch().run(mixed)[0]
check("batch pads tokens", tuple(out[0][0].shape) == (2, 77, 768), str(tuple(out[0][0].shape)))

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL PROMPT TESTS PASSED")
