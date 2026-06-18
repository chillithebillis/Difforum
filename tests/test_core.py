"""
Standalone tests for the Difforum schedule engine.

Run from the package root with the ComfyUI venv python:
    python -m tests.test_core
(or with pytest if available). No torch / ComfyUI imports here.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import build_schedule, compile_expr  # noqa: E402
from core.expr import ExprError  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    status = "ok  " if cond else "FAIL"
    if not cond:
        _failures.append(name)
    print(f"  [{status}] {name}{('  -> ' + detail) if detail and not cond else ''}")


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


# --- expression evaluator ---------------------------------------------------
print("expr evaluator:")
e = compile_expr("2 + 3 * 4")
check("arithmetic", approx(e.eval(), 14.0), str(e.eval()))
check("constant detection", e.is_constant)

e = compile_expr("0.5*sin(2*pi*t/30)")
check("sin uses t", not e.is_constant)
check("sin value at t=7.5", approx(e.eval({"t": 7.5}), 0.5 * math.sin(2 * math.pi * 7.5 / 30)))

e = compile_expr("clamp(t, 0, 10)")
check("clamp high", approx(e.eval({"t": 99}), 10.0))
check("clamp low", approx(e.eval({"t": -5}), 0.0))

e = compile_expr("t if t > 5 else -t")
check("ternary true", approx(e.eval({"t": 8}), 8.0))
check("ternary false", approx(e.eval({"t": 2}), -2.0))

e = compile_expr("1/0")
check("zero division -> 0", approx(e.eval(), 0.0))

# security: forbidden names / calls must raise at compile time
for bad in ("__import__('os')", "open('x')", "exec('1')", "a.b", "[1,2,3]"):
    try:
        compile_expr(bad)
        check(f"reject {bad!r}", False, "did not raise")
    except ExprError:
        check(f"reject {bad!r}", True)

# --- schedule interpolation -------------------------------------------------
print("schedule interpolation:")
s = build_schedule("0:(0), 10:(100)", max_frames=11, fps=10)
check("linear endpoints", approx(s.at(0), 0.0) and approx(s.at(10), 100.0))
check("linear midpoint", approx(s.at(5), 50.0), str(s.at(5)))
check("length", len(s) == 11)

s = build_schedule("0:(0), 10:(100)", max_frames=11, easing="step")
check("step holds left", approx(s.at(5), 0.0) and approx(s.at(10), 100.0))

s = build_schedule("0:(5)", max_frames=5)
check("single keyframe constant", all(approx(v, 5.0) for v in s.as_list()))

s = build_schedule("42", max_frames=3)
check("bare expr shorthand", all(approx(v, 42.0) for v in s.as_list()))

# oscillator keeps oscillating across a single segment
s = build_schedule("0:(sin(t))", max_frames=8, fps=24)
check("oscillator at t=0", approx(s.at(0), 0.0))
check("oscillator at t=1", approx(s.at(1), math.sin(1)))

# seconds variable
s = build_schedule("0:(s)", max_frames=25, fps=24)
check("seconds var", approx(s.at(24), 1.0), str(s.at(24)))

# --- audio-reactive extra vars ----------------------------------------------
print("audio-reactive vars:")
amp = [0.0, 0.5, 1.0, 1.0]
s = build_schedule("0:(0.3 + 0.7*amp)", max_frames=4, extra_vars={"amp": amp})
check("amp frame0", approx(s.at(0), 0.3))
check("amp frame2", approx(s.at(2), 1.0))
check("amp clamps short series", approx(s.at(3), 1.0))

# --- summary ----------------------------------------------------------------
print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL TESTS PASSED")
