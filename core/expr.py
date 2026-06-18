"""
Safe math expression evaluator for Difforum schedules.

No external deps, no eval(). Parses a restricted subset of Python expressions
via the `ast` module and walks the tree with a whitelist of operators,
functions and variables. This is what powers Deforum-style schedule strings
like:  0:(0), 60:(0.5*sin(2*pi*t/30))

Available in expressions:
  variables : t  (current frame, float)
              f  (alias of t)
              s  (seconds = t / fps)
              fps, max_f
              pi, e, tau
              any extra vars injected by the caller (e.g. audio.* flattened
              to amp, low, mid, high, onset, beat ...)
  functions : sin cos tan asin acos atan atan2 sinh cosh tanh
              abs sqrt exp log log10 pow floor ceil round
              min max sign clamp clip lerp smoothstep
"""

from __future__ import annotations

import ast
import math
from typing import Mapping


# ----------------------------------------------------------------------------
# whitelisted functions
# ----------------------------------------------------------------------------

def _sign(x: float) -> float:
    return (x > 0) - (x < 0)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    if edge0 == edge1:
        return 0.0 if x < edge0 else 1.0
    t = _clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


FUNCTIONS = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "abs": abs, "sqrt": math.sqrt, "exp": math.exp,
    "log": math.log, "log10": math.log10, "pow": math.pow,
    "floor": math.floor, "ceil": math.ceil, "round": round,
    "min": min, "max": max,
    "sign": _sign, "clamp": _clamp, "clip": _clamp,
    "lerp": _lerp, "smoothstep": _smoothstep,
}

CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau}

# ast node types we allow
_ALLOWED_NODES = (
    ast.Expression, ast.Constant, ast.Name, ast.Load,
    ast.BinOp, ast.UnaryOp, ast.Call,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
    ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    ast.IfExp, ast.Compare,
    ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq,
    ast.BoolOp, ast.And, ast.Or,
)


class ExprError(ValueError):
    """Raised when an expression is malformed or uses disallowed names."""


class CompiledExpr:
    """A parsed, validated expression ready to be evaluated many times."""

    __slots__ = ("source", "_code", "_const")

    def __init__(self, source: str):
        self.source = source
        try:
            tree = ast.parse(source, mode="eval")
        except SyntaxError as exc:  # noqa: PERF203
            raise ExprError(f"invalid expression {source!r}: {exc}") from exc
        _validate(tree)
        self._code = compile(tree, "<difforum-expr>", "eval")
        # detect constant expressions (no free variables) for fast-path
        self._const = not _references_vars(tree)

    @property
    def is_constant(self) -> bool:
        return self._const

    def eval(self, variables: Mapping[str, float] | None = None) -> float:
        env = {"__builtins__": {}}
        env.update(CONSTANTS)
        env.update(FUNCTIONS)
        if variables:
            env.update(variables)
        try:
            return float(eval(self._code, env))  # noqa: S307 - AST is whitelisted
        except ZeroDivisionError:
            return 0.0
        except Exception as exc:  # noqa: BLE001
            raise ExprError(f"error evaluating {self.source!r}: {exc}") from exc


def _validate(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ExprError(
                f"disallowed syntax {type(node).__name__} in expression"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
                name = getattr(node.func, "id", "?")
                raise ExprError(f"unknown function {name!r}")


_RESERVED = set(FUNCTIONS) | set(CONSTANTS)


def _references_vars(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in _RESERVED:
            return True
    return False


def compile_expr(source: str) -> CompiledExpr:
    """Parse and validate `source`, returning a reusable CompiledExpr."""
    return CompiledExpr(str(source).strip())
