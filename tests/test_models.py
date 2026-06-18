"""Integrity tests for the Difforum model catalog."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import models as M  # noqa: E402

_failures = []


def check(name, cond, detail=""):
    if not cond:
        _failures.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}{('  -> ' + detail) if detail and not cond else ''}")


print("model catalog:")

# every entry has a valid role and non-empty required fields
for m in M.all_entries():
    ok = m.role in M.ROLES and m.name and m.source and m.trainable and m.family
    check(f"entry {m.key} well-formed", ok, repr(m))

# keys are unique
keys = [m.key for m in M.all_entries()]
check("unique keys", len(keys) == len(set(keys)))

# every recipe references existing model keys
for r in M.recipes():
    for k in r.parts:
        check(f"recipe {r.key} -> {k} exists", k in keys, f"missing {k}")

# family filters return something sensible
check("sd15 has checkpoints", any(m.role == "checkpoint" for m in M.by_family("sd15")))
check("wan22 has video_model", any(m.role == "video_model" for m in M.by_family("wan22")))

# recipes filter by family
check("sd15 recipes exist", len(M.recipes("sd15")) >= 2)
check("wan22 recipes exist", len(M.recipes("wan22")) >= 1)

# every recipe family has training info
for r in M.recipes():
    check(f"training info for {r.family}", r.family in M.TRAINING, r.family)

# summarize works and mentions a source URL
for k in M.recipe_keys():
    s = M.summarize_recipe(k)
    check(f"summarize {k}", isinstance(s, str) and ("http" in s), "no url in summary")

# as_dict round-trips for node output
d = M.get_recipe("wan22_12gb").as_dict()
check("recipe as_dict has models", isinstance(d.get("models"), list) and len(d["models"]) == 2)

print()
if _failures:
    print(f"FAILED ({len(_failures)}): {', '.join(_failures)}")
    sys.exit(1)
print("ALL MODEL TESTS PASSED")
