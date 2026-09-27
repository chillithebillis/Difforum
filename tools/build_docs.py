"""
Generate docs/NODES.md from the nodes themselves (description, inputs,
tooltips, outputs), so the reference can never lag behind the code.

    python tools/build_docs.py            # write
    python tools/build_docs.py --check    # fail if docs/NODES.md is stale
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path

PACK = Path(__file__).resolve().parents[1]
if "difforum" not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        "difforum", PACK / "__init__.py", submodule_search_locations=[str(PACK)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["difforum"] = mod
    spec.loader.exec_module(mod)

from difforum.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS  # noqa: E402

WIDGETS = {"INT", "FLOAT", "STRING", "BOOLEAN"}


def _type(spec) -> str:
    t = spec[0]
    if isinstance(t, (list, tuple)):
        shown = ", ".join(map(str, t[:6])) + (", ..." if len(t) > 6 else "")
        return f"choice ({shown})"
    return str(t)


def _default(spec) -> str:
    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
    d = opts.get("default")
    if d is None:
        return ""
    s = str(d).replace("\n", " ")
    return s if len(s) < 40 else s[:37] + "..."


def render() -> str:
    by_cat: dict[str, list] = {}
    for cid, cls in NODE_CLASS_MAPPINGS.items():
        by_cat.setdefault(cls.CATEGORY, []).append((cid, cls))
    out = ["# Node reference", "",
           "Generated from the code by `tools/build_docs.py`. Every node also shows this text "
           "as its description inside ComfyUI.", ""]
    for cat in sorted(by_cat):
        out += [f"## {cat.split('/', 1)[1]}", ""]
        for cid, cls in by_cat[cat]:
            out += [f"### {NODE_DISPLAY_NAME_MAPPINGS[cid]}", "", f"`{cid}`", ""]
            doc = inspect.cleandoc(cls.__doc__ or "")
            if doc:
                out += [doc, ""]
            it = cls.INPUT_TYPES()
            rows = []
            for section in ("required", "optional"):
                for name, spec in it.get(section, {}).items():
                    opts = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                    tip = str(opts.get("tooltip", "")).replace("|", "/")
                    opt = "" if section == "required" else " *(optional)*"
                    rows.append(f"| `{name}`{opt} | {_type(spec)} | {_default(spec)} | {tip} |")
            if rows:
                out += ["| input | type | default | notes |", "|---|---|---|---|", *rows, ""]
            names = getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)
            outs = ", ".join(f"`{n}` ({t})" for n, t in zip(names, cls.RETURN_TYPES))
            if outs:
                out += [f"**Outputs:** {outs}", ""]
    return "\n".join(out).rstrip() + "\n"


if __name__ == "__main__":
    text = render()
    path = PACK / "docs" / "NODES.md"
    if "--check" in sys.argv:
        if not path.exists() or path.read_text(encoding="utf-8") != text:
            print("docs/NODES.md is stale - run python tools/build_docs.py")
            sys.exit(1)
    else:
        path.write_text(text, encoding="utf-8")
        print("wrote docs/NODES.md")
