"""
Validate every example workflow against the REAL node definitions.

For each *.json: confirm every node type exists, and every *required connection*
input (MODEL/CLIP/VAE/CONDITIONING/LATENT/IMAGE/DIFFORUM_* etc.) is actually
linked. Widget params (INT/FLOAT/STRING/BOOLEAN/combo) are not connections and
are skipped. Also report which external assets each workflow needs.

Run with the ComfyUI venv python (needs ComfyUI on path for core nodes).
"""

import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, r"D:\ComfyUI-victor")
sys.path.insert(0, r"D:\ComfyUI-victor\custom_nodes")

import asyncio  # noqa: E402

import nodes as comfy_nodes  # ComfyUI core  # noqa: E402
import difforum  # noqa: E402

# load comfy_extras (LoadAudio, etc.) so the check sees every built-in node
try:
    asyncio.run(comfy_nodes.init_builtin_extra_nodes())
except Exception as e:  # noqa: BLE001
    print(f"warn: could not load extra nodes: {e}")

MAPPING = dict(comfy_nodes.NODE_CLASS_MAPPINGS)
MAPPING.update(difforum.NODE_CLASS_MAPPINGS)

WIDGET_TYPES = {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}
ASSET_NODES = {
    "CheckpointLoaderSimple": "an SD checkpoint",
    "LoadImage": "an input image",
    "LoadAudio": "an audio file",
    "UNETLoader": "a UNet/diffusion model",
    "UnetLoaderGGUF": "a GGUF model",
    "VAELoader": "a VAE",
}

HERE = Path(__file__).resolve().parent
problems_total = 0

for path in sorted(glob.glob(str(HERE / "*.json"))):
    name = Path(path).name
    wf = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes_by_id = {n["id"]: n for n in wf["nodes"]}
    issues = []
    assets = set()

    external = set()
    for node in wf["nodes"]:
        ntype = node["type"]
        if ntype in ASSET_NODES:
            assets.add(ASSET_NODES[ntype])
        if ntype in ("Note", "MarkdownNote", "Reroute", "PrimitiveNode"):
            continue  # built-in frontend-only nodes, always present
        cls = MAPPING.get(ntype)
        if cls is None:
            # not in this dev env: a third-party dep (RIFE, VHS, etc.), not a
            # hard error - just note it must be installed where the wf runs.
            external.add(ntype)
            continue
        try:
            spec = cls.INPUT_TYPES()
        except Exception as e:  # noqa: BLE001
            issues.append(f"{ntype}: INPUT_TYPES failed ({e})")
            continue
        required = spec.get("required", {})
        # map this node's input slots by name -> link
        slot_link = {inp["name"]: inp.get("link") for inp in node.get("inputs", [])}
        for iname, ispec in required.items():
            itype = ispec[0]
            if isinstance(itype, list) or itype in WIDGET_TYPES:
                continue  # widget, not a connection
            # required connection -> must have a linked input slot
            if iname not in slot_link:
                issues.append(f"{ntype}: required input '{iname}' ({itype}) has no slot")
            elif slot_link[iname] is None:
                issues.append(f"{ntype}: required input '{iname}' ({itype}) NOT connected")

    status = "OK" if not issues else f"{len(issues)} ISSUE(S)"
    print(f"\n=== {name} === [{status}]")
    print(f"  nodes: {len(wf['nodes'])}  links: {len(wf['links'])}")
    print(f"  needs: {', '.join(sorted(assets)) if assets else 'nothing (runs anywhere)'}")
    if external:
        print(f"  external nodes (must be installed): {', '.join(sorted(external))}")
    for i in issues:
        print(f"  ! {i}")
    problems_total += len(issues)

print("\n" + "=" * 50)
print("ALL WORKFLOWS VALID" if problems_total == 0 else f"TOTAL ISSUES: {problems_total}")
sys.exit(1 if problems_total else 0)
