# Contributing to Difforum

Thanks for your interest. Difforum is MIT-licensed and built to be extended.

## Ground rules

- **Keep dependencies minimal.** The engine uses numpy and torch (which ship with
  ComfyUI); OpenCV only where a node needs it, and imported lazily. No librosa,
  numexpr or pandas.
- **Engine first.** Put logic in `core/` (pure, testable, no ComfyUI), and keep
  nodes thin. Import the engine relatively (`from ..core import ...`).
- **Every node gets a docstring and tooltips.** They become the in-app
  description and `docs/NODES.md`.
- **Style:** match the surrounding code, no em dashes, and run `ruff check .`

## Before a pull request

```bash
pytest                                      # no GPU or ComfyUI needed
python tools/build_workflows.py             # if you changed any node inputs
python tools/build_docs.py                  # if you changed docstrings / tooltips
ruff check .
```

CI runs all of these, plus `--check` on the generated files.

## Developer Certificate of Origin

Contributions are accepted under the [DCO](https://developercertificate.org/).
Sign off each commit with `git commit -s`.

## Good places to extend

- New camera moves: `core/camera_presets.py`. Add the expressions and a
  `MOVE_INFO` entry; the Director picks it up automatically, and a glyph in
  `js/director_timeline.js` is optional.
- New audio reactions: `REACTIONS` and `_react()` in `core/direction.py`.
- More bridges: follow `nodes/bridges.py`. Output plain types, or call core
  nodes through `call_comfy_node()`.
- Camera interchange for more DCCs (Nuke, Houdini, Unreal): `core/export.py`.
