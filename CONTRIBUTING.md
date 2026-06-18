# Contributing to Difforum

Thanks for your interest. Difforum is MIT-licensed and built to be extended.

## Ground rules

- **Keep dependencies minimal.** Core logic uses numpy and a safe AST evaluator;
  torch is only used inside nodes that need it. No librosa / numexpr / pandas, so
  the package keeps installing cleanly on Python 3.12 and the Comfy Registry.
- **Run the tests** before opening a PR (no GPU needed):

  ```bash
  for t in core audio hybrid models warp color effects prompt plot integration; do
    python tests/test_$t.py
  done
  ```

- **Match the surrounding style** (naming, comment density, no em dashes).
- **Regenerate the example workflows** if you change a node's inputs:
  `python examples/_build_examples.py` then `python examples/_validate_workflows.py`.

## Developer Certificate of Origin (DCO)

To keep the project's licensing clean and flexible, contributions are accepted
under the [DCO](https://developercertificate.org/): you certify that you wrote
the patch (or have the right to submit it) and agree it can ship under the
project's MIT license. Sign off each commit:

```bash
git commit -s -m "your message"
```

which adds a `Signed-off-by: Your Name <you@example.com>` line.

## Scope ideas

Good places to extend: new schedule functions, camera presets, effects (the
`core/effects.py` and `core/symmetry.py` style), reactive audio modes, model
catalog recipes, and the realtime path (NDI sink, MIDI/OSC control). See
[DESIGN.md](DESIGN.md) and [REALTIME.md](REALTIME.md).
