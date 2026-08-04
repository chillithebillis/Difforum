"""Test bootstrap: make the pack importable as `difforum` whatever the folder is called.

`tests/test_integration.py` and `tests/test_prompt.py` import node modules as
`difforum.nodes.*` on purpose: those tests inject a stub top-level `nodes`
module (standing in for ComfyUI's own), so importing the pack's internal
`nodes` subpackage by bare name would collide with the stub.

The catch is the folder name. A `git clone` produces `Difforum`; a Comfy
Registry install produces `difforum`. Python matches module names
case-sensitively even on macOS, where the filesystem does not - so hardcoding
either spelling breaks the suite on the other layout, and on Linux CI.

This binds the name to whatever the directory actually is, via a module whose
`__path__` points at the pack. Importing the package's own `__init__` is
deliberately avoided: it pulls in every node module, which needs `comfy` at
import time and is not available under bare pytest.
"""

import sys
import types
from pathlib import Path

_PACK = Path(__file__).resolve().parent

# `from core.x import ...` / `from nodes.x import ...` - how most tests import.
if str(_PACK) not in sys.path:
    sys.path.insert(0, str(_PACK))

# `from difforum.nodes.x import ...` - the package-qualified form.
if "difforum" not in sys.modules:
    _alias = types.ModuleType("difforum")
    _alias.__path__ = [str(_PACK)]
    sys.modules["difforum"] = _alias
