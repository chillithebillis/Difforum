"""Run the 0.x script-style engine checks (tests/scripts/check_*.py) as tests."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = sorted((Path(__file__).parent / "scripts").glob("check_*.py"))


@pytest.mark.parametrize("script", SCRIPTS, ids=[s.stem for s in SCRIPTS])
def test_script(script):
    r = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
