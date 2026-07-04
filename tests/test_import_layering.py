"""Guard: the core boundary modules must import without pulling in torch (D7).

Kept fast and deterministic by checking in a fresh subprocess, so the result does not
depend on whether an earlier test in the session already imported torch.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_core_modules_import_without_torch() -> None:
    code = (
        "import importlib, sys\n"
        "for module in ('src.config', 'src.schemas', 'src.utils'):\n"
        "    importlib.import_module(module)\n"
        "assert 'torch' not in sys.modules, 'torch leaked into the core import graph'\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
