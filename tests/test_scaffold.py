"""Phase 0 scaffold guards.

SPEC decision D10: single-author git history, mechanically enforced.
These tests run the commit-msg hook as a subprocess against attribution
strings that must be rejected and normal messages that must pass, so the
guarantee is enforced by the suite continuously, not only by a one-off
live demonstration.
"""

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / ".githooks" / "commit-msg"

REJECTED_MESSAGES = [
    "Add pipeline\n\nCo-Authored-By: Claude <noreply@anthropic.com>",
    "Add pipeline\n\nco-authored-by: claude fable <noreply@anthropic.com>",
    "Add pipeline\n\nGenerated with Claude Code",
    "Add pipeline\n\n\U0001f916 Generated with [Claude Code](https://claude.com/claude-code)",
    "Wire scheduler, credit to https://anthropic.com",
]

ACCEPTED_MESSAGES = [
    "Phase 0: scaffold .claude agents, rules, and verify gate",
    "Clamp LoRA weight at the 1.5 upper bound",
    "Add OpenPose skeleton preview to ControlNet tab",
]


def run_hook(message: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    msg_file = tmp_path / "COMMIT_EDITMSG"
    msg_file.write_text(message, encoding="utf-8")
    return subprocess.run([str(HOOK), str(msg_file)], capture_output=True, text=True)


def test_hook_exists_and_is_executable() -> None:
    assert HOOK.exists(), "commit-msg hook missing"
    assert HOOK.stat().st_mode & 0o111, "commit-msg hook is not executable"


@pytest.mark.parametrize("message", REJECTED_MESSAGES)
def test_hook_rejects_attribution(message: str, tmp_path: Path) -> None:
    result = run_hook(message, tmp_path)
    assert result.returncode != 0, f"hook accepted attribution string: {message!r}"


@pytest.mark.parametrize("message", ACCEPTED_MESSAGES)
def test_hook_accepts_clean_messages(message: str, tmp_path: Path) -> None:
    result = run_hook(message, tmp_path)
    assert result.returncode == 0, result.stderr
