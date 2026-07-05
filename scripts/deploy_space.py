"""Deploy VoxelCraft to its Hugging Face Space (create if needed, then upload the working tree).

Requires an HF write token to be logged in first: `venv/bin/hf auth login`. Run from the repo
root with the project venv:

    venv/bin/python scripts/deploy_space.py

Uploads the current working tree to the Space via the HTTP API (no git push / credential setup),
ignoring the venv, git internals, model caches, and other non-runtime files. Check build status
with: HfApi().get_space_runtime(REPO_ID).stage  (want "RUNNING").
"""

from __future__ import annotations

import sys

from huggingface_hub import HfApi, create_repo

REPO_ID = "sheryyahmed457/voxelcraft"

IGNORE = [
    ".git/*",
    ".git/**",
    "venv/*",
    "venv/**",
    "models/*",
    "models/**",
    "*__pycache__*",
    "*.pyc",
    ".pytest_cache/*",
    ".mypy_cache/*",
    ".ruff_cache/*",
    "smoke_output.png",
    "*.safetensors",
    ".claude/settings.local.json",
]


def main() -> None:
    api = HfApi()
    print("authenticated as:", api.whoami()["name"])

    url = create_repo(REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True)
    print("space repo:", url)

    commit = api.upload_folder(
        folder_path=".",
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy VoxelCraft: SD 1.5 LoRA application pipeline with ControlNet",
        ignore_patterns=IGNORE,
    )
    print("upload commit:", commit)
    print("SPACE_URL: https://huggingface.co/spaces/" + REPO_ID)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # surface the real error for triage
        print("DEPLOY_ERROR:", type(exc).__name__, str(exc), file=sys.stderr)
        raise
