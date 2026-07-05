# VoxelCraft — Project Instructions

Stable Diffusion 1.5 + LoRA application pipeline with ControlNet, Gradio Blocks UI, **live at $0.00 on Hugging Face Spaces (free CPU-basic)**. Portfolio artifact for Shery Labs; sole author Sheharyar Ahmed.

Requirements live in `SPEC.md` at the repo root. Every locked decision lives in `.claude/agents/meta/decisions.md` — log new decisions there, with rationale and alternatives, before acting on them.

## Current status & handoff (updated 2026-07-06)

**Shipped and live** on the free tier. Phases 0–4 are complete; the three tasks in "Next session — start here" below plus Phase 5 remain.

- **Live app:** https://sheryyahmed457-voxelcraft.hf.space · **Space page:** https://huggingface.co/spaces/sheryyahmed457/voxelcraft · **GitHub:** https://github.com/sheharyarr-ahmed/voxelcraft (all pushed, single-author).
- **Verified live:** Tab 1 (text→image + LoRA) generates on the Space with a populated metadata panel. Tab 2 (pose/ControlNet) is coded but **not yet live-verified** (next-session task).
- **Hardware = free CPU-basic** (~4–7 min/image). **ZeroGPU is now PRO-only** ($9/mo; the API returns `402` — decision A15). The `@spaces.GPU` path is wired but dormant, gated on the `SPACES_ZERO_GPU` env var; a hardware switch enables it with no code change if PRO is ever added.
- **LoRA registry = 2 entries** (`pixelart`, `render3d`), both HF-hosted and load-verified. Watercolor and CuteCartoon were dropped: LoRAs with **text-encoder layers** crash diffusers 0.39 `load_lora_weights` with `IndexError` (A13). Any new LoRA must be load-verified (UNet-only) before registering.
- **HF auth:** already logged in as `sheryyahmed457` (write token at `~/.cache/huggingface/token`).
- **Redeploy:** `venv/bin/python scripts/deploy_space.py` (uploads the working tree to the Space via `HfApi().upload_folder`, ignoring venv/models/caches). Build status: `HfApi().get_space_runtime('sheryyahmed457/voxelcraft').stage` (want `RUNNING`).
- **Local generation OOMs** on the 8 GB M1 at the VAE decode (D11/A11) — verify inference on the Space, never locally. Unit tests, mypy, and `import app; build_demo()` run fine locally.
- **Open minor item:** the Stop-hook gate (`.claude/verify.sh` + `.claude/settings.json`) was never installed (permission classifier blocked agent-authored hooks); run `venv/bin/pytest -q` manually. Optional.

## Anti-fabrication wall (HARD — governs all code, docs, commits, and copy)

- **Never claim:** trained a custom LoRA, GPU training pipeline, custom model training experience, production AI video pipeline.
- **Always claim (accurate):** LoRA application pipeline with weight blending, ControlNet integration for pose-controlled generation, Diffusers production patterns, HF Spaces deployment workflow, documented training methodology.
- The README carries an explicit "What this is / What this is not" section. The training notebook is a reference, labeled "not executed by author on custom data".

## Hard constraints

- Budget $0.00: no paid APIs, no paid hosting, no Anthropic API at runtime.
- Single-author git history: no Co-Authored-By, no "Generated with Claude Code", no AI attribution in any commit on any remote. Enforced by `.githooks/commit-msg` (`core.hooksPath=.githooks`). Never bypass with `--no-verify`.
- Pre-trained public LoRAs only, commercial-use license verified by Sheharyar personally. Pause and ask; never mark an entry verified yourself.
- SD 1.5 only (no SDXL). fp16 + attention slicing + device detection. Lazy model loading.
- Pydantic v2 at every input boundary.

## Rules

@rules/code-style.md
@rules/anti-patterns.md
@rules/license-compliance.md

## Phase plan (status)

0. **Scaffold** ✅ (commit 72ae3e8).
1. **Local inference smoke test** ✅ — pipeline + registry; local generation deferred to Spaces (D11/A11).
2. **Gradio app** ✅ — three-tab Blocks; Tab 1 verified live on the Space.
3. **ControlNet** ✅ — coded and unit-tested; **live pose verification pending** (next-session task 3).
4. **Deploy + docs** ✅ — live Space, README/ARCHITECTURE/DEPLOY/notebook. Remaining polish → next-session tasks 1–2.
5. **Portfolio conversion** — separate session, post-ship (demo recording, Upwork entry, LinkedIn).

## Next session — start here (three tasks, all $0)

1. **GitHub metadata polish.** Set the repo About description and topics (`stable-diffusion`, `lora`, `controlnet`, `diffusers`, `gradio`, `huggingface`, `generative-ai`, `python`), the website/homepage field → the live Space URL, and a social-preview image (true-black/mint brand). `gh repo edit sheharyarr-ahmed/voxelcraft --description ... --homepage ... --add-topic ...` sets the first three; the social preview is set in the GitHub web UI. **Confirm the About text with Sheharyar** before applying.
2. **Example images.** Generate 3–4 outputs on the live Space (Tab 1, pixelart/render3d LoRAs), commit them to `examples/` (small file size), and reference them in the README so the repo shows output without the ~6-min wait. Generation is browser-driven (the API is private) — Sheharyar generates and hands over the files, or approve temporarily exposing the API to script it.
3. **Pose-tab live test.** On the Space, upload a clear photo of a person → confirm skeleton preview → posed stylized output; confirm a no-person photo returns the friendly "no pose detected" error and a >5 MB upload is rejected. Closes the Phase 3 live gate.

## Manual pause points (stop and ask Sheharyar — his answers, not yours)

- LoRA license verification — done for the current 2 (A13); any new LoRA needs his sign-off and must be load-verified.
- **GitHub metadata polish** (next-session task 1) — confirm the About text with him.

## Commands

- Tests: `venv/bin/pytest -q`
- Format: `venv/bin/black --line-length 100 src tests && venv/bin/isort src tests`
- Types: `venv/bin/mypy --strict src`
- Gate: `.claude/verify.sh` — wired as the Stop hook; a red suite blocks completion.

Inference itself is not unit-tested (nondeterministic, GPU-bound); it is covered by phase acceptance checks and the live demo.

## Subagents

- `voxelcraft-architect` (opus) — design review, anti-pattern detection. Consult before structural changes.
- `diffusers-doctor` (sonnet) — diffusers code review, VRAM discipline. Run after touching `src/pipelines/`.
- `gradio-builder` (sonnet) — Blocks UI and Spaces patterns for `app.py`.
- `hf-deployer` (sonnet) — deployment checklist, Phase 4 release gate.
- `content-writer` (haiku) — README and architecture docs, honest framing.
