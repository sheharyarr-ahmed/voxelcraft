# VoxelCraft — Project Instructions

Stable Diffusion 1.5 + LoRA application pipeline with ControlNet, Gradio Blocks UI, deployed at $0.00 on Hugging Face Spaces ZeroGPU (CPU-basic fallback). Portfolio artifact for Shery Labs; sole author Sheharyar Ahmed.

Requirements live in `SPEC.md` at the repo root. Every locked decision lives in `.claude/agents/meta/decisions.md` — log new decisions there, with rationale and alternatives, before acting on them.

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

## Phase plan (each phase gated by its acceptance check and at least one commit)

0. **Scaffold** — `.claude/` populated, commit-msg hook proven, venv + requirements, empty suite green, verify.sh wired.
1. **Local inference smoke test** — one 512x512 image locally, LoRA registry with 3 verified entries, registry tests green.
2. **Gradio app** — three-tab Blocks UI, lazy loading, Pydantic boundaries, friendly error states, metadata panel; Tab 1 end to end locally.
3. **ControlNet** — reference photo → skeleton preview → posed stylized output, locally.
4. **Deploy + docs** — live Space, README with HF YAML frontmatter, ARCHITECTURE.md, LORA_TRAINING.md, Colab notebook, examples, GitHub metadata polished.
5. **Portfolio conversion** — separate session, post-ship.

## Manual pause points (stop and ask Sheharyar — his answers, not yours)

- Phase 0: GitHub repo settings confirmation.
- Phase 1: civitai LoRA license verification.
- Phase 4: HF account + Space creation + ZeroGPU hardware request; GitHub metadata polish (About, topics, social preview, website field).

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
