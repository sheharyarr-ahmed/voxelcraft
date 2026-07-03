# Decision Log — VoxelCraft

Append-only. Every locked decision from SPEC.md is logged here with rationale and the
alternatives it beat. New decisions and amendments get a dated entry before the change
lands. This is the file the `voxelcraft-architect` agent reviews against.

(Note: this is a log, not an agent definition — it lives here because SPEC.md locks the path.)

## Locked decisions (SPEC.md, logged 2026-07-03)

**D1 — Hugging Face Spaces ZeroGPU over Vercel / Railway / Modal / Replicate.**
SD needs GPU inference and multi-GB weights; Vercel and Railway cannot host that. Modal and
Replicate are paid. ZeroGPU is the only zero-cost GPU host and the category-native surface SD
buyers expect. Fallback: CPU-basic Space (2–5 min per image, demo-viable with queue messaging).

**D2 — Stable Diffusion 1.5 over SDXL.**
SD 1.5 + LoRA + ControlNet fits the ZeroGPU VRAM envelope (~6–8 GB). SDXL needs 12 GB+ and
risks OOM. Stability beats output quality for a portfolio demo.

**D3 — No Anthropic API at runtime.**
The AI provider is open-source Stable Diffusion. Claude in the runtime would add cost and prove
nothing this project exists to prove. Claude Code is the build tool only.

**D4 — Pre-trained public LoRAs only, license-verified by hand.**
Three civitai LoRAs (one anime, one realistic, one painterly), each personally verified by
Sheharyar for commercial-use permission on the model card. Manual pause point — never marked
verified by build tooling.

**D5 — Gradio Blocks over Streamlit.**
Native HF Spaces SDK, built-in request queue, zero-friction deploy.

**D6 — Pydantic v2 at every input boundary.**
Prompt length, weight bounds, upload validation. Same boundary discipline as Zod in ReelMind.

**D7 — Lazy model loading.**
Models load on first generation, not app start. Cold-start discipline for Spaces.

**D8 — fp16 + attention slicing + device detection.**
VRAM discipline with graceful CPU fallback.

**D9 — Committed .claude/ scaffold.**
The build scaffold is itself a portfolio credibility artifact, repeating the ReelMind pattern.

**D10 — Single-author git history, mechanically enforced.**
`.githooks/commit-msg` rejects "Co-Authored-By: Claude", "Generated with Claude Code", and any
AI-attribution string; `core.hooksPath=.githooks`; author locked to Sheharyar Ahmed. The HF
Space is a second git remote; the same rules apply to every push there. Never `--no-verify`.

**D11 — Local smoke test is correctness-only.**
M1 8 GB CPU generation takes minutes per image; acceptable once. If local generation OOMs,
smoke-test directly on Spaces hardware and log that decision here.

**D12 — Manual UI steps are explicit pause points.**
GitHub repo settings confirmation (Phase 0); civitai LoRA license verification (Phase 1);
HF account + Space creation + ZeroGPU request and GitHub metadata polish (Phase 4).

## Amendments

**A1 (2026-07-03) — Skills are directories, not flat files.**
SPEC's file tree lists `.claude/skills/<name>.md`, but Claude Code only discovers skills at
`.claude/skills/<name>/SKILL.md`. Working tooling beats literal tree fidelity; the four skills
keep their spec names.

**A2 (2026-07-03) — `requirements-dev.txt` added.**
pytest/black/isort/mypy are build-time tools. Keeping them out of `requirements.txt` keeps the
HF Spaces runtime install lean. Alternative (one merged file) rejected: dev tooling in the
production image.

**A3 (2026-07-03) — `tests/test_scaffold.py` added in Phase 0.**
Makes the Phase 0 suite green with a real assertion — the commit-msg hook rejects attribution
strings and accepts clean messages — instead of a placeholder test, and keeps D10 continuously
enforced by the suite rather than only by the one-off live demo.
