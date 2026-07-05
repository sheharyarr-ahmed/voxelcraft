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

**A4 (2026-07-05) — `peft` added to requirements.txt.**
diffusers 0.39's multi-adapter API (`set_adapters`, `enable_lora`, `disable_lora`, and
`load_lora_weights(..., adapter_name=)` stacking) raises `ValueError("PEFT backend is required")`
without it — verified against the installed venv source. Alternatives rejected: `fuse_lora` (banned
per D-loras — bakes weights, kills per-request re-weighting); the legacy single-LoRA path (cannot
stack the 1–2 adapters the SPEC requires).

**A5 (2026-07-05) — `pyproject.toml` created.**
Pre-authorized by `rules/code-style.md` ("config lands in pyproject.toml … once that file is
created"). Forced now because mypy `--strict` needs a home for per-module overrides:
`controlnet_aux.*` ships no py.typed (→ `ignore_missing_imports`), and diffusers/transformers ship
py.typed over largely untyped implementations (→ `disallow_untyped_calls = false` scoped to
`src.pipelines.*`). Also carries black/isort/pytest config. No `[project]`/`[build-system]` — this
is a Gradio app, not a distributable package; Spaces installs from requirements.txt.

**A6 (2026-07-05) — D11 smoke test lives at `scripts/smoke_test.py`, excluded from pytest.**
The Stop hook runs `pytest -q` on every session stop; a minutes-long CPU generation must never fire
there. A standalone script pytest never collects removes that risk entirely (chosen over a
`@pytest.mark.smoke` + `addopts` arrangement, which is strictly more moving parts). Also records that
`scripts/` and `src/exceptions.py` are additions beyond SPEC's literal file tree (A1 precedent:
working structure beats tree fidelity).

**A7 (2026-07-05) — gradio 6 API corrections.**
Audited against installed gradio 6.19: `launch(show_api=...)` was removed — the API surface is
hidden per event listener via `api_visibility="private"` (SPEC excludes a programmatic API), and
`show_progress` takes `"full"|"minimal"|"hidden"` literals, not booleans. `agents/build/gradio-builder.md`
is amended to match when `app.py` lands (its `demo.launch(show_api=False)` guidance was gradio-5-era).

**A8 (2026-07-05) — test suite extended with `test_lora_manager.py` / `test_controlnet_processor.py`.**
The LoRA and pose state machines are pure control flow (load-once, enable-before-set_adapters,
disable→re-enable, blank-skeleton rejection) and are unit-testable against a fake pipeline/detector
with zero model weights. Inference itself stays untested per the SPEC (nondeterministic, GPU-bound;
covered by phase acceptance + live demo).

**A9 (2026-07-05) — `accelerate` added to requirements.txt.**
Without it diffusers falls back to `low_cpu_mem_usage=False` and double-allocates on load — the fp32
UNet peaks near 2×3.4 GB transiently, untenable against this 8 GB M1's ~5.7 GB MPS working set.
accelerate is pure-Python, $0, and the Spaces standard. New runtime dependency, logged per the same
protocol A4 followed.

**A10 (2026-07-05) — `DPMSolverMultistepScheduler` @ 20 steps over checkpoint-default PNDM @ 50.**
DPM++ 2M reaches comparable SD 1.5 quality at 20–25 steps, a 2–2.5× wall-clock cut — the difference
between failing and meeting the SPEC's <5-min CPU-basic target, and it also shrinks the local smoke
run and ZeroGPU quota per generation. Set identically on every device (a device-conditional step
count would falsify the committed example images' metadata). The scheduler name is read back from
`type(pipe.scheduler).__name__` into the metadata panel, never hardcoded.

**A11 (2026-07-05) — Local inference OOMs on the 8 GB M1; D11 fallback invoked.**
Two D11 smoke runs (CPU fp32, cached weights) each completed all 10 denoise steps and then crashed
with SIGBUS during the VAE decode: the fp32 decode of a 512×512 image is a large activation spike on
top of the ~5.5 GB resident model, and the machine was already swap-bound (11 GB of 12 GB swap in use,
~16 MB RAM free). `enable_vae_tiling()`/`enable_vae_slicing()` were tried and reverted — tiling does
not engage at 512×512 (below its size threshold), so it is a no-op here and keeping it would be
speculative (simplicity-first). Per decision D11 ("if local generation OOMs, smoke-test directly on
Spaces hardware and log that decision; all real inference happens on Spaces"), local end-to-end
generation is not achievable on this hardware. Local evidence the pipeline is wired correctly: both
runs reached 10/10 denoise steps before the decode, and earlier runs surfaced load failures cleanly
as typed `PipelineLoadError`. Full image generation (Tab 1) and the ControlNet path (Tab 2) move to
Hugging Face Spaces verification in Phase 4 (fp16 ZeroGPU, or CPU-basic where the 16 GB RAM envelope
fits). The smoke script and `generate()` remain committed and unchanged.

**A12 (2026-07-05) — ZeroGPU `@spaces.GPU` wiring deferred to the ZeroGPU-grant moment.**
The first deploy targets CPU-basic, which runs the app as-is (16 GB RAM handles the fp32 decode that
OOMs an 8 GB machine) and needs no `@spaces.GPU`. The ZeroGPU path (fork semantics, building fp16 on
the parent, moving to CUDA inside the decorated child) cannot be verified locally or on CPU-basic, and
adding it blind risks breaking the CPU-basic deploy. The pure-inference seam (`_infer` / `_infer_posed`)
is already isolated, so the wiring is a localized addition made when ZeroGPU is granted and can be
verified against real Spaces logs. Alternatives rejected: shipping untested ZeroGPU plumbing now
(deploy-time surprises); building only for ZeroGPU and losing the CPU-basic fallback the SPEC requires (D1).
