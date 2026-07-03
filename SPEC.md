# VoxelCraft SPEC

Stable Diffusion + LoRA application pipeline with ControlNet integration. Free-tier deployment on Hugging Face Spaces. Portfolio artifact for SheryLabs, sole author Sheharyar Ahmed.

## Goal

Ship a public, live, zero-cost demonstration of generative image AI engineering that closes the Stable Diffusion / LoRA / ControlNet skill gap on the Upwork profile with honest framing.

The app (Gradio, deployed on Hugging Face Spaces ZeroGPU) does three things:

1. **Text to image with LoRA stacking.** User enters a prompt, selects 1 to 2 pre-trained public LoRAs from a license-verified registry, adjusts weight sliders (0.0 to 1.5), and generates a 512x512 image via Stable Diffusion 1.5. A metadata panel shows LoRAs applied, weights, seed, scheduler, and inference time. This panel is the proof-of-understanding screenshot, the equivalent of ReelMind's Agent Trace UI.
2. **Pose-controlled generation.** User uploads a reference image. OpenPose extracts the skeleton (shown as intermediate output). ControlNet plus SD 1.5 plus an optional LoRA style generates a new image with the same pose in a different style.
3. **Training methodology reference.** A static tab renders the LoRA training walkthrough (dataset prep, captioning, rank/alpha, learning rate, steps) plus a link to a Colab reference notebook. Banner text states plainly: "Methodology reference. This app applies pre-trained LoRAs."

**The anti-fabrication wall governs everything.** Never claim: trained a custom LoRA, GPU training pipeline, custom model training experience, production AI video pipeline. Always claim: LoRA application pipeline with weight blending, ControlNet integration for pose-controlled generation, Diffusers production patterns, HF Spaces deployment workflow, documented training methodology. The README carries an explicit "What this is / What this is not" section.

**Budget: $0.00 total.** No Anthropic API in this project (Stable Diffusion via Hugging Face is the AI provider; Claude Code is the build tool, not a runtime dependency). No paid models, no paid hosting.

**Definition of done:** public repo, live Space generating images end to end from a fresh browser, clean single-author commit history, all model licenses documented, honest README, tests green, 4 to 6 example outputs committed, GitHub metadata polished. Sheharyar can answer from memory in a vetting call: why SD 1.5 over SDXL, what LoRA rank means, how ControlNet conditioning works, why fp16, what attention slicing does, and exactly where hands-on experience ends and documented methodology begins.

## Files

```
voxelcraft/
├── .claude/                          # Build scaffold, COMMITTED (credibility artifact)
│   ├── CLAUDE.md                     # Project instructions, anti-fabrication wall, phase plan
│   ├── agents/
│   │   ├── meta/voxelcraft-architect.md    # Opus. Design review, anti-pattern detection
│   │   ├── meta/decisions.md               # Decision log with rationale per entry
│   │   ├── build/diffusers-doctor.md       # Sonnet. diffusers code review, VRAM discipline
│   │   ├── build/gradio-builder.md         # Sonnet. Blocks UI, Spaces patterns
│   │   ├── build/hf-deployer.md            # Sonnet. Deployment checklist
│   │   └── docs/content-writer.md          # Haiku. README, architecture docs
│   ├── skills/
│   │   ├── lora-loading.md
│   │   ├── controlnet-preprocessing.md
│   │   ├── prompt-validation.md
│   │   └── memory-optimization.md
│   ├── rules/
│   │   ├── code-style.md             # black (line 100), isort, mypy strict
│   │   ├── anti-patterns.md          # reject list: training claims, paid LoRAs, attribution
│   │   └── license-compliance.md     # civitai + HF Hub license verification protocol
│   └── verify.sh                     # pytest -q, gates the Stop hook
├── .githooks/commit-msg              # rejects Claude attribution strings mechanically
├── app.py                            # HF Spaces entry point, thin wrapper importing src/
├── src/
│   ├── config.py                     # LORA_REGISTRY: url, author, license, commercial_use per entry
│   ├── schemas.py                    # Pydantic v2: GenerationRequest, ControlNetRequest
│   ├── pipelines/
│   │   ├── sd_pipeline.py            # lazy-loaded SD 1.5 singleton, fp16, attention slicing, device detection
│   │   ├── lora_manager.py           # load_lora, set_adapter_weights, stack validation
│   │   └── controlnet_processor.py   # OpenPose extraction, ControlNet pipeline composition
│   └── utils.py                      # image validation (format, size ≤5MB, resize 512), cleanup, metadata capture
├── tests/
│   ├── test_config.py                # every registry entry has license and commercial_use=True
│   ├── test_schemas.py               # prompt ≤77 tokens, weight bounds 0.0 to 1.5, control-char stripping
│   └── test_utils.py
├── notebooks/
│   └── lora_training_reference.ipynb # Colab T4 walkthrough, labeled "not executed by author on custom data"
├── docs/
│   ├── ARCHITECTURE.md               # Mermaid pipeline diagram plus decision summaries
│   ├── LORA_TRAINING.md              # theoretical methodology, honest framing
│   └── DEPLOY.md                     # HF Spaces deployment checklist
├── examples/                         # 4 to 6 committed output images, small file size
├── requirements.txt                  # gradio, diffusers, transformers, torch, pillow, controlnet-aux, pydantic, huggingface-hub
├── README.md                         # HF Spaces YAML frontmatter plus portfolio README combined
├── LICENSE                           # MIT
└── .gitignore                        # venv, *.safetensors, model cache, .claude/settings.local.json
```

## Decisions

Each entry is locked. Log in `.claude/agents/meta/decisions.md` with date on first commit.

1. **Hugging Face Spaces over Vercel/Railway/Modal/Replicate.** Vercel and Railway cannot run this app: SD needs GPU inference and multi-GB weights, which serverless Node platforms do not provide. Modal and Replicate are paid. HF Spaces ZeroGPU is the only zero-cost GPU host and is the category-native surface SD buyers expect. Fallback: CPU-basic Space (2 to 5 min generation, still demo-viable with queue messaging) if ZeroGPU is not granted.
2. **Stable Diffusion 1.5 over SDXL.** SD 1.5 plus LoRA plus ControlNet fits ZeroGPU VRAM (~6 to 8GB). SDXL needs 12GB+ and risks OOM. Stability beats output quality for a portfolio demo.
3. **No Anthropic API.** The AI provider is open-source Stable Diffusion. Wiring Claude into runtime would add cost and prove nothing this project is meant to prove. API credits stay reserved for agentic projects where Claude is the runtime brain.
4. **Pre-trained public LoRAs only, license-verified by hand.** Three LoRAs from civitai.com (one anime, one realistic, one painterly), each personally verified by Sheharyar for commercial-use permission on the model card. This verification is a manual step Claude Code must pause for; it is a vetting-call question and the answer must be Sheharyar's own.
5. **Gradio Blocks over Streamlit.** Native HF Spaces SDK, built-in request queue, zero-friction deploy.
6. **Pydantic v2 at every input boundary.** Prompt length, weight bounds, upload validation. Matches the Zod-at-every-boundary discipline from ReelMind.
7. **Lazy model loading.** Models load on first generation, not app start. Cold-start discipline for Spaces.
8. **fp16 plus attention slicing plus device detection.** VRAM discipline, graceful CPU fallback.
9. **Committed .claude/ scaffold.** The build scaffold is itself a portfolio credibility artifact, repeating the ReelMind pattern.
10. **Single-author git history, mechanically enforced.** `.githooks/commit-msg` rejects "Co-Authored-By: Claude", "Generated with Claude Code", and any AI-attribution string. `git config core.hooksPath .githooks`. Author locked to Sheharyar Ahmed. The HF Space is a second git remote; the same rules apply to every push there.
11. **Local smoke test is correctness-only.** M1 8GB CPU generation takes minutes per image; acceptable once. If local generation OOMs, smoke-test directly on Spaces hardware and log that decision. All real inference happens on Spaces.
12. **Manual UI steps are explicit pause points.** Claude Code must stop and ask Sheharyar to perform: GitHub repo settings confirmation (Phase 0), civitai LoRA license verification (Phase 1), HF account plus Space creation plus ZeroGPU hardware request (Phase 4), GitHub metadata polish: About description, topics (stable-diffusion, lora, controlnet, diffusers, gradio, huggingface, generative-ai, python), social preview image via Claude Design on the true-black/mint brand system, website field pointing to the live Space (Phase 4).

**Build phases, each gated by an acceptance check and at least one commit:**

- **Phase 0, scaffold (1 to 2h):** .claude/ populated, hook installed and proven (test commit containing an attribution string must be rejected), venv plus requirements, empty test suite green, verify.sh wired.
- **Phase 1, local inference smoke test (2 to 3h):** sd_pipeline.py generates one 512x512 image locally, LoRA registry populated with 3 verified entries, registry tests green.
- **Phase 2, Gradio app (3 to 4h):** three-tab Blocks UI, lazy loading, Pydantic boundaries, user-friendly error states, metadata panel. Tab 1 works end to end locally.
- **Phase 3, ControlNet (2 to 3h):** reference photo in, skeleton preview, posed stylized image out, locally.
- **Phase 4, deploy plus docs (3 to 4h):** live Space, README with YAML frontmatter, ARCHITECTURE.md with Mermaid diagram, LORA_TRAINING.md, Colab notebook, examples committed, GitHub metadata polished.
- **Phase 5, portfolio conversion (separate session, post-ship):** 60 to 90s demo recording, Upwork portfolio entry #7, LinkedIn Featured / Projects / announcement post per the ReelMind sequence.

## Out of scope

Documented in the README as deliberate skips, never promised, never implied:

- Custom LoRA training execution (no GPU; the Colab notebook is a reference, not a claim)
- SDXL
- img2img and inpainting
- User accounts, auth, or result persistence
- Payment or monetization layer
- API endpoints for programmatic access
- Batch generation
- Video generation of any kind
- Anthropic API integration
- Android or mobile clients

## Verification

**Automated gate (`.claude/verify.sh`, runs on the Stop hook):** `pytest -q`. Minimum suite: registry integrity (every LoRA entry has url, author, license, commercial_use=True), prompt validation (token limit, control-character stripping, weight bounds), upload validation. Inference itself is not unit-tested (nondeterministic, GPU-bound); it is covered by phase acceptance checks and the live demo.

**Per-phase acceptance:**

- Phase 0: `pytest -q` green; first commit pushed; hook demonstrably rejects an attribution test-commit; `git log --format='%an %ae'` shows only Sheharyar Ahmed.
- Phase 1: one test image generated; `pytest tests/test_config.py` green; three civitai model cards personally verified and documented in config.py.
- Phase 2: full Tab 1 flow (prompt to image to metadata panel) works locally; invalid inputs produce friendly errors, not stack traces.
- Phase 3: reference photo produces skeleton preview and a posed, stylized output locally.
- Phase 4: the live Space generates an image end to end from a fresh browser with no auth; README renders correctly on both GitHub and HF; contributor graph on GitHub shows a single author.

**End-to-end check (the spec's one-line proof):** a stranger opens the Space URL, types a prompt, picks a LoRA, and receives an image with a populated metadata panel in under 60 seconds on ZeroGPU (or under 5 minutes on CPU-basic fallback), then opens the repo and finds every model's license documented and an honest "What this is / What this is not" section in the README.
