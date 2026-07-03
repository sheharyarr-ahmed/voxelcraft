# Anti-patterns — hard reject list

If a diff, commit, or doc matches anything below, stop and rewrite it. These are not style
preferences; each one breaks a locked decision or the anti-fabrication wall.

## 1. Training claims anywhere

Any comment, docstring, README line, or commit message implying we trained a LoRA, ran a GPU
training pipeline, or fine-tuned a model. **Why:** anti-fabrication wall — this project applies
pre-trained LoRAs; claiming training is a lie that survives in git history forever.
**Instead:** describe application: "applies voxel-style LoRA at weight 0.8", "blends two adapters
via `set_adapters`". Training methodology may be *documented* as future work, never as done.

## 2. AI-attribution strings in commits

"Co-Authored-By: Claude", "Generated with Claude Code", or any variant. **Why:** D10 —
single-author history, mechanically enforced by `.githooks/commit-msg`. Running
`git commit --no-verify` to get around the hook is itself on this list. **Instead:** write the
commit message plainly; if the hook rejects it, fix the message, never the hook.

## 3. Anything that costs money

Paid APIs, hosted inference behind a billing account, gated model access, paid CI minutes.
**Why:** budget is $0.00 total; also D3 — no Anthropic API at runtime. **Instead:** HF Spaces
ZeroGPU (CPU-basic fallback), public HF Hub downloads, free-tier everything. If a feature needs
a credit card, the feature is out.

## 4. Unverified LoRA or model reference

Loading any LoRA/model not present in `LORA_REGISTRY` (src/config.py) with a complete entry:
`url`, `author`, `license`, `commercial_use`. **Why:** D4 — license verification is manual and
done by Sheharyar; tooling never marks it verified. **Instead:** add the registry entry with a
`# LICENSE UNVERIFIED` marker and stop for manual review before wiring it into the UI.

## 5. Model load at import time

`from_pretrained`, `hf_hub_download`, or weight I/O at module scope. **Why:** D7 — imports must
be instant; Spaces cold-starts and pytest both die on eager loads. **Instead:** lazy singleton in
`src/pipelines/sd_pipeline.py` — first call to `get_pipeline()` loads, everything after reuses.

## 6. SDXL or any non-SD-1.5 base

Including "just trying" SDXL, SD 2.x, or SD3 checkpoints or their LoRAs. **Why:** D2 — SD 1.5
only; SDXL doubles memory, breaks the LoRA registry, and won't fit the ZeroGPU/CPU budget.
**Instead:** SD 1.5 base and SD-1.5-compatible LoRAs, full stop.

## 7. Raw user input reaching a pipeline

Passing Gradio component values straight into diffusers calls. **Why:** D6 — Pydantic v2 at
every input boundary. **Instead:** construct `GenerationRequest` / `ControlNetRequest`
(src/schemas.py) first; the pipeline layer accepts only validated schema instances.

## 8. Stack traces or pydantic internals in the UI

`ValidationError` dumps, tracebacks, or `repr()` of exceptions shown to the user. **Why:** it
leaks internals and reads as broken software. **Instead:** catch at the handler boundary and
raise `gr.Error("Prompt must be 3-500 characters.")` — one human sentence per failure mode.

## 9. Out-of-scope features

img2img, inpainting, auth/accounts, persistence, payments, public API endpoints, batch
generation, video, mobile clients. **Why:** locked scope — each one adds surface without adding
to what the artifact demonstrates. **Instead:** if it seems necessary, it goes in SPEC.md as a
proposed scope change for Sheharyar to approve, not in code.

## 10. Committing weights or caches

`*.safetensors`, `*.ckpt`, `*.bin`, HF cache dirs. **Why:** repo bloat, license redistribution
risk, and pushes to the HF Space remote will fail on file size. **Instead:** keep them in
`.gitignore`; weights are always fetched at runtime from the registry URL.

## 11. Speculative abstraction

Plugin systems, generic "model zoo" loaders, configuration frameworks, base classes with one
subclass. **Why:** this is a two-pipeline app with a dict registry; abstraction here is résumé
noise, not engineering. **Instead:** the simplest thing that ships — a plain dict in config.py,
two concrete pipeline modules, functions over class hierarchies until a second caller exists.
