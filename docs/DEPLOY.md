# Deploy to Hugging Face Spaces

VoxelCraft targets a ZeroGPU Space with a CPU-basic fallback. Budget is $0.00 — no paid hosting,
no paid inference, no secrets required for the base app.

## One-time setup (manual — Sheharyar)

1. **Create the Space.** huggingface.co → New Space → SDK **Gradio**, hardware **CPU basic** to
   start. Name it `voxelcraft`.
2. **Hardware.** CPU-basic (free) is the $0 default and runs the demo end to end (~4–7 min per
   image). **ZeroGPU is no longer free** — it now requires an HF PRO subscription ($9/mo) and
   returns `402 Payment Required` for a free account. The `@spaces.GPU` wiring is in place and
   env-gated (A14), so if the owner ever has PRO, switching hardware to ZeroGPU makes generation
   ~seconds with no code change; on the free budget the app stays on CPU-basic.
3. **Add the Space as a second git remote** and push:
   ```bash
   git remote add space https://huggingface.co/spaces/<user>/voxelcraft
   git push space main
   ```
   The `commit-msg` hook (single-author history, D10) applies to every remote, the Space included —
   never push with `--no-verify`.
4. **civitai LoRA weights (only if civitai-hosted LoRAs are registered).** Add a
   `CIVITAI_API_TOKEN` Space secret (free account) so the app can fetch those weights on first use;
   HF-hosted LoRAs need no token. Weights re-download on Space restart — that is accepted (a $0
   budget means no paid persistent storage).

## Preflight checklist (run before every push to the Space)

- [ ] `README.md` starts with valid HF YAML frontmatter (`sdk: gradio`, `sdk_version` matching the
      Space's supported Gradio, `app_file: app.py`, `license: mit`).
- [ ] `requirements.txt` resolves on the Spaces Linux image; dev tooling stays in
      `requirements-dev.txt`, out of the runtime image.
- [ ] The app runs with **no secrets** for the base (LoRA-free) path.
- [ ] No weights in git history: `git log --stat | grep -E '\.(safetensors|ckpt|bin)$'` is empty;
      weights download lazily on first request.
- [ ] `git log --format='%an %ae'` shows only Sheharyar Ahmed; no AI-attribution strings anywhere.
- [ ] `venv/bin/pytest -q` green, `venv/bin/mypy --strict src` clean.

## Post-deploy verification (closes the D11 fallback for local generation)

Local end-to-end generation is not verifiable on an 8 GB machine (decision A11); it is verified
here instead:

- [ ] Fresh browser, no auth: prompt → 512×512 image with a populated metadata panel in under
      60 s on ZeroGPU (under 5 min on CPU-basic).
- [ ] Pose tab: reference photo → skeleton preview → posed image; a no-person photo returns the
      friendly "no pose detected" error without spending a generation.
- [ ] Invalid inputs (over-long prompt, duplicate LoRA, out-of-range weight) each produce one
      `gr.Error` sentence, never a stack trace.
- [ ] `README.md` renders correctly on both GitHub and the Space.
- [ ] Commit 4–6 example outputs (small file size) generated on the Space into `examples/`.

## Notes on the ZeroGPU path

ZeroGPU is no longer free — it requires an HF PRO subscription ($9/mo) and returns
`402 Payment Required` for a free account (decision A15). On the $0 budget the app therefore runs
on CPU-basic. The `@spaces.GPU` wiring is already in place and gated on the `SPACES_ZERO_GPU`
env var (A14): off-ZeroGPU it is the identity and `spaces` is never imported, so local and
CPU-basic runs are unchanged. If the owner ever has PRO, switching the Space hardware to ZeroGPU
activates the GPU path (pipeline built fp16 on CPU in the parent, moved to CUDA inside the forked
`@spaces.GPU` child) and generation drops to seconds — no code change required.

The `sdk_version` in the README frontmatter must match a Gradio version the Space supports — if the
pinned version is newer than the Space runtime offers, pin down to the newest supported version and
log the change in the decision log rather than silently diverging.
