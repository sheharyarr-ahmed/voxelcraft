---
title: VoxelCraft
emoji: 🎨
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.19.0
app_file: app.py
pinned: false
license: mit
---

# VoxelCraft

A Stable Diffusion 1.5 **LoRA application** pipeline with ControlNet pose control, built on
Diffusers and served with Gradio on Hugging Face Spaces. Enter a prompt, stack up to two
pre-trained LoRAs with per-adapter weight control, or drive generation from a reference pose —
every result comes with a metadata panel showing exactly what was applied.

**Live demo:** _set after deploy_ · **Source:** https://github.com/sheharyarr-ahmed/voxelcraft

## What this is / What this is not

This section is deliberate and load-bearing. It states exactly where hands-on work ends.

**What this is:**

- A **LoRA application** pipeline — loading pre-trained `.safetensors` adapters via Diffusers
  `load_lora_weights`, stacking up to two with `set_adapters`, and blending them at per-adapter
  weights from 0.0 to 1.5.
- **ControlNet integration** for pose-controlled generation — OpenPose skeleton extraction from
  a reference photo, then pose-conditioned generation.
- **Diffusers production patterns** — lazy model loading, fp16/fp32 device policy, attention
  slicing, device detection, and typed error handling at every boundary.
- A **Hugging Face Spaces deployment** workflow, from Gradio Blocks UI to ZeroGPU.
- A **documented** LoRA training methodology reference (`docs/LORA_TRAINING.md`).

**What this is not:**

- **Not custom LoRA training.** No model was trained by the author on custom data; there is no
  GPU training pipeline here. The Colab notebook under `notebooks/` is a methodology *reference*,
  explicitly not executed by the author on custom data.
- **Not SDXL** — SD 1.5 only, chosen to fit the free ZeroGPU / CPU-basic memory envelope.
- **Not** img2img, inpainting, video, batch generation, user accounts, result persistence, a
  programmatic API, or a mobile client. These are deliberate scope exclusions, not omissions.

## Features

1. **Text to image with LoRA stacking.** Prompt in, select 1–2 license-verified LoRAs from the
   registry, adjust weight sliders, generate a 512×512 image. A metadata panel reports the LoRAs
   applied, weights, seed, scheduler, steps, and inference time. (The curated LoRAs are pending
   manual license verification; until then the tab runs on base SD 1.5 with an empty-state notice.)
2. **Pose-controlled generation.** Upload a reference photo; OpenPose extracts the skeleton
   (shown as intermediate output); ControlNet + SD 1.5 (+ an optional LoRA style) generate a new
   image matching the pose.
3. **Training methodology reference.** A static tab rendering the LoRA training walkthrough, with
   the banner: _"Methodology reference. This app applies pre-trained LoRAs."_

## Architecture

Validated Pydantic request → lazy SD 1.5 singleton → LoRA manager → seeded DPM++ 2M denoise →
VAE decode → safety checker → image + metadata. The ControlNet pipeline is composed from the base
pipeline's components (`from_pipe`) so the UNet, VAE, and text encoder are shared, not duplicated.
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full diagram and the decision rationale.

## Model licenses

Every model artifact loaded at runtime is documented. Base and conditioning models:

| Artifact | Model ID | License |
| --- | --- | --- |
| SD 1.5 base weights | `stable-diffusion-v1-5/stable-diffusion-v1-5` | CreativeML Open RAIL-M (use-based restrictions apply) |
| ControlNet OpenPose | `lllyasviel/control_v11p_sd15_openpose` | Open RAIL |
| OpenPose annotator | `lllyasviel/Annotators` | Unspecified — verify per the ambiguity rule before use |

LoRA adapters are pre-trained, third-party models. Each entry is added to the registry
(`src/config.py`) only after its commercial-use license is verified by hand; the registry records
the model-card URL, author, exact license/permission wording, and date checked. The three curated
LoRAs (anime / realistic / painterly) are pending that final license verification step.

## Run locally

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
# Pre-download the SD 1.5 weights once (first generation is a multi-GB download):
venv/bin/python -m src.pipelines.sd_pipeline
venv/bin/python app.py
```

Note on memory: end-to-end generation needs roughly 6 GB of free RAM (or a GPU). On an 8 GB
machine the CPU fp32 decode can exhaust memory; the hosted Space is the intended way to generate
images. Development, tests, and type checks run comfortably on 8 GB.

Development checks:

```bash
venv/bin/pip install -r requirements-dev.txt
venv/bin/pytest -q
venv/bin/mypy --strict src
venv/bin/black --check --line-length 100 src tests app.py
```

## Tech stack

Python 3.12 · Diffusers · Transformers · PyTorch · PEFT · controlnet-aux · Pydantic v2 · Gradio ·
Hugging Face Spaces (free CPU-basic; `@spaces.GPU`-wired for ZeroGPU should PRO ever be added).
Single-author history is enforced mechanically by a `commit-msg` git hook.

## License

MIT — see [`LICENSE`](LICENSE). This covers the VoxelCraft application code only; the models it
loads carry their own licenses, documented above.
