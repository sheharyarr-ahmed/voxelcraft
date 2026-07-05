# LoRA Training Methodology

> **Methodology reference. This app applies pre-trained LoRAs.**

This document is a technical reference on Stable Diffusion 1.5 LoRA training methodology. No
custom LoRA was trained by the author on custom data; VoxelCraft applies pre-trained,
license-verified LoRAs from a public registry with runtime weight blending via diffusers.
The methodology below is documented reference knowledge, included so the design decisions the
app *does* make (rank-agnostic loading, weight ranges, stacking) are grounded in how the
artifacts they consume are produced.

## Dataset preparation

LoRA training typically starts with a curated set of 50–500 images at 512×512 (SD 1.5's native
resolution). The set is balanced across the concept being taught: uniform lighting, pose
variety, and consistent framing reduce mode collapse and overfitting. A smaller, higher-quality
set (100–200 images) often outperforms a larger, noisier one.

SD 1.5 expects square inputs, so aspect-ratio correction — center-crop or pad — matters. A
held-out validation set (5–10%) is used to detect overfitting during training rather than after.

## Captioning

Every training image carries a text caption. Captions teach the model *when* to apply the
learned style through the cross-attention mechanism and anchor the concept to natural language.

Captions usually include a rare "trigger word" (for example `zxc painting`) that keeps the
learned concept from leaking into unrelated generations, paired with descriptive attributes:
`zxc painting, oil on canvas, impressionist`. Strategies range from automated (BLIP-generated
captions — fast, less precise), to manual (most reliable, labor-intensive), to hybrid (auto-
generate, then hand-edit trigger words and domain detail). More specific captions converge
faster and reduce catastrophic forgetting of the base model's general knowledge.

## Rank and alpha

LoRA rank is the dimensionality of the weight-update matrices — it sets how much capacity the
adapter has. Low-rank decomposition expresses a weight delta as the product of two smaller
matrices: a layer weight of shape `(out, in)` is approximated by `(out, r) × (r, in)`, cutting
parameters from `O(out × in)` to `O((out + in) × r)` and yielding small, portable adapters.

Typical rank values:

- **4–8** — minimal capacity; style transfers and simple concept binding; 2–5 MB files.
- **16–32** — the practical sweet spot for most single-concept fine-tuning; 8–25 MB files.
- **64–128** — high capacity for complex concepts or large datasets; 50–100 MB; higher
  overfitting risk and slower load.

Alpha scales the update magnitude at inference: the effective contribution is roughly
`weight × alpha / rank`. Setting `alpha ≈ rank` is empirically stable; a rank-16 adapter with
alpha 16 is more predictable than alpha 1 or 50. The tradeoff is straightforward — higher rank
buys expressivity at the cost of file size, overfitting risk, and inference speed.

## Learning rate and optimizer

LoRA training typically uses AdamW at learning rates of **1e-4 to 5e-4**, tuned against batch
size, step count, and dataset size. A common recipe starts at 2e-4, applies a warmup over the
first 10–20% of steps to stabilize early gradients, and decays (cosine or linear) toward the end
to refine fine detail. 8-bit AdamW (bitsandbytes) trims optimizer memory so larger batches fit on
limited VRAM.

## Steps and epochs

Duration balances convergence against overfitting. For a ~200-image set, 500–1000 steps at batch
size 4 is roughly 2–5 epochs; pushing to 2000–5000 steps on a small set risks overfitting, where
validation loss rises while training loss keeps falling. Larger sets (500+ images) tolerate more
steps (5000–10000) before stabilizing. Telltale overfitting: a widening train/validation gap, or
outputs that reproduce training images instead of generalizing the concept. Checkpointing every
100–200 steps allows picking the best snapshot before quality degrades.

## How VoxelCraft applies these artifacts

The app does not execute the training above; it consumes its output — pre-trained `.safetensors`
adapters. At runtime the application layer:

1. **Loads adapters via diffusers** — `load_lora_weights(repo_or_path, weight_name, adapter_name)`
   deserializes the trained rank matrices and attaches them to the SD 1.5 UNet and text encoder.
2. **Stacks up to two adapters** — `enable_lora()` then `set_adapters(names, adapter_weights)`
   blends their effects, with per-adapter weights from 0.0 (no effect) to 1.5 (amplified).
3. **Captures metadata** — the panel reports the applied LoRAs and weights, seed, scheduler
   (`DPMSolverMultistepScheduler`), steps (default 20, range 1–50), and guidance scale.

Every registry LoRA is pre-trained externally (civitai, Hugging Face Hub) and verified for
commercial-use license before inclusion. `src/config.py` records the model URL, author, license,
and weight source (`repo_id` + `weight_name` for the Hub, `download_url` for civitai).

A companion Colab T4 training notebook (Phase 4) walks through the standard approach on a public
dataset; it was not executed by the author on custom data.

## Where hands-on experience ends

The author's demonstrated, hands-on work is LoRA *application*:

- Loading and stacking pre-trained adapters via diffusers `load_lora_weights()` / `set_adapters()`.
- Runtime weight blending with validation bounds 0.0–1.5 (`src/pipelines/lora_manager.py`).
- ControlNet integration for pose-controlled generation and reference-image preprocessing.
- VRAM discipline: dtype selection, attention slicing, device detection, CPU fallback
  (`src/pipelines/sd_pipeline.py`).
- Gradio Blocks UI, Pydantic v2 input validation, and HF Spaces deployment patterns.

The methodology sections above — dataset curation, captioning, rank/alpha tuning, learning-rate
selection, convergence monitoring — are reference knowledge, not executed training. That
distinction is deliberate and load-bearing: the shipped, verifiable capability is applying
pre-trained LoRAs with honest metadata, not training them.
