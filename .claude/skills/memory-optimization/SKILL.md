---
name: memory-optimization
description: VRAM and RAM discipline for SD 1.5 on ZeroGPU and an 8 GB M1. Use when writing pipeline code or debugging out-of-memory failures.
---

# Memory Optimization

## Budget to design against

| Component | Footprint |
|---|---|
| SD 1.5 fp16 resident (UNet + VAE + text encoder) | ~2 GB |
| ControlNet + OpenPose annotator, fp16 | ~0.7-1 GB |
| Each LoRA adapter | tens of MB |
| SPEC envelope on ZeroGPU | 6-8 GB |
| Local M1 8 GB, CPU fp32 | ~4-5 GB RAM |

On the M1 a 512x512 image takes minutes on CPU. That is acceptable exactly once — the D11
smoke test — not a dev loop. Iterate on the Space, not locally.

## Always

- Call `pipe.enable_attention_slicing()` immediately after pipeline construction. It costs a few
  percent of speed and buys the headroom that keeps ZeroGPU inside the envelope.
- Dtype follows device: `torch.float16` on CUDA, `torch.float32` on CPU and MPS (D8). fp16 on
  MPS produces black or NaN images on SD 1.5 — a known MPS numerical issue — so MPS is fp32,
  never attempt fp16 there.
- Device detection order is cuda, then mps, then cpu (D8). One function in `sd_pipeline.py` owns
  this; nothing else calls `torch.cuda.is_available()` directly.

## Singleton discipline (D7)

- The SD 1.5 pipeline is a lazy singleton behind a `threading.Lock`. Nothing loads at import
  time — `import app` must complete in milliseconds with no weights touched.
- Never hold two full pipelines. Build the ControlNet pipeline by composing from the base
  pipeline's components (`StableDiffusionControlNetPipeline(**pipe.components, controlnet=cn)`
  or `.from_pipe()`), so the UNet, VAE, and text encoder are shared, not duplicated. Two
  resident SD 1.5 copies is ~4 GB gone for nothing.

## accelerate-gated flags

`low_cpu_mem_usage=True` and friends require `accelerate`, which is not currently a dependency.
Do not silently add it to make a flag work — adding a dependency is a decision-log entry first.
If a flag errors with "requires accelerate", that is your cue to stop and log the decision, not
to pip install and move on.

## Between generations

```python
gc.collect()
if device == "cuda":
    torch.cuda.empty_cache()
elif device == "mps":
    torch.mps.empty_cache()
```

Also unload LoRA adapters the user deselected (`pipe.delete_adapters(name)` /
`unload_lora_weights()` per `lora_manager.py`). Adapters that survive a generation they were
not selected for are the most common slow leak in this app.

## Do not add

- `enable_model_cpu_offload()` / `enable_sequential_cpu_offload()` — only if an observed OOM on
  the actual Space demands it, and then log the decision. They trade large latency for memory
  we should not need at 512x512 with slicing on.
- xformers — torch >= 2.0 SDPA already provides memory-efficient attention; xformers is a
  build-fragile dependency for no gain here.
- Batch generation — out of scope, and the fastest way to blow the envelope.

## OOM diagnosis order

1. **Resident pipeline count** — did something construct a second full pipeline instead of
   composing from `pipe.components`?
2. **Dtype actually in use** — print `pipe.unet.dtype`; a silent fp32 load doubles the footprint.
3. **Resolution creep** — anything past 512x512 scales attention memory quadratically; the
   Pydantic schema should have rejected it.
4. **Leaked adapters** — list `pipe.get_active_adapters()`; failed unloads leave LoRAs attached
   across generations.

Check them in that order. Each earlier cause is both more likely and cheaper to confirm.
