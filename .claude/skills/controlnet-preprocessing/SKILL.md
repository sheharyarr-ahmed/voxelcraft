---
name: controlnet-preprocessing
description: OpenPose skeleton extraction and ControlNet pipeline composition for pose-controlled generation. Use when writing or reviewing src/pipelines/controlnet_processor.py.
---

## Skeleton extraction

```python
from controlnet_aux import OpenposeDetector

detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
skeleton = detector(reference_image)  # PIL.Image, black background + colored limbs
```

- The detector call returns the skeleton map directly as a PIL image. Do not
  post-process it — pass it to the pipeline as-is.
- SPEC requires the skeleton be shown in the UI as the intermediate output. Return
  it from the processor alongside the final image so Tab 2 can render both; do not
  keep it internal.
- The detector downloads its body-estimation weights on first `from_pretrained`;
  they land in the HF cache, not the repo.

## ControlNet weights and composition

```python
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose",
    torch_dtype=base_pipe.dtype,  # must match the base pipeline (fp16 on GPU, D8)
)
cn_pipe = StableDiffusionControlNetPipeline(**base_pipe.components, controlnet=controlnet)
```

- Build the ControlNet pipeline from the existing text2img singleton's `.components`
  — the UNet, VAE, text encoder, and tokenizer are shared by reference, so VRAM does
  not double and there is no second SD 1.5 download. Never call
  `StableDiffusionControlNetPipeline.from_pretrained` for the base weights.
- A dtype mismatch between controlnet and base pipeline fails mid-denoise with an
  opaque "expected Half but found Float" — matching `base_pipe.dtype` at load time
  is the fix, not casting later.
- LoRAs loaded on the shared UNet stay active in the composed pipeline. An optional
  style adapter stacks on top of the pose condition via the lora-loading skill; no
  extra wiring in this module.

## Laziness and memory

- Neither the detector nor the ControlNet loads at import or app start (D7). Both
  load on the first Tab-2 generation and are cached as module-level singletons after,
  same pattern as sd_pipeline.py.
- ControlNet adds roughly 0.7 GB in fp16 on top of the base pipeline. On the
  CPU-basic Spaces fallback that headroom matters — lazy loading is what keeps Tab 1
  usable when nobody touches Tab 2.

## Input handling

- Validate the reference upload through src/utils.py **before** anything touches the
  detector: format allowlist, <=5MB, resize to 512. `ControlNetRequest`
  (src/schemas.py) is the boundary — corrupt or oversized uploads are rejected there
  with a field-level error, never mid-pipeline (D6).
- `controlnet_conditioning_scale` defaults to 1.0. It is a float on
  `ControlNetRequest`, exposed as a slider in Tab 2; 1.0 means "follow the pose",
  lower values loosen adherence.

## Failure modes

- **No person in the reference photo:** OpenPose does not raise — it returns an
  all-black skeleton. Check for this (e.g. `skeleton.getbbox() is None` or a
  max-pixel-value test) and return a friendly "no pose detected in this image" error
  *before* spending 20+ seconds of generation on a blank condition.
- **Bad uploads:** already dead at the schema boundary. If a decode error ever
  surfaces inside this module, the utils validation has a gap — fix it there, do not
  add a try/except here.
