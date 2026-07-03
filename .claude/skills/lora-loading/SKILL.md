---
name: lora-loading
description: Load, stack, and weight pre-trained LoRAs on SD 1.5 with diffusers. Use when writing or reviewing src/pipelines/lora_manager.py or the LoRA registry in src/config.py.
---

## Registry contract (src/config.py)

`LORA_REGISTRY` is a dict keyed by a short slug (e.g. `"voxel"`, `"pixel_art"`). Every entry carries:

- `url` — canonical source page (HF model page or civitai model page).
- `author` — creator name as published at the source.
- `license` — exact license string from the source page.
- `commercial_use` — bool. **True only after Sheharyar has manually read the license (D4).**
  Tooling, scripts, and Claude never flip this to True. Default new entries to False.
- Weight location, one of:
  - HF-hosted: `repo_id` + `weight_name` (the `.safetensors` filename inside the repo), or
  - civitai: absolute path under the gitignored cache dir (see "civitai downloads").
- `base_model` — must be `"sd-1.5"`. This field gates loading (D2).

## Loading

```python
pipe.load_lora_weights(repo_id_or_dir, weight_name=weight_name, adapter_name=registry_key)
```

- `adapter_name` is always the registry key — it is the join key between the registry,
  the manager's state, and the metadata panel.
- Load each adapter at most once per process. The manager keeps a `set[str]` of loaded
  adapter names; skip `load_lora_weights` on a hit. Re-loading the same adapter_name
  raises in recent diffusers and silently duplicates memory in older ones.
- Loading happens lazily inside the manager, never at import time (D7).

## Stacking and weighting

```python
pipe.set_adapters(["voxel", "pixel_art"], adapter_weights=[1.0, 0.6])
```

- 1–2 adapters per generation, each weight in 0.0–1.5.
- `GenerationRequest` (src/schemas.py) already enforces these bounds; the manager
  re-validates before calling `set_adapters` — defense in depth, since the manager
  is also callable from tests and future entry points that bypass the schema.
- `set_adapters` is cheap; call it per request. Adapters stay resident, only the
  active set and weights change.

## Disable and cleanup

- LoRA-free request: `pipe.disable_lora()`. Do not unload — adapters stay resident.
  But `disable_lora()` sets `_disable_adapters=True` on each PEFT `BaseTunerLayer`, and
  `set_adapters()` never clears it — so call `pipe.enable_lora()` before `set_adapters`
  on the next LoRA request, or it silently runs with no LoRA effect.
- Memory pressure (CPU-basic fallback on Spaces): `pipe.unload_lora_weights()` frees
  all adapters; clear the manager's loaded-name set in the same code path or the
  skip-if-loaded check goes stale.
- **Never `fuse_lora()`.** Fusing bakes weights into the UNet/text-encoder tensors;
  per-request re-weighting then requires unfuse + re-fuse, which is slow and drifts
  numerically. Runtime-switchable adapters are the whole point of this pipeline.

## Compatibility guard

Check `entry.base_model == "sd-1.5"` before `load_lora_weights` and raise a typed
error naming the registry key if it fails. An SDXL LoRA does not fail cleanly — it
surfaces as a wall of state-dict key mismatches (`unet.up_blocks...` shape errors)
deep inside PEFT, which is useless to a user. SD 1.5 only, no SDXL, ever (D2).

## civitai downloads

- Download once into the gitignored cache dir; filename derived from the registry key.
- Record the exact source URL in the registry entry — civitai pages disappear, and
  license verification (D4) is meaningless without a pinned source.
- Weights are never committed. If a `.safetensors` shows up in `git status`, fix
  `.gitignore` before anything else.

## Metadata

Every generation records `adapter_names: list[str]` and `adapter_weights: list[float]`
(empty lists for LoRA-free runs) via src/utils.py, alongside seed, scheduler, and
inference time. The UI metadata panel renders these verbatim — the registry key is
the display name, so keep keys human-readable.
