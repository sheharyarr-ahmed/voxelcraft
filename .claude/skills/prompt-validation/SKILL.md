---
name: prompt-validation
description: Pydantic v2 validation for every user input boundary. Use when writing or reviewing src/schemas.py and src/utils.py.
---

## Pydantic v2 idioms only (D6)

`field_validator`, `model_validator(mode="after")`, `Field(ge=..., le=...)`, `ConfigDict`.
Never v1 `@validator`, `class Config`, or `.dict()` — mypy strict will not catch the
deprecation, so review for it explicitly. Importing `LORA_REGISTRY` from src/config.py
into schemas.py is fine: it is static data, no model weights load at import time (D7).

## GenerationRequest.prompt

Normalize inside one `field_validator("prompt")`, in this order:

```python
v = re.sub(r"[\x00-\x1f\x7f]", "", v)   # strip control chars (incl. \n, \t, \r)
v = re.sub(r"\s+", " ", v).strip()      # collapse whitespace runs
if not v:
    raise ValueError("Prompt is empty after removing control characters.")
```

Cap at the schema with `Field(max_length=500)` — a cheap character bound, not the real
limit. The definitive limit is CLIP's 77 tokens, and it cannot live in the schema: the
tokenizer ships with the lazily loaded pipeline (D7), so schemas.py must never import it.
Immediately before generation, run `pipe.tokenizer(prompt, truncation=False).input_ids`
and compare against `pipe.tokenizer.model_max_length` (77). If over, truncate to the
first 77 tokens, `gr.Warning(f"Prompt exceeds CLIP's 77-token limit; last {n} tokens
were ignored.")`, and record the truncation in the metadata panel. Never truncate
silently — diffusers does exactly that by default, which is the failure mode this
check exists to surface.

## LoRA selection

`loras: dict[str, float]` mapping registry key to weight, `default_factory=dict`.
One `field_validator` enforces all three rules with a human sentence per failure:

- at most 2 entries (`"Select at most 2 LoRAs."`),
- every key `in LORA_REGISTRY` (`f"Unknown LoRA: {key!r}."`),
- every weight a float in 0.0–1.5 (`f"Weight for {key!r} must be between 0.0 and 1.5."`).

lora_manager re-validates the same bounds before `set_adapters` — keep both; the
manager is reachable from tests and future entry points that bypass the schema.

## seed

`seed: int | None = Field(default=None, ge=-1, lt=2**32)`. `None` or `-1` means draw
`random.randint(0, 2**32 - 1)` at generation time. The metadata panel always shows the
resolved seed — reporting `-1` makes the run unreproducible, which defeats the panel.

## ControlNetRequest

`class ControlNetRequest(GenerationRequest)` — same prompt, LoRA, and seed rules by
inheritance, plus:

- `reference_image`: required, carried as `PIL.Image.Image` with
  `model_config = ConfigDict(arbitrary_types_allowed=True)`. The schema only asserts
  presence; byte-level checks happen in src/utils.py *before* the schema is built.
- `conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0)`.

## Upload validation (src/utils.py)

Cheapest check first; nothing decodes until the file has passed the size gate:

1. Extension allowlist `{".png", ".jpg", ".jpeg", ".webp"}` **and** MIME allowlist
   `{"image/png", "image/jpeg", "image/webp"}`. Neither alone — extensions lie and
   Gradio's reported MIME comes from the client.
2. `Path(path).stat().st_size > 5 * 1024 * 1024` → reject before `Image.open`. This is the
   decompression-bomb gate; it must precede any decoding.
3. `img = Image.open(path); img.verify()` — then **reopen** with `Image.open(path)`.
   `verify()` invalidates the file object; any later `load()` on the same handle
   raises or returns garbage. Reject if the decoded `img.format` is not in
   `{"PNG", "JPEG", "WEBP"}` — this catches renamed files the allowlists missed.
4. `img.convert("RGB")`, then resize long side to 512 with `Image.LANCZOS`. SD 1.5 and
   the OpenPose extractor both want 512-scale inputs; anything larger wastes VRAM on
   the ZeroGPU/CPU-basic budget.

Each rejection raises `ValueError` with one user-facing sentence naming the actual
limit ("Image is 7.2 MB; the limit is 5 MB.").

## Error mapping to the UI

```python
except ValidationError as e:
    raise gr.Error(e.errors()[0]["msg"].removeprefix("Value error, ")) from None
```

One sentence, the first error's `msg` — the validators above already write them for
humans. Strip the `"Value error, "` prefix: for validator-raised `ValueError`s,
pydantic v2 renders `msg` as `Value error, <sentence>`, so the raw `msg` would leak
pydantic's rendering into the UI. Never `str(e)`: pydantic's rendering leaks type
names, loc tuples, and docs URLs into the UI. Never let a traceback reach Gradio; log the full exception
server-side if it is needed for debugging.
