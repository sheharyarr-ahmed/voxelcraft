"""Upload validation, seed resolution, token-overflow reporting, and metadata capture.

Imports PIL and the standard library only — never torch — so the unit suite imports it in
milliseconds and the Stop-hook gate stays fast.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from src import config
from src.exceptions import UploadValidationError


def validate_upload(path: Path) -> Image.Image:
    """Validate an uploaded image (suffix, format, size, decodability) and return it RGB, <=512.

    Order matters: the cheap suffix and size checks run before any decode, so a hostile or
    oversized file is rejected without spending memory on it.
    """
    suffix = path.suffix.lower()
    if suffix not in config.ALLOWED_UPLOAD_SUFFIXES:
        raise UploadValidationError(
            f"Unsupported file type {suffix or '(none)'}. Use PNG, JPEG, or WebP."
        )

    size = path.stat().st_size
    if size > config.MAX_UPLOAD_BYTES:
        size_mb = size / (1024 * 1024)
        limit_mb = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise UploadValidationError(f"Image is {size_mb:.1f} MB; the limit is {limit_mb} MB.")

    # verify() proves the bytes decode but leaves the file object unusable, so reopen after.
    try:
        with Image.open(path) as probe:
            probe.verify()
        image = Image.open(path)
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise UploadValidationError("This file is not a readable image.") from exc

    if image.format not in config.ALLOWED_IMAGE_FORMATS:
        raise UploadValidationError(
            f"Image content is {image.format or 'unrecognized'}, not PNG, JPEG, or WebP."
        )

    oriented = ImageOps.exif_transpose(image) or image  # honor phone-camera orientation
    result = oriented.convert("RGB")
    result.thumbnail((config.IMAGE_SIZE, config.IMAGE_SIZE), Image.Resampling.LANCZOS)
    return result


def resolve_seed(seed: int | None) -> int:
    """Return the seed, drawing a random one when it is None or the -1 sentinel."""
    if seed is None or seed == -1:
        return random.randint(0, config.SEED_MAX)
    return seed


def clip_token_overflow(tokenizer: Any, prompt: str) -> int:
    """Return how many prompt tokens exceed CLIP's 77-token window (0 if it fits).

    Diffusers truncates internally; this exists so the truncation can be reported to the
    user and recorded in metadata rather than happening silently.
    """
    token_count = len(tokenizer(prompt).input_ids)
    limit: int = tokenizer.model_max_length
    return max(0, token_count - limit)


def capture_metadata(
    *,
    adapter_names: list[str],
    adapter_weights: list[float],
    seed: int,
    scheduler: str,
    steps: int,
    guidance_scale: float,
    inference_seconds: float,
    device: str,
    dtype: str,
    truncated_tokens: int = 0,
    safety_checker: bool = True,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the metadata-panel dict shown in the UI after a generation.

    ``adapter_names`` / ``adapter_weights`` are the applied selection returned by
    ``lora_manager.apply_loras``, which guarantees the pipeline's active adapter set matches
    them (calling ``enable_lora()`` then ``set_adapters()``, or raising) — so the panel
    reflects what was actually applied, not merely what was requested.
    """
    metadata: dict[str, Any] = {
        "loras_applied": adapter_names,
        "lora_weights": adapter_weights,
        "seed": seed,
        "scheduler": scheduler,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "inference_seconds": round(inference_seconds, 2),
        "device": device,
        "dtype": dtype,
        "truncated_tokens": truncated_tokens,
        "safety_checker": "enabled" if safety_checker else "disabled",
    }
    if extra:
        metadata.update(extra)
    return metadata
