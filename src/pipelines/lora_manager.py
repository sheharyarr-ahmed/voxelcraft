"""Load, stack, and weight pre-trained LoRA adapters on an SD 1.5 pipeline.

Encodes the two diffusers behaviours verified against the installed venv:

* an adapter is loaded at most once per process (repeat requests only re-weight), and
* ``enable_lora()`` must precede ``set_adapters()`` after any ``disable_lora()`` — otherwise
  the generation silently runs with no LoRA effect while the metadata panel still lists it.

The pipeline is passed in and never imported here, so the whole state machine is exercised
in tests with a fake object and zero model weights.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlretrieve

from src import config
from src.exceptions import LoraLoadError

logger = logging.getLogger(__name__)


def _loaded_for(pipe: Any) -> set[str]:
    """Return the set of adapter names loaded onto this specific pipe.

    Stored on the pipe object rather than a module global so it is scoped to the pipe's
    lifetime: a rebuilt pipeline starts with a fresh set instead of inheriting stale names
    (which would make set_adapters raise on a name the new pipe never loaded), and tests
    isolate naturally per fake instance. The registry is capped at a handful of entries
    (SPEC D4), so the set stays small — no eviction needed.
    """
    loaded: set[str] | None = getattr(pipe, "_vc_loaded_adapters", None)
    if loaded is None:
        loaded = set()
        pipe._vc_loaded_adapters = loaded
    return loaded


def apply_loras(pipe: Any, loras: dict[str, float]) -> tuple[list[str], list[float]]:
    """Load, enable, and weight the requested adapters; disable all LoRAs when none requested.

    Returns the applied adapter names and weights for the metadata panel.
    """
    loaded = _loaded_for(pipe)
    if not loras:
        if loaded:
            pipe.disable_lora()
        return [], []

    _validate_selection(loras)
    for key in loras:
        if key not in loaded:
            _load_adapter(pipe, key, loaded)

    names = list(loras)
    weights = [loras[key] for key in names]
    pipe.enable_lora()  # clears any prior disable_lora(); set_adapters() alone does not
    pipe.set_adapters(names, adapter_weights=weights)
    return names, weights


def unload_all(pipe: Any) -> None:
    """Unload every adapter and clear the loaded-name set (manual memory-pressure valve)."""
    loaded = _loaded_for(pipe)
    if loaded:
        pipe.unload_lora_weights()
        loaded.clear()


def _validate_selection(loras: dict[str, float]) -> None:
    # Deliberate second boundary. The count/registry/weight rules mirror the Pydantic schema
    # (D6), but the lora-loading skill calls for defense in depth here too: apply_loras is a
    # reusable function that must not trust its caller, and the SD-1.5-base check below has no
    # schema equivalent. All failures raise LoraLoadError so callers see one consistent type.
    if len(loras) > config.MAX_LORAS:
        raise LoraLoadError(f"Select at most {config.MAX_LORAS} LoRAs.")
    for key, weight in loras.items():
        entry = config.LORA_REGISTRY.get(key)
        if entry is None:
            raise LoraLoadError(f"Unknown LoRA: {key!r}.")
        if entry.base_model != config.SD15_BASE:
            raise LoraLoadError(f"LoRA {key!r} is not an SD 1.5 adapter.")
        if not config.LORA_WEIGHT_MIN <= weight <= config.LORA_WEIGHT_MAX:
            raise LoraLoadError(
                f"Weight for {key!r} must be between "
                f"{config.LORA_WEIGHT_MIN} and {config.LORA_WEIGHT_MAX}."
            )


def _load_adapter(pipe: Any, key: str, loaded: set[str]) -> None:
    entry = config.LORA_REGISTRY[key]
    try:
        if entry.repo_id is not None:
            pipe.load_lora_weights(entry.repo_id, weight_name=entry.weight_name, adapter_name=key)
        else:
            weights_path = _ensure_local_weights(key, entry)
            pipe.load_lora_weights(
                str(weights_path.parent), weight_name=weights_path.name, adapter_name=key
            )
    except LoraLoadError:
        raise  # keep the specific message from _ensure_local_weights
    except Exception as exc:  # diffusers raises a variety of load-time errors; normalize them
        raise LoraLoadError(f"Could not load LoRA {key!r}.") from exc
    loaded.add(key)


def _ensure_local_weights(key: str, entry: config.LoraEntry) -> Path:
    """Return the local path to a civitai LoRA, fetching it once and verifying its checksum.

    The primary local flow is a manual file drop at this path (civitai downloads generally
    need an account token); the fetch is a best-effort fallback. The checksum guards against
    a truncated download or a swapped-out file.
    """
    config.LORA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    target = config.LORA_CACHE_DIR / f"{key}.safetensors"
    if not target.exists():
        if entry.download_url is None:
            raise LoraLoadError(f"LoRA {key!r} has no local weights and no download source.")
        logger.info("Fetching LoRA %r from %s", key, entry.download_url)
        try:
            urlretrieve(entry.download_url, target)  # noqa: S310 - registry URLs are vetted
        except (URLError, OSError) as exc:
            raise LoraLoadError(f"Could not download LoRA {key!r}.") from exc
    if entry.sha256 and _sha256(target) != entry.sha256:
        target.unlink(missing_ok=True)
        raise LoraLoadError(f"LoRA {key!r} failed its checksum; the file may be corrupt.")
    return target


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
