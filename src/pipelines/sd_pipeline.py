"""Lazy Stable Diffusion 1.5 singleton and text-to-image generation.

Nothing loads at import time (D7): the pipeline is built on first ``get_pipeline()`` call
and cached for the process. Device and dtype policy lives here and nowhere else (D8):
fp16 only on CUDA; fp32 on MPS (fp16 there produces black/NaN images) and CPU. The
``VOXELCRAFT_DEVICE`` env var overrides detection so the smoke test can pin CPU.

The GPU-bound step is isolated in ``_infer`` so Phase 4 can wrap exactly that call with
``@spaces.GPU`` for ZeroGPU without moving model loading or LoRA mutation off the parent
process. The whole mutate-then-infer critical section runs under ``_gpu_lock`` so two
concurrent Gradio requests can never interleave LoRA state or double-book memory,
independent of the queue's concurrency configuration.
"""

from __future__ import annotations

import gc
import logging
import os
import threading
import time
from typing import Any

import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline

from src import config
from src.exceptions import GenerationError, PipelineLoadError
from src.pipelines import lora_manager
from src.schemas import GenerationRequest
from src.utils import capture_metadata, clip_token_overflow, resolve_seed

logger = logging.getLogger(__name__)

# Typed Any: diffusers pipelines compose their components (tokenizer, scheduler, …)
# dynamically, so those attributes are not statically declared on the class.
_pipeline: Any = None
_load_lock = threading.Lock()  # guards one-time singleton construction
_gpu_lock = threading.Lock()  # serializes the mutate-then-infer critical section


def detect_device() -> tuple[str, torch.dtype]:
    """Resolve (device, dtype). VOXELCRAFT_DEVICE wins; else cuda -> mps -> cpu."""
    override = os.environ.get("VOXELCRAFT_DEVICE")
    if override:
        device = override.lower()
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    # fp16 only on CUDA; MPS and CPU stay fp32 (fp16 on MPS yields black/NaN on SD 1.5).
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype


def get_pipeline() -> Any:
    """Load SD 1.5 on first call and cache it; subsequent calls return the same instance."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _load_lock:
        if _pipeline is not None:
            return _pipeline
        device, dtype = detect_device()
        variant = "fp16" if dtype == torch.float16 else None
        start = time.perf_counter()
        try:
            pipe: Any = StableDiffusionPipeline.from_pretrained(
                config.SD15_MODEL_ID, torch_dtype=dtype, variant=variant
            )
        except (OSError, ValueError) as exc:  # HTTPError subclasses OSError
            raise PipelineLoadError(
                "Could not load the Stable Diffusion model. Check the connection and retry."
            ) from exc
        # DPM++ 2M at 20 steps over the checkpoint default PNDM@50 (A10).
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        pipe = pipe.to(device)
        pipe.enable_attention_slicing()
        _pipeline = pipe
        logger.info("Loaded SD 1.5 on %s (%s) in %.1fs", device, dtype, time.perf_counter() - start)
        return _pipeline


def is_loaded() -> bool:
    """Whether the pipeline has been constructed (used for the first-load UI hint)."""
    return _pipeline is not None


def free_memory() -> None:
    """Reclaim allocator caches between generations (device-appropriate)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def generate(request: GenerationRequest) -> tuple[Any, dict[str, Any]]:
    """Run a validated text-to-image request; return (PIL image, metadata dict)."""
    with _gpu_lock:
        pipe = get_pipeline()
        adapter_names, adapter_weights = lora_manager.apply_loras(pipe, request.loras)
        seed = resolve_seed(request.seed)
        # A CPU generator is portable across cpu/mps/cuda, avoiding device-mismatch errors.
        generator = torch.Generator("cpu").manual_seed(seed)
        overflow = clip_token_overflow(pipe.tokenizer, request.prompt)
        start = time.perf_counter()
        try:
            image = _infer(pipe, request, generator)
        except torch.cuda.OutOfMemoryError as exc:
            raise GenerationError("Out of memory — try fewer LoRAs or retry in a moment.") from exc
        finally:
            free_memory()
        elapsed = time.perf_counter() - start
        metadata = capture_metadata(
            adapter_names=adapter_names,
            adapter_weights=adapter_weights,
            seed=seed,
            scheduler=type(pipe.scheduler).__name__,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            inference_seconds=elapsed,
            device=str(pipe.device),
            dtype=str(pipe.dtype),
            truncated_tokens=overflow,
            safety_checker=pipe.safety_checker is not None,
        )
    return image, metadata


def _infer(pipe: Any, request: GenerationRequest, generator: Any) -> Any:
    """The GPU-bound denoise step. Phase 4 wraps exactly this with ``@spaces.GPU``."""
    with torch.inference_mode():
        result = pipe(
            prompt=request.prompt,
            num_inference_steps=request.steps,
            guidance_scale=request.guidance_scale,
            height=config.IMAGE_SIZE,
            width=config.IMAGE_SIZE,
            generator=generator,
        )
    flagged = getattr(result, "nsfw_content_detected", None)
    if flagged and flagged[0]:
        raise GenerationError(
            "The safety filter flagged this output. Try a different prompt or seed."
        )
    return result.images[0]


if __name__ == "__main__":  # documented warm-up: pre-download weights before launching app.py
    logging.basicConfig(level=logging.INFO)
    get_pipeline()
    logger.info("Pipeline warm-up complete.")
