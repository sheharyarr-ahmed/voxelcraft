"""OpenPose skeleton extraction and ControlNet pipeline composition (pose-controlled gen).

Lazy singletons (D7): the detector and ControlNet pipeline load on first Tab-2 use. The
ControlNet pipeline is built from the existing SD 1.5 pipeline's components via ``from_pipe``,
so the UNet, VAE, and text encoder are shared by reference — never a second base download or a
duplicate resident model. ControlNet dtype is matched to the base pipeline or denoising fails
mid-loop. The detector runs on CPU (pose estimation on a 512px image is ~1s and keeps the
skeleton-preview step, which the SPEC requires as intermediate output, off the GPU path).

Pose generation shares ``sd_pipeline.GPU_LOCK`` with text-to-image so the two tabs cannot run
concurrently — on the 8 GB dev machine the base plus ControlNet plus detector cannot coexist
while both are active.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

from src import config
from src.exceptions import GenerationError, PipelineLoadError, PoseDetectionError
from src.pipelines import lora_manager, sd_pipeline
from src.schemas import ControlNetRequest
from src.utils import capture_metadata, clip_token_overflow, resolve_seed

logger = logging.getLogger(__name__)

_detector: Any = None
_cn_pipeline: Any = None
_load_lock = threading.Lock()


def get_detector() -> Any:
    """Load the OpenPose detector on first use (CPU-resident)."""
    global _detector
    if _detector is not None:
        return _detector
    with _load_lock:
        if _detector is not None:
            return _detector
        # Imported here, not at module top: controlnet_aux is slow to import and noisy.
        from controlnet_aux import OpenposeDetector

        try:
            _detector = OpenposeDetector.from_pretrained(config.ANNOTATOR_MODEL_ID)
        except (OSError, ValueError) as exc:
            raise PipelineLoadError(
                "Could not load the pose detector. Check the connection and retry."
            ) from exc
        logger.info("Loaded OpenPose detector")
        return _detector


def get_controlnet_pipeline() -> Any:
    """Compose a ControlNet pipeline from the shared SD 1.5 components on first use."""
    global _cn_pipeline
    if _cn_pipeline is not None:
        return _cn_pipeline
    with _load_lock:
        if _cn_pipeline is not None:
            return _cn_pipeline
        base = sd_pipeline.get_pipeline()
        try:
            controlnet = ControlNetModel.from_pretrained(
                config.CONTROLNET_MODEL_ID, torch_dtype=base.dtype
            )
            # from_pipe reuses base.components by reference — no second UNet/VAE download.
            pipe: Any = StableDiffusionControlNetPipeline.from_pipe(base, controlnet=controlnet)
        except (OSError, ValueError) as exc:
            raise PipelineLoadError(
                "Could not load the ControlNet model. Check the connection and retry."
            ) from exc
        pipe = pipe.to(base.device)
        pipe.enable_attention_slicing()
        _cn_pipeline = pipe
        logger.info("Composed ControlNet pipeline on %s", base.device)
        return _cn_pipeline


def extract_pose(image: Any) -> Any:
    """Extract the OpenPose skeleton; raise PoseDetectionError when no pose is found."""
    detector = get_detector()
    skeleton = detector(image, output_type="pil")
    if skeleton.getbbox() is None:  # an all-black frame means no keypoints were detected
        raise PoseDetectionError(
            "No pose detected in this image — try a photo with a clearly visible person."
        )
    return skeleton


def generate_posed(request: ControlNetRequest) -> tuple[Any, Any, dict[str, Any]]:
    """Run a validated pose-controlled request; return (skeleton, image, metadata).

    Pose extraction runs before the GPU lock is taken — it is cheap, CPU-only, and lets a
    no-pose upload fail without ever occupying the generation slot.
    """
    skeleton = extract_pose(request.reference_image)
    with sd_pipeline.GPU_LOCK:
        pipe = get_controlnet_pipeline()
        adapter_names, adapter_weights = lora_manager.apply_loras(pipe, request.loras)
        seed = resolve_seed(request.seed)
        generator = torch.Generator("cpu").manual_seed(seed)
        overflow = clip_token_overflow(pipe.tokenizer, request.prompt)
        start = time.perf_counter()
        try:
            image, run_device, run_dtype = _infer_posed(pipe, request, skeleton, generator)
            elapsed = time.perf_counter() - start
        except RuntimeError as exc:
            if not sd_pipeline.is_oom(exc):
                raise
            raise GenerationError("Out of memory — try fewer LoRAs or retry in a moment.") from exc
        finally:
            sd_pipeline.free_memory()
        metadata = capture_metadata(
            adapter_names=adapter_names,
            adapter_weights=adapter_weights,
            seed=seed,
            scheduler=type(pipe.scheduler).__name__,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            inference_seconds=elapsed,
            device=run_device,
            dtype=run_dtype,
            truncated_tokens=overflow,
            safety_checker=pipe.safety_checker is not None,
            extra={"conditioning_scale": request.conditioning_scale},
        )
    return skeleton, image, metadata


@sd_pipeline.gpu
def _infer_posed(pipe: Any, request: ControlNetRequest, skeleton: Any, generator: Any) -> Any:
    """The GPU-bound pose-conditioned denoise step; on ZeroGPU runs on an attached GPU (child).

    Returns (image, device, dtype) so the metadata reports where inference actually ran.
    """
    if torch.cuda.is_available() and str(pipe.device).startswith("cpu"):
        pipe.to("cuda")  # ZeroGPU attaches CUDA only inside this call
    device, dtype = str(pipe.device), str(pipe.dtype)
    with torch.inference_mode():
        result = pipe(
            prompt=request.prompt,
            image=skeleton,
            controlnet_conditioning_scale=request.conditioning_scale,
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
    return result.images[0], device, dtype
