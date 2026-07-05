"""VoxelCraft — Gradio Blocks entry point (the only module that imports gradio).

Three tabs: text-to-image with LoRA stacking (Tab 1), pose-controlled generation (Tab 2),
and a static training-methodology reference (Tab 3). Pipeline modules are imported inside
handler bodies so ``import app`` stays instant and free of torch. Every raw input crosses the
Pydantic boundary before any pipeline call, and every failure maps to a single-sentence
``gr.Error`` rather than a traceback.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import gradio as gr

from src import config

logger = logging.getLogger(__name__)

METHODOLOGY_DOC = Path(__file__).parent / "docs" / "LORA_TRAINING.md"
TAB3_BANNER = "> **Methodology reference. This app applies pre-trained LoRAs.**"
NONE = "(none)"


def _seed_value(raw: float | None) -> int | None:
    """Map an empty/blank seed field to None (random); otherwise coerce to int."""
    if raw is None or raw == "":
        return None
    return int(raw)


def _first_error_message(exc: Exception) -> str:
    """Extract one human sentence from a pydantic ValidationError, dropping its prefix."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        detail = errors()
        if detail:
            message = str(detail[0].get("msg", ""))
            return message.removeprefix("Value error, ") or "Invalid input."
    return str(exc) or "Invalid input."


def _collect_loras(selections: list[tuple[str, float]]) -> dict[str, float]:
    """Turn the LoRA rows into a validated {name: weight} dict, rejecting duplicates."""
    loras: dict[str, float] = {}
    for name, weight in selections:
        if name and name != NONE:
            if name in loras:
                raise gr.Error(f"{name!r} is selected twice — pick two different LoRAs.")
            loras[name] = float(weight)
    return loras


def _lora_rows() -> tuple[Any, Any, Any, Any]:
    """Build two LoRA selection rows, or an empty-state notice when the registry is empty.

    Returns four components in a fixed order (dropdown1, weight1, dropdown2, weight2) so both
    tabs wire the same handler signature. When no LoRAs are registered, State components hold
    the "(none)" sentinel so the handler code path is identical.
    """
    registry_keys = list(config.LORA_REGISTRY)
    if not registry_keys:
        gr.Markdown(
            "_No LoRAs are registered yet — license verification is pending. "
            "Generation runs on the base Stable Diffusion 1.5 model._"
        )
        return gr.State(NONE), gr.State(1.0), gr.State(NONE), gr.State(1.0)

    choices = [NONE, *registry_keys]
    triggers = " · ".join(
        f"**{key}**: `{entry.trigger}`"
        for key, entry in config.LORA_REGISTRY.items()
        if entry.trigger
    )
    if triggers:
        gr.Markdown(
            f"Include a LoRA's trigger word in your prompt for the strongest effect — {triggers}"
        )
    components: list[Any] = []
    for index in (1, 2):
        with gr.Row():
            components.append(gr.Dropdown(choices=choices, value=NONE, label=f"LoRA {index}"))
            components.append(
                gr.Slider(
                    minimum=config.LORA_WEIGHT_MIN,
                    maximum=config.LORA_WEIGHT_MAX,
                    value=1.0,
                    step=0.05,
                    label=f"Weight {index}",
                )
            )
    return components[0], components[1], components[2], components[3]


def _first_load_notice() -> None:
    from src.pipelines import sd_pipeline

    if not sd_pipeline.is_loaded():
        gr.Info("First generation loads Stable Diffusion 1.5 — this takes a moment.")


def _warn_truncation(metadata: dict[str, object]) -> None:
    if metadata.get("truncated_tokens"):
        gr.Warning(
            "Prompt exceeded CLIP's 77-token limit; "
            f"the last {metadata['truncated_tokens']} tokens were ignored."
        )


def on_generate(
    prompt: str,
    lora1: str,
    weight1: float,
    lora2: str,
    weight2: float,
    seed: float | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008 (gradio idiom)
) -> tuple[object, dict[str, object]]:
    """Tab 1 handler: validate inputs, generate an image, return it with its metadata."""
    from pydantic import ValidationError

    from src.exceptions import VoxelCraftError
    from src.pipelines import sd_pipeline
    from src.schemas import GenerationRequest

    loras = _collect_loras([(lora1, weight1), (lora2, weight2)])
    try:
        request = GenerationRequest(prompt=prompt, loras=loras, seed=_seed_value(seed))
    except ValidationError as exc:
        raise gr.Error(_first_error_message(exc)) from None

    _first_load_notice()
    try:
        image, metadata = sd_pipeline.generate(request)
    except VoxelCraftError as exc:
        raise gr.Error(str(exc)) from None

    _warn_truncation(metadata)
    return image, metadata


def on_pose(
    prompt: str,
    reference_path: str | None,
    conditioning_scale: float,
    lora1: str,
    weight1: float,
    lora2: str,
    weight2: float,
    seed: float | None,
    progress: gr.Progress = gr.Progress(track_tqdm=True),  # noqa: B008 (gradio idiom)
) -> tuple[object, object, dict[str, object]]:
    """Tab 2 handler: validate the upload, extract the pose, generate a posed image."""
    from pydantic import ValidationError

    from src.exceptions import VoxelCraftError
    from src.pipelines import controlnet_processor
    from src.schemas import ControlNetRequest
    from src.utils import validate_upload

    if not reference_path:
        raise gr.Error("Upload a reference photo first.")
    try:
        reference_image = validate_upload(Path(reference_path))
    except VoxelCraftError as exc:
        raise gr.Error(str(exc)) from None

    loras = _collect_loras([(lora1, weight1), (lora2, weight2)])
    try:
        request = ControlNetRequest(
            prompt=prompt,
            reference_image=reference_image,
            loras=loras,
            seed=_seed_value(seed),
            conditioning_scale=conditioning_scale,
        )
    except ValidationError as exc:
        raise gr.Error(_first_error_message(exc)) from None

    _first_load_notice()
    try:
        skeleton, image, metadata = controlnet_processor.generate_posed(request)
    except VoxelCraftError as exc:
        raise gr.Error(str(exc)) from None

    _warn_truncation(metadata)
    return skeleton, image, metadata


def _load_methodology() -> str:
    try:
        return METHODOLOGY_DOC.read_text(encoding="utf-8")
    except OSError:
        return f"{TAB3_BANNER}\n\nTraining methodology reference — see docs/LORA_TRAINING.md."


def _build_generate_tab() -> None:
    gr.Markdown("### Text to image\nStable Diffusion 1.5, 512x512.")
    prompt = gr.Textbox(label="Prompt", lines=2, placeholder="a lighthouse at sunset")
    lora1, weight1, lora2, weight2 = _lora_rows()
    seed = gr.Number(label="Seed (blank = random)", value=None, precision=0)
    generate_btn = gr.Button("Generate", variant="primary")
    image_out = gr.Image(label="Result", format="png")
    metadata_out = gr.JSON(label="Generation metadata")

    generate_btn.click(
        fn=on_generate,
        inputs=[prompt, lora1, weight1, lora2, weight2, seed],
        outputs=[image_out, metadata_out],
        concurrency_limit=1,
        concurrency_id="gpu",
        api_visibility="private",
        show_progress="full",
    )


def _build_pose_tab() -> None:
    gr.Markdown(
        "### Pose control\nUpload a reference photo; VoxelCraft extracts the OpenPose "
        "skeleton and generates a new image matching the pose."
    )
    prompt = gr.Textbox(label="Prompt", lines=2, placeholder="a knight in armor, dramatic light")
    reference = gr.Image(label="Reference photo", type="filepath")
    conditioning = gr.Slider(
        minimum=config.CONDITIONING_SCALE_MIN,
        maximum=config.CONDITIONING_SCALE_MAX,
        value=config.DEFAULT_CONDITIONING_SCALE,
        step=0.05,
        label="ControlNet conditioning scale",
    )
    lora1, weight1, lora2, weight2 = _lora_rows()
    seed = gr.Number(label="Seed (blank = random)", value=None, precision=0)
    pose_btn = gr.Button("Generate posed image", variant="primary")
    with gr.Row():
        skeleton_out = gr.Image(label="Detected pose")
        image_out = gr.Image(label="Result", format="png")
    metadata_out = gr.JSON(label="Generation metadata")

    pose_btn.click(
        fn=on_pose,
        inputs=[prompt, reference, conditioning, lora1, weight1, lora2, weight2, seed],
        outputs=[skeleton_out, image_out, metadata_out],
        concurrency_limit=1,
        concurrency_id="gpu",
        api_visibility="private",
        show_progress="full",
    )


def build_demo() -> gr.Blocks:
    """Construct the three-tab Blocks app (no models load here — only on generate)."""
    with gr.Blocks(title="VoxelCraft") as demo:
        gr.Markdown("# VoxelCraft\nStable Diffusion 1.5 + LoRA application with ControlNet.")
        with gr.Tab("Generate"):
            _build_generate_tab()
        with gr.Tab("Pose Control"):
            _build_pose_tab()
        with gr.Tab("Training Methodology"):
            gr.Markdown(_load_methodology())
    return demo


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    demo = build_demo()
    demo.queue(default_concurrency_limit=1, max_size=8)
    demo.launch()
