"""Pydantic v2 request models — the validation boundary every user input crosses (D6).

Imports ``config`` and PIL only; never torch or gradio. The CLIP 77-token check is
deliberately not here: the tokenizer is lazy-loaded (D7) and lives in the pipeline layer,
which reports overflow into the metadata panel rather than silently truncating.

The LoRA validators read ``config.LORA_REGISTRY`` dynamically (not at class-definition
time) so tests can register fixtures with ``monkeypatch.setitem`` and the empty-registry
production state degrades to LoRA-free generation.
"""

from __future__ import annotations

import re

from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src import config

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")


class GenerationRequest(BaseModel):
    """A validated text-to-image request."""

    prompt: str = Field(min_length=config.PROMPT_MIN_LEN, max_length=config.PROMPT_MAX_LEN)
    loras: dict[str, float] = Field(default_factory=dict)
    seed: int | None = Field(default=None, ge=config.SEED_MIN, le=config.SEED_MAX)
    steps: int = Field(default=config.DEFAULT_STEPS, ge=config.STEPS_MIN, le=config.STEPS_MAX)
    guidance_scale: float = Field(
        default=config.DEFAULT_GUIDANCE, ge=config.GUIDANCE_MIN, le=config.GUIDANCE_MAX
    )

    @field_validator("prompt", mode="before")
    @classmethod
    def _clean_prompt(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Prompt must be text.")
        cleaned = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub("", value)).strip()
        if not cleaned:
            raise ValueError("Prompt is empty after removing control characters.")
        return cleaned

    @field_validator("loras")
    @classmethod
    def _check_loras(cls, value: dict[str, float]) -> dict[str, float]:
        if len(value) > config.MAX_LORAS:
            raise ValueError(f"Select at most {config.MAX_LORAS} LoRAs.")
        for key, weight in value.items():
            if key not in config.LORA_REGISTRY:
                raise ValueError(f"Unknown LoRA: {key!r}.")
            if not config.LORA_WEIGHT_MIN <= weight <= config.LORA_WEIGHT_MAX:
                raise ValueError(
                    f"Weight for {key!r} must be between "
                    f"{config.LORA_WEIGHT_MIN} and {config.LORA_WEIGHT_MAX}."
                )
        return value


class ControlNetRequest(GenerationRequest):
    """A pose-controlled request: everything above plus a reference image and scale.

    The reference image's bytes are validated in ``utils.validate_upload`` before this
    model is constructed; here it is a presence-and-type check only.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    reference_image: Image.Image
    conditioning_scale: float = Field(
        default=config.DEFAULT_CONDITIONING_SCALE,
        ge=config.CONDITIONING_SCALE_MIN,
        le=config.CONDITIONING_SCALE_MAX,
    )
