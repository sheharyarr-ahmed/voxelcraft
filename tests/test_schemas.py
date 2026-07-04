"""GenerationRequest / ControlNetRequest validation boundary (D6)."""

import pytest
from PIL import Image
from pydantic import ValidationError

from src import config
from src.config import LoraEntry
from src.schemas import ControlNetRequest, GenerationRequest


@pytest.fixture
def registered_loras(monkeypatch: pytest.MonkeyPatch) -> tuple[str, str, str]:
    entry = LoraEntry(
        url="https://example.com/model",
        author="tester",
        license="MIT",
        commercial_use=True,
        base_model=config.SD15_BASE,
        repo_id="tester/model",
        weight_name="model.safetensors",
    )
    for key in ("anime", "realistic", "painterly"):
        monkeypatch.setitem(config.LORA_REGISTRY, key, entry)
    return "anime", "realistic", "painterly"


def test_prompt_strips_control_chars_and_collapses_whitespace() -> None:
    req = GenerationRequest(prompt="  hello\x00\x07   world  ")
    assert req.prompt == "hello world"


def test_prompt_all_control_chars_rejected() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="\x00\x01\x02")


@pytest.mark.parametrize("bad", ["ab", "x" * 501])
def test_prompt_length_bounds(bad: str) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(prompt=bad)


def test_too_many_loras_rejected(registered_loras: tuple[str, str, str]) -> None:
    with pytest.raises(ValidationError, match="at most 2"):
        GenerationRequest(prompt="a castle", loras={k: 1.0 for k in registered_loras})


def test_unknown_lora_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown LoRA"):
        GenerationRequest(prompt="a castle", loras={"ghost": 1.0})


@pytest.mark.parametrize("weight", [-0.1, 1.51])
def test_lora_weight_out_of_bounds_rejected(
    registered_loras: tuple[str, str, str], weight: float
) -> None:
    anime = registered_loras[0]
    with pytest.raises(ValidationError, match="between"):
        GenerationRequest(prompt="a castle", loras={anime: weight})


@pytest.mark.parametrize("weight", [0.0, 1.5])
def test_lora_weight_bounds_inclusive(
    registered_loras: tuple[str, str, str], weight: float
) -> None:
    anime = registered_loras[0]
    req = GenerationRequest(prompt="a castle", loras={anime: weight})
    assert req.loras[anime] == weight


@pytest.mark.parametrize("bad_seed", [-2, 2**32])
def test_seed_out_of_range_rejected(bad_seed: int) -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(prompt="a castle", seed=bad_seed)


@pytest.mark.parametrize("seed", [None, -1, 0, 2**32 - 1])
def test_seed_in_range_accepted(seed: int | None) -> None:
    GenerationRequest(prompt="a castle", seed=seed)


def test_controlnet_request_requires_image() -> None:
    with pytest.raises(ValidationError):
        ControlNetRequest(prompt="a castle")  # type: ignore[call-arg]


def test_controlnet_request_inherits_prompt_cleaning() -> None:
    req = ControlNetRequest(prompt="  hi\x00 there  ", reference_image=Image.new("RGB", (8, 8)))
    assert req.prompt == "hi there"


@pytest.mark.parametrize("scale, ok", [(2.1, False), (1.0, True)])
def test_controlnet_conditioning_bounds(scale: float, ok: bool) -> None:
    image = Image.new("RGB", (8, 8))
    if ok:
        req = ControlNetRequest(prompt="a castle", reference_image=image, conditioning_scale=scale)
        assert req.conditioning_scale == scale
    else:
        with pytest.raises(ValidationError):
            ControlNetRequest(prompt="a castle", reference_image=image, conditioning_scale=scale)
