"""Upload validation, seed resolution, token overflow, and metadata capture."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from src import config, utils
from src.exceptions import UploadValidationError


def test_validate_upload_accepts_png_and_downscales(tmp_path: Path) -> None:
    path = tmp_path / "big.png"
    Image.new("RGB", (1024, 768)).save(path)
    result = utils.validate_upload(path)
    assert result.mode == "RGB"
    assert max(result.size) == config.IMAGE_SIZE


def test_validate_upload_accepts_webp(tmp_path: Path) -> None:
    path = tmp_path / "photo.webp"
    Image.new("RGB", (600, 400)).save(path, format="WEBP")
    assert utils.validate_upload(path).mode == "RGB"


def test_validate_upload_rejects_bad_extension(tmp_path: Path) -> None:
    path = tmp_path / "clip.gif"
    Image.new("RGB", (10, 10)).save(path, format="GIF")
    with pytest.raises(UploadValidationError, match="Unsupported file type"):
        utils.validate_upload(path)


def test_validate_upload_rejects_oversize_with_size_named(tmp_path: Path) -> None:
    path = tmp_path / "huge.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * (config.MAX_UPLOAD_BYTES + 1))
    with pytest.raises(UploadValidationError, match="MB"):
        utils.validate_upload(path)


def test_validate_upload_rejects_garbage_bytes(tmp_path: Path) -> None:
    path = tmp_path / "fake.png"
    path.write_bytes(b"not an image at all")
    with pytest.raises(UploadValidationError, match="not a readable image"):
        utils.validate_upload(path)


def test_validate_upload_rejects_format_spoof(tmp_path: Path) -> None:
    # A real GIF wearing a .png name: passes the suffix check, caught by the format check.
    path = tmp_path / "sneaky.png"
    Image.new("RGB", (10, 10)).save(path, format="GIF")
    with pytest.raises(UploadValidationError, match="not PNG"):
        utils.validate_upload(path)


def test_resolve_seed_passes_through_explicit_value() -> None:
    assert utils.resolve_seed(42) == 42


@pytest.mark.parametrize("seed", [None, -1])
def test_resolve_seed_draws_random(seed: int | None) -> None:
    value = utils.resolve_seed(seed)
    assert 0 <= value <= config.SEED_MAX


class _FakeTokenizer:
    """Minimal stand-in for CLIPTokenizer: reports a fixed token count and a 77 window."""

    model_max_length = 77

    def __init__(self, token_count: int) -> None:
        self._token_count = token_count

    def __call__(self, prompt: str) -> SimpleNamespace:
        return SimpleNamespace(input_ids=list(range(self._token_count)))


def test_clip_token_overflow_under_limit() -> None:
    assert utils.clip_token_overflow(_FakeTokenizer(50), "short") == 0


def test_clip_token_overflow_over_limit() -> None:
    assert utils.clip_token_overflow(_FakeTokenizer(100), "long") == 23


def test_capture_metadata_has_locked_keys() -> None:
    metadata = utils.capture_metadata(
        adapter_names=[],
        adapter_weights=[],
        seed=42,
        scheduler="DPMSolverMultistepScheduler",
        steps=20,
        guidance_scale=7.5,
        inference_seconds=1.234,
        device="cpu",
        dtype="torch.float32",
    )
    for key in (
        "loras_applied",
        "lora_weights",
        "seed",
        "scheduler",
        "steps",
        "guidance_scale",
        "inference_seconds",
        "device",
        "dtype",
        "truncated_tokens",
        "safety_checker",
    ):
        assert key in metadata
    assert metadata["loras_applied"] == []
    assert metadata["inference_seconds"] == 1.23
    assert metadata["safety_checker"] == "enabled"


def test_capture_metadata_merges_extra() -> None:
    metadata = utils.capture_metadata(
        adapter_names=["anime"],
        adapter_weights=[0.8],
        seed=1,
        scheduler="s",
        steps=20,
        guidance_scale=7.5,
        inference_seconds=1.0,
        device="cpu",
        dtype="torch.float32",
        extra={"conditioning_scale": 1.0},
    )
    assert metadata["conditioning_scale"] == 1.0
    assert metadata["loras_applied"] == ["anime"]
