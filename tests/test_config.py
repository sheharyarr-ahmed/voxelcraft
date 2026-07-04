"""Registry integrity and pinned-constant guards (rules/license-compliance.md, D2, D4).

The registry parametrization is vacuously green while ``LORA_REGISTRY`` is empty; the
constant guards keep the file meaningful in that state and lock the values every other
module depends on.
"""

import re

import pytest

from src import config

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def test_pinned_model_ids() -> None:
    assert config.SD15_MODEL_ID == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert config.CONTROLNET_MODEL_ID == "lllyasviel/control_v11p_sd15_openpose"
    assert config.ANNOTATOR_MODEL_ID == "lllyasviel/Annotators"


def test_generation_bounds() -> None:
    assert config.IMAGE_SIZE == 512
    assert config.MAX_LORAS == 2
    assert (config.LORA_WEIGHT_MIN, config.LORA_WEIGHT_MAX) == (0.0, 1.5)
    assert config.DEFAULT_STEPS == 20
    assert config.MAX_UPLOAD_BYTES == 5 * 1024 * 1024


@pytest.mark.parametrize("key, entry", list(config.LORA_REGISTRY.items()))
def test_registry_entry_complete_and_verified(key: str, entry: config.LoraEntry) -> None:
    assert entry.url.startswith("https://"), f"{key}: url must be a real model-card link"
    assert entry.author, f"{key}: author required"
    assert entry.license, f"{key}: license required"
    assert entry.commercial_use is True, f"{key}: not license-verified (D4)"
    assert entry.base_model == config.SD15_BASE, f"{key}: must be SD 1.5 (D2)"

    hf_hosted = entry.repo_id is not None and entry.weight_name is not None
    civitai_hosted = entry.download_url is not None
    assert hf_hosted ^ civitai_hosted, f"{key}: exactly one weight source required"
    if civitai_hosted:
        assert entry.sha256 and _SHA256.match(entry.sha256), f"{key}: civitai entry needs sha256"
