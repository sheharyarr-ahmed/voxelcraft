"""Static configuration: pinned model IDs, generation bounds, and the LoRA registry.

Pure data — no torch, no pydantic, no I/O at import time (D7). ``LORA_REGISTRY`` stays
empty until each entry's commercial-use license is verified by hand (D4); tooling never
sets ``commercial_use=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- Model identifiers (pinned) -------------------------------------------------------
# The historical runwayml/stable-diffusion-v1-5 repo was removed from the Hub; this is
# the live community mirror. SDXL and other bases are out of scope (D2).
SD15_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROLNET_MODEL_ID = "lllyasviel/control_v11p_sd15_openpose"
ANNOTATOR_MODEL_ID = "lllyasviel/Annotators"
SD15_BASE = "sd-1.5"

# --- Generation bounds ----------------------------------------------------------------
IMAGE_SIZE = 512
MAX_LORAS = 2
LORA_WEIGHT_MIN = 0.0
LORA_WEIGHT_MAX = 1.5
DEFAULT_STEPS = 20  # DPM++ 2M reaches SD 1.5 quality here; see decisions.md A10
DEFAULT_GUIDANCE = 7.5
STEPS_MIN = 1
STEPS_MAX = 50
GUIDANCE_MIN = 1.0
GUIDANCE_MAX = 15.0
CONDITIONING_SCALE_MIN = 0.0
CONDITIONING_SCALE_MAX = 2.0
DEFAULT_CONDITIONING_SCALE = 1.0
PROMPT_MIN_LEN = 3
PROMPT_MAX_LEN = 500
SEED_MIN = -1  # -1 (and None) mean "draw a random seed"
SEED_MAX = 2**32 - 1

# --- Upload limits --------------------------------------------------------------------
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_UPLOAD_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
ALLOWED_IMAGE_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})

# --- Local cache (gitignored via models/ and *.safetensors) ---------------------------
LORA_CACHE_DIR = Path("models/loras")


@dataclass(frozen=True)
class LoraEntry:
    """One license-verified, pre-trained LoRA the app can apply.

    ``commercial_use`` is ``True`` only after Sheharyar personally verifies the model
    card (D4). Exactly one weight source is set: an HF Hub pair (``repo_id`` +
    ``weight_name``) or a civitai download (``download_url``, fetched to ``LORA_CACHE_DIR``
    and checked against ``sha256``).
    """

    url: str
    author: str
    license: str
    commercial_use: bool
    base_model: str
    trigger: str | None = None
    repo_id: str | None = None
    weight_name: str | None = None
    download_url: str | None = None
    sha256: str | None = None


# Each entry's commercial-use license was verified by hand against the Hugging Face model
# card on the date noted (rules/license-compliance.md, D4). All are HF-hosted and public, so
# they load via load_lora_weights(repo_id, weight_name=...) with no auth token.
LORA_REGISTRY: dict[str, LoraEntry] = {
    # Verified 2026-07-05 — HF card + API (gated: false). License creativeml-openrail-m, the
    # same license SD 1.5 ships under; its Attachment A use-restrictions do not restrict
    # commercial use of generated images. Pin the -sd-1-5 file (the repo also has an SD-2.1 one).
    "watercolor": LoraEntry(
        url="https://huggingface.co/fladdict/watercolor",
        author="fladdict",
        license="CreativeML Open RAIL-M",
        commercial_use=True,
        base_model=SD15_BASE,
        trigger="watercolor painting",
        repo_id="fladdict/watercolor",
        weight_name="fladdict-watercolor-sd-1-5.safetensors",
    ),
    # Verified 2026-07-05 — HF card. bespoke-lora-trained-license; its license_link permits
    # "Sell images they generate" and "Run on services that generate images for money"; only
    # "Sell this model or merges" is disallowed, which this app does not do.
    "pixelart": LoraEntry(
        url="https://huggingface.co/artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
        author="artificialguybr",
        license="bespoke-lora-trained-license (commercial image use permitted)",
        commercial_use=True,
        base_model=SD15_BASE,
        trigger="Pixel Art, PixArFK",
        repo_id="artificialguybr/pixelartredmond-1-5v-pixel-art-loras-for-sd-1-5",
        weight_name="PixelArtRedmond15V-PixelArt-PIXARFK.safetensors",
    ),
    # Verified 2026-07-05 — HF card. Same bespoke-lora-trained-license as pixelart: commercial
    # use of generated images permitted, selling the model disallowed (not needed here).
    "render3d": LoraEntry(
        url="https://huggingface.co/artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5",
        author="artificialguybr",
        license="bespoke-lora-trained-license (commercial image use permitted)",
        commercial_use=True,
        base_model=SD15_BASE,
        trigger="3D Render Style, 3DRenderAF",
        repo_id="artificialguybr/3d-redmond-1-5v-3d-render-style-for-liberte-redmond-sd-1-5",
        weight_name="3DRedmond15V-LiberteRedmond-3DRenderStyle-3DRenderAF.safetensors",
    ),
}
