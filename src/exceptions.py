"""Typed exceptions raised by the validation and pipeline layers.

Every exception here carries a single, user-facing sentence as its message. Only the
Gradio layer in ``app.py`` catches ``VoxelCraftError`` and maps it to ``gr.Error``;
nothing under ``src/`` imports gradio (rules/code-style.md).
"""


class VoxelCraftError(Exception):
    """Base class for every error VoxelCraft raises deliberately."""


class PipelineLoadError(VoxelCraftError):
    """The Stable Diffusion pipeline or one of its model components failed to load."""


class LoraLoadError(VoxelCraftError):
    """A LoRA could not be validated, downloaded, or applied."""


class GenerationError(VoxelCraftError):
    """Image generation failed at inference time (including out-of-memory)."""


class PoseDetectionError(VoxelCraftError):
    """OpenPose found no usable skeleton in the reference image."""


class UploadValidationError(VoxelCraftError, ValueError):
    """An uploaded image failed format, size, or decode validation.

    Also a ``ValueError`` so upload checks read naturally as value validation while
    still being catchable as a single ``VoxelCraftError`` at the UI boundary.
    """
