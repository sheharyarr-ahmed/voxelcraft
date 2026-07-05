"""OpenPose extraction guard — fake detector, no weights, no downloads (Phase 3, A8).

Only ``extract_pose`` is unit-testable without model weights; ControlNet composition and
inference are covered by the Phase 3 acceptance run and the live demo (SPEC design).
"""

import pytest
from PIL import Image, ImageDraw

from src.exceptions import PoseDetectionError
from src.pipelines import controlnet_processor


class _FakeDetector:
    def __init__(self, skeleton: Image.Image) -> None:
        self._skeleton = skeleton

    def __call__(self, image: Image.Image, output_type: str = "pil") -> Image.Image:
        return self._skeleton


def test_extract_pose_rejects_blank_skeleton(monkeypatch: pytest.MonkeyPatch) -> None:
    blank = Image.new("RGB", (512, 512), (0, 0, 0))
    monkeypatch.setattr(controlnet_processor, "get_detector", lambda: _FakeDetector(blank))
    with pytest.raises(PoseDetectionError, match="No pose detected"):
        controlnet_processor.extract_pose(Image.new("RGB", (512, 512)))


def test_extract_pose_returns_skeleton_when_pose_found(monkeypatch: pytest.MonkeyPatch) -> None:
    skeleton = Image.new("RGB", (512, 512), (0, 0, 0))
    ImageDraw.Draw(skeleton).line((100, 100, 220, 300), fill=(255, 0, 0), width=6)
    monkeypatch.setattr(controlnet_processor, "get_detector", lambda: _FakeDetector(skeleton))
    result = controlnet_processor.extract_pose(Image.new("RGB", (512, 512)))
    assert result is skeleton
    assert result.getbbox() is not None
