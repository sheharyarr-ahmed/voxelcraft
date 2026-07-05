"""LoRA manager state machine, driven by a fake pipeline (A8 — zero model weights).

Asserts the two verified diffusers gotchas: load-once semantics and enable_lora() before
set_adapters() so a plain-then-LoRA request sequence never silently drops the LoRA.
"""

from typing import Any, Callable

import pytest

from src import config
from src.config import LoraEntry
from src.exceptions import LoraLoadError
from src.pipelines import lora_manager


class FakePipe:
    """Records the diffusers LoRA calls the manager makes, in order."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []

    def load_lora_weights(
        self, source: str, weight_name: str | None = None, adapter_name: str | None = None
    ) -> None:
        self.calls.append(("load", adapter_name))

    def set_adapters(self, names: list[str], adapter_weights: list[float] | None = None) -> None:
        self.calls.append(("set_adapters", tuple(names), tuple(adapter_weights or [])))

    def enable_lora(self) -> None:
        self.calls.append(("enable_lora",))

    def disable_lora(self) -> None:
        self.calls.append(("disable_lora",))

    def unload_lora_weights(self) -> None:
        self.calls.append(("unload",))


@pytest.fixture
def register(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    def add(key: str, base: str = config.SD15_BASE) -> None:
        entry = LoraEntry(
            url="https://example.com/model",
            author="tester",
            license="MIT",
            commercial_use=True,
            base_model=base,
            repo_id="tester/model",
            weight_name=f"{key}.safetensors",
        )
        monkeypatch.setitem(config.LORA_REGISTRY, key, entry)

    add("anime")
    add("realistic")
    add("sdxl_one", base="sdxl")
    return add


def _kinds(pipe: FakePipe) -> list[Any]:
    return [call[0] for call in pipe.calls]


def test_empty_selection_with_nothing_loaded_is_a_noop(register: Callable[..., None]) -> None:
    pipe = FakePipe()
    assert lora_manager.apply_loras(pipe, {}) == ([], [])
    assert pipe.calls == []


def test_empty_selection_after_a_load_disables(register: Callable[..., None]) -> None:
    pipe = FakePipe()
    lora_manager.apply_loras(pipe, {"anime": 1.0})
    pipe.calls.clear()
    lora_manager.apply_loras(pipe, {})
    assert pipe.calls == [("disable_lora",)]


def test_enable_lora_precedes_set_adapters(register: Callable[..., None]) -> None:
    pipe = FakePipe()
    names, weights = lora_manager.apply_loras(pipe, {"anime": 0.8})
    assert _kinds(pipe) == ["load", "enable_lora", "set_adapters"]
    assert ("set_adapters", ("anime",), (0.8,)) in pipe.calls
    assert (names, weights) == (["anime"], [0.8])


def test_adapter_loaded_only_once(register: Callable[..., None]) -> None:
    pipe = FakePipe()
    lora_manager.apply_loras(pipe, {"anime": 1.0})
    lora_manager.apply_loras(pipe, {"anime": 0.5})
    assert [call for call in pipe.calls if call[0] == "load"] == [("load", "anime")]


def test_disable_then_reselect_reenables_without_reloading(register: Callable[..., None]) -> None:
    pipe = FakePipe()
    lora_manager.apply_loras(pipe, {"anime": 1.0})
    lora_manager.apply_loras(pipe, {})  # disable_lora
    pipe.calls.clear()
    lora_manager.apply_loras(pipe, {"anime": 1.0})
    assert _kinds(pipe) == ["enable_lora", "set_adapters"]  # no reload; re-enabled


def test_unknown_lora_rejected(register: Callable[..., None]) -> None:
    with pytest.raises(LoraLoadError, match="Unknown LoRA"):
        lora_manager.apply_loras(FakePipe(), {"ghost": 1.0})


def test_too_many_loras_rejected(register: Callable[..., None]) -> None:
    with pytest.raises(LoraLoadError, match="at most 2"):
        lora_manager.apply_loras(FakePipe(), {"anime": 1.0, "realistic": 1.0, "sdxl_one": 1.0})


def test_sdxl_base_rejected(register: Callable[..., None]) -> None:
    with pytest.raises(LoraLoadError, match="not an SD 1.5"):
        lora_manager.apply_loras(FakePipe(), {"sdxl_one": 1.0})


@pytest.mark.parametrize("weight", [-0.1, 1.6])
def test_weight_out_of_bounds_rejected(register: Callable[..., None], weight: float) -> None:
    with pytest.raises(LoraLoadError, match="between"):
        lora_manager.apply_loras(FakePipe(), {"anime": weight})


def test_failed_load_is_not_cached(register: Callable[..., None]) -> None:
    class BoomPipe(FakePipe):
        def load_lora_weights(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("boom")

    pipe = BoomPipe()
    with pytest.raises(LoraLoadError):
        lora_manager.apply_loras(pipe, {"anime": 1.0})
    assert "anime" not in lora_manager._loaded_for(pipe)
