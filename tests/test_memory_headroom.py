from types import SimpleNamespace

import pytest

from deepcompressor.dataset.cache import _check_memory_headroom

GIB = 1024**3


def test_memory_headroom_allows_high_percentage_when_memory_is_available() -> None:
    memory = SimpleNamespace(percent=93.0, available=10 * GIB, total=128 * GIB)

    assert _check_memory_headroom("testing", memory=memory, minimum_available_gib=4) is memory


def test_memory_headroom_rejects_low_available_memory() -> None:
    memory = SimpleNamespace(percent=89.0, available=3 * GIB, total=32 * GIB)

    with pytest.raises(RuntimeError, match=r"3\.00 GiB available, 4\.00 GiB required"):
        _check_memory_headroom("testing", memory=memory, minimum_available_gib=4)


def test_memory_headroom_validates_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPCOMPRESSOR_MIN_AVAILABLE_MEMORY_GIB", "invalid")
    memory = SimpleNamespace(percent=50.0, available=16 * GIB, total=32 * GIB)

    with pytest.raises(ValueError, match="must be a non-negative number"):
        _check_memory_headroom("testing", memory=memory)
