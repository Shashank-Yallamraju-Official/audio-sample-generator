import numpy as np
import numpy.typing as npt
import pytest

SR = 48_000


def sine(freq: float, seconds: float, amp: float, channels: int = 2) -> npt.NDArray[np.float32]:
    t = np.arange(int(SR * seconds)) / SR
    mono = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.tile(mono, (channels, 1))


@pytest.fixture
def sr() -> int:
    return SR
