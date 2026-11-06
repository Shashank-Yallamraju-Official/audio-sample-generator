import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from automix.sidechain import DuckSettings, apply_ducking, detect_onsets, duck_curve

SR = 48_000


def kick_track(onset_times_s: list[float], seconds: float = 2.0) -> npt.NDArray[np.float32]:
    """Decaying 55 Hz thumps at the given times."""
    out = np.zeros((2, int(SR * seconds)), dtype=np.float32)
    length = int(0.15 * SR)
    t = np.arange(length) / SR
    hit = (np.sin(2 * np.pi * 55 * t) * np.exp(-t * 30)).astype(np.float32)
    for onset in onset_times_s:
        start = int(onset * SR)
        end = min(start + length, out.shape[1])
        out[:, start:end] += hit[: end - start]
    return out


def test_detects_each_kick_within_5_ms() -> None:
    times = [0.0, 0.5, 1.0, 1.5]
    onsets = detect_onsets(kick_track(times), SR)
    assert len(onsets) == len(times)
    np.testing.assert_allclose(onsets / SR, times, atol=0.005)


def test_no_onsets_in_silence() -> None:
    assert len(detect_onsets(np.zeros((2, SR), dtype=np.float32), SR)) == 0


def test_ignores_retriggers_closer_than_min_interval() -> None:
    onsets = detect_onsets(kick_track([0.5, 0.52]), SR, min_interval_s=0.05)
    assert len(onsets) == 1


def test_duck_curve_is_unity_without_onsets() -> None:
    curve = duck_curve(SR, np.array([], dtype=np.int64), SR, DuckSettings(depth_db=6.0))
    np.testing.assert_array_equal(curve, np.ones(SR, dtype=np.float32))


def test_duck_curve_reaches_full_depth_after_attack() -> None:
    curve = duck_curve(
        SR, np.array([1000]), SR, DuckSettings(6.0, attack_s=0.002, hold_s=0.03, release_s=0.1)
    )
    at_hold = 1000 + int(0.002 * SR) + 10
    assert 20 * np.log10(curve[at_hold]) == pytest.approx(-6.0, abs=0.01)


def test_duck_curve_recovers_to_unity_after_release() -> None:
    curve = duck_curve(
        SR, np.array([0]), SR, DuckSettings(12.0, attack_s=0.002, hold_s=0.03, release_s=0.1)
    )
    assert curve[int(0.2 * SR)] == pytest.approx(1.0)


def test_duck_curve_is_unity_before_first_onset() -> None:
    curve = duck_curve(SR, np.array([SR // 2]), SR, DuckSettings(depth_db=6.0))
    np.testing.assert_array_equal(curve[: SR // 2], 1.0)


@given(depth=st.floats(0.0, 40.0), onset=st.integers(0, SR - 1))
def test_duck_curve_stays_between_floor_and_unity(depth: float, onset: int) -> None:
    curve = duck_curve(SR, np.array([onset]), SR, DuckSettings(depth_db=depth))
    floor = 10 ** (-depth / 20)
    assert curve.min() >= floor - 1e-6
    assert curve.max() <= 1.0 + 1e-6


def test_apply_ducking_multiplies_every_channel() -> None:
    audio = np.ones((2, 4), dtype=np.float32)
    curve = np.array([1.0, 0.5, 0.25, 1.0], dtype=np.float32)
    np.testing.assert_allclose(apply_ducking(audio, curve), [curve, curve])


def test_apply_ducking_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="length"):
        apply_ducking(np.ones((2, 4), dtype=np.float32), np.ones(3, dtype=np.float32))


def test_duck_settings_default_to_a_gentle_sixteenth_style_duck() -> None:
    settings = DuckSettings()
    assert settings.depth_db == 6.0
    assert settings.release_s == 0.1


def deep_kick_track(sr: int, times_s: list[float], seconds: float = 4.0) -> npt.NDArray[np.float32]:
    """Slow 50 Hz kicks with a long tail, rendered at ``sr``."""
    out = np.zeros((2, int(sr * seconds)), dtype=np.float32)
    t = np.arange(int(0.3 * sr)) / sr
    hit = (np.sin(2 * np.pi * 50 * t) * np.exp(-t * 12)).astype(np.float32)
    for onset in times_s:
        start = int(onset * sr)
        end = min(start + len(hit), out.shape[1])
        out[:, start:end] += hit[: end - start]
    return out


@pytest.mark.parametrize("sr", [44_100, 48_000, 88_200, 96_000])
def test_detects_each_kick_once_at_any_sample_rate(sr: int) -> None:
    times = [i * 0.5 for i in range(8)]
    onsets = detect_onsets(deep_kick_track(sr, times), sr)
    assert len(onsets) == len(times)
    np.testing.assert_allclose(onsets / sr, times, atol=0.005)
