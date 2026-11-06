import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from automix.gain import active_rms_dbfs, gain_stage, peak_dbfs, rms_dbfs

from .conftest import SR, sine


def test_rms_of_full_scale_sine_is_minus_3_db() -> None:
    assert rms_dbfs(sine(1000, 1.0, 1.0)) == pytest.approx(-3.01, abs=0.01)


def test_peak_of_half_scale_sine_is_minus_6_db() -> None:
    assert peak_dbfs(sine(1000, 1.0, 0.5)) == pytest.approx(-6.02, abs=0.01)


def test_silence_measures_negative_infinity() -> None:
    silence = np.zeros((2, 1000), dtype=np.float32)
    assert rms_dbfs(silence) == -np.inf
    assert peak_dbfs(silence) == -np.inf


def test_gain_stage_hits_target_rms() -> None:
    result = gain_stage(sine(1000, 1.0, 0.01), target_rms_dbfs=-18.0, max_peak_dbfs=-1.0)
    assert rms_dbfs(result.audio) == pytest.approx(-18.0, abs=0.01)


def test_gain_stage_reports_applied_gain() -> None:
    quiet = sine(1000, 1.0, 0.01)
    result = gain_stage(quiet, target_rms_dbfs=-18.0, max_peak_dbfs=-1.0)
    assert result.gain_db == pytest.approx(-18.0 - rms_dbfs(quiet), abs=0.01)


def test_gain_stage_respects_peak_ceiling_over_rms_target() -> None:
    # A spiky signal: hitting -18 dBFS RMS would push its peak above -6 dBFS.
    spiky = np.zeros((2, 48_000), dtype=np.float32)
    spiky[:, ::4800] = 0.5
    result = gain_stage(spiky, target_rms_dbfs=-18.0, max_peak_dbfs=-6.0)
    assert peak_dbfs(result.audio) == pytest.approx(-6.0, abs=0.01)
    assert result.peak_limited


def test_gain_stage_leaves_silence_untouched() -> None:
    silence = np.zeros((2, 1000), dtype=np.float32)
    result = gain_stage(silence)
    assert result.gain_db == 0.0
    np.testing.assert_array_equal(result.audio, silence)


def test_gain_stage_does_not_mutate_input() -> None:
    audio = sine(1000, 0.1, 0.01)
    before = audio.copy()
    gain_stage(audio)
    np.testing.assert_array_equal(audio, before)


@given(amp=st.floats(min_value=1e-4, max_value=1.0), target=st.floats(-30.0, -10.0))
def test_gain_stage_never_exceeds_peak_ceiling(amp: float, target: float) -> None:
    result = gain_stage(sine(440, 0.05, amp), target_rms_dbfs=target, max_peak_dbfs=-6.0)
    assert peak_dbfs(result.audio) <= -6.0 + 1e-3


def bursts(starts: list[float], seconds: float = 10.0) -> npt.NDArray[np.float32]:
    """One second of half-scale sine at each start time, silence elsewhere."""
    out = np.zeros((2, int(SR * seconds)), dtype=np.float32)
    for start in starts:
        out[:, int(start * SR) : int((start + 1) * SR)] = sine(1000, 1.0, 0.5)
    return out


def test_active_rms_matches_rms_for_a_part_that_plays_throughout() -> None:
    steady = sine(1000, 2.0, 0.5)
    assert active_rms_dbfs(steady, SR) == pytest.approx(rms_dbfs(steady), abs=0.01)


def test_active_rms_ignores_the_silence_around_a_part() -> None:
    # Whole-file RMS reads 10 dB low here; the edges of the burst cost about 1 dB.
    playing = rms_dbfs(sine(1000, 1.0, 0.5))
    assert active_rms_dbfs(bursts([4.0]), SR) == pytest.approx(playing, abs=1.5)


def test_active_rms_is_the_same_however_often_a_part_plays() -> None:
    once = active_rms_dbfs(bursts([4.0]), SR)
    often = active_rms_dbfs(bursts([0.0, 2.0, 4.0, 6.0, 8.0]), SR)
    assert once == pytest.approx(often, abs=0.1)


def test_active_rms_of_silence_is_negative_infinity() -> None:
    assert active_rms_dbfs(np.zeros((2, SR), dtype=np.float32), SR) == -np.inf


def test_active_rms_of_a_clip_shorter_than_one_window_is_its_rms() -> None:
    short = sine(1000, 0.1, 0.5)
    assert active_rms_dbfs(short, SR) == pytest.approx(rms_dbfs(short), abs=0.01)
