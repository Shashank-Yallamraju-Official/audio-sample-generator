import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from automix.gain import gain_stage, peak_dbfs, rms_dbfs

from .conftest import sine


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
