import numpy as np
import numpy.typing as npt
import pytest

from automix.compression import (
    COMP_BY_ROLE,
    MAX_GR_PER_STAGE_DB,
    CompSettings,
    compress,
    fit_threshold,
    measure_gr_db,
)
from automix.eq import Role
from automix.gain import rms_dbfs

from .conftest import SR, sine


def dynamic_signal(seconds: float = 2.0) -> npt.NDArray[np.float32]:
    """A 220 Hz tone alternating between loud and quiet every 100 ms, like a played part."""
    tone = sine(220, seconds, 1.0)
    n = tone.shape[1]
    envelope = np.where((np.arange(n) // (SR // 10)) % 2 == 0, 0.5, 0.1).astype(np.float32)
    return tone * envelope


SETTINGS = CompSettings(ratio=4.0, attack_ms=10.0, release_ms=100.0, target_gr_db=4.0)


def test_every_role_has_compression_settings() -> None:
    assert set(COMP_BY_ROLE) == set(Role)


def test_measure_gr_is_zero_when_threshold_is_above_signal() -> None:
    assert measure_gr_db(dynamic_signal(), SR, threshold_db=0.0, settings=SETTINGS) == (
        pytest.approx(0.0, abs=0.05)
    )


def test_measure_gr_grows_as_threshold_drops() -> None:
    audio = dynamic_signal()
    gentle = measure_gr_db(audio, SR, threshold_db=-10.0, settings=SETTINGS)
    heavy = measure_gr_db(audio, SR, threshold_db=-30.0, settings=SETTINGS)
    assert heavy > gentle > 0.0


@pytest.mark.parametrize("target", [2.0, 4.0, 6.0])
def test_fit_threshold_lands_on_target_gain_reduction(target: float) -> None:
    audio = dynamic_signal()
    settings = CompSettings(ratio=4.0, attack_ms=10.0, release_ms=100.0, target_gr_db=target)
    threshold = fit_threshold(audio, SR, settings)
    assert measure_gr_db(audio, SR, threshold, settings) == pytest.approx(target, abs=0.25)


def test_compress_matches_input_loudness_with_makeup_gain() -> None:
    audio = dynamic_signal()
    result = compress(audio, SR, SETTINGS)
    assert rms_dbfs(result.audio) == pytest.approx(rms_dbfs(audio), abs=0.1)
    assert result.makeup_db == pytest.approx(result.gr_db, abs=0.1)


def test_compress_reduces_dynamic_range() -> None:
    audio = dynamic_signal()
    out = compress(audio, SR, SETTINGS).audio

    def loud_to_quiet_db(x: npt.NDArray[np.float32]) -> float:
        loud = x[:, SR // 20 : SR // 10]  # settled part of a loud segment
        quiet = x[:, SR // 10 + SR // 20 : SR // 5]  # settled part of a quiet segment
        return rms_dbfs(loud) - rms_dbfs(quiet)

    assert loud_to_quiet_db(out) < loud_to_quiet_db(audio) - 2.0


def test_heavy_targets_split_into_two_stages() -> None:
    audio = dynamic_signal()
    heavy = CompSettings(ratio=4.0, attack_ms=5.0, release_ms=80.0, target_gr_db=8.0)
    result = compress(audio, SR, heavy)
    assert len(result.stages) == 2
    assert all(stage.gr_db <= MAX_GR_PER_STAGE_DB + 0.25 for stage in result.stages)
    assert result.gr_db == pytest.approx(8.0, abs=0.5)


def test_light_targets_use_one_stage() -> None:
    assert len(compress(dynamic_signal(), SR, SETTINGS).stages) == 1


def test_compress_silence_is_a_no_op() -> None:
    silence = np.zeros((2, SR), dtype=np.float32)
    result = compress(silence, SR, SETTINGS)
    np.testing.assert_array_equal(result.audio, silence)
    assert result.stages == ()
    assert result.gr_db == 0.0


def test_compress_does_not_mutate_input() -> None:
    audio = dynamic_signal(0.5)
    before = audio.copy()
    compress(audio, SR, SETTINGS)
    np.testing.assert_array_equal(audio, before)


def test_returns_are_not_compressed_by_default() -> None:
    audio = dynamic_signal(0.5)
    result = compress(audio, SR, COMP_BY_ROLE[Role.RETURN])
    np.testing.assert_array_equal(result.audio, audio)
    assert result.stages == ()
