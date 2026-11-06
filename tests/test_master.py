import numpy as np
import numpy.typing as npt
import pytest
from scipy import signal

from automix.gain import peak_dbfs, rms_dbfs
from automix.master import MAX_LIMITING_DB, limit, master, measure

from .conftest import SR, sine


def song(seconds: float = 4.0, seed: int = 0) -> npt.NDArray[np.float32]:
    """Noise-like music bed with kick-like transients, so a limiter has peaks to catch."""
    rng = np.random.default_rng(seed)
    bed = (0.05 * rng.standard_normal((2, int(SR * seconds)))).astype(np.float32)
    t = np.arange(int(0.1 * SR)) / SR
    hit = (0.6 * np.sin(2 * np.pi * 60 * t) * np.exp(-t * 30)).astype(np.float32)
    for start in range(0, bed.shape[1] - len(hit), SR // 2):
        bed[:, start : start + len(hit)] += hit
    return bed


def test_measure_integrated_loudness_of_a_stereo_sine() -> None:
    # BS.1770: a 0 dBFS 1 kHz sine in one channel reads -3.01 LUFS, so in both channels 0 LUFS,
    # and at amplitude 0.1 (-20 dBFS) it reads -20 LUFS.
    assert measure(sine(1000, 2.0, 0.1), SR).lufs == pytest.approx(-20.0, abs=0.1)


def test_true_peak_catches_intersample_overs() -> None:
    t = np.arange(SR) / SR
    tone = (0.9 * np.sin(2 * np.pi * (SR / 4) * t + np.pi / 4)).astype(np.float32)
    stereo = np.tile(tone, (2, 1))
    meter = measure(stereo, SR)
    assert meter.true_peak_dbtp > peak_dbfs(stereo) + 2.5


def test_measure_silence() -> None:
    meter = measure(np.zeros((2, SR), dtype=np.float32), SR)
    assert meter.lufs == -np.inf
    assert meter.true_peak_dbtp == -np.inf


def test_master_hits_an_achievable_loudness_target() -> None:
    result = master(song(), SR, target_lufs=-14.0)
    assert result.after.lufs == pytest.approx(-14.0, abs=0.5)
    assert not result.capped


def test_master_never_exceeds_the_true_peak_ceiling() -> None:
    for target in (-14.0, -9.0):
        result = master(song(), SR, target_lufs=target, ceiling_dbtp=-1.0)
        assert result.after.true_peak_dbtp <= -1.0 + 0.05


def test_master_caps_limiting_instead_of_crushing_the_mix() -> None:
    result = master(song(), SR, target_lufs=-4.0)
    assert result.capped
    assert result.limiting_db <= MAX_LIMITING_DB + 0.01
    assert result.after.lufs < -4.0


def test_master_of_silence_is_a_no_op() -> None:
    silence = np.zeros((2, SR), dtype=np.float32)
    result = master(silence, SR, target_lufs=-14.0)
    np.testing.assert_array_equal(result.audio, silence)


def _band_db(audio: npt.NDArray[np.float32], low: float, high: float) -> float:
    spectrum = np.abs(np.fft.rfft(audio[0])) ** 2
    freqs = np.fft.rfftfreq(audio.shape[1], 1 / SR)
    return float(10 * np.log10(spectrum[(freqs >= low) & (freqs < high)].sum()))


def _dull(audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    sos = signal.butter(2, 2000, btype="lowpass", fs=SR, output="sos")
    return signal.sosfilt(sos, audio, axis=-1).astype(np.float32)


def test_reference_match_moves_tonal_balance_toward_the_reference() -> None:
    reference = song(seed=1)
    dull = _dull(song(seed=2))

    def tilt(x: npt.NDArray[np.float32]) -> float:
        return _band_db(x, 4000, 12000) - _band_db(x, 100, 1000)

    result = master(dull, SR, target_lufs=-14.0, reference=reference)
    assert abs(tilt(result.audio) - tilt(reference)) < abs(tilt(dull) - tilt(reference)) - 3


def test_reference_eq_is_capped() -> None:
    result = master(_dull(song(seed=2)), SR, target_lufs=-14.0, reference=song(seed=1))
    assert result.match_eq_db
    assert max(abs(g) for g in result.match_eq_db.values()) <= 4.0 + 1e-6


def test_reference_sets_the_loudness_target() -> None:
    reference = song(seed=1) * np.float32(0.5)
    result = master(song(seed=2), SR, target_lufs=-9.0, reference=reference)
    assert result.target_lufs == pytest.approx(measure(reference, SR).lufs, abs=0.01)


def test_limit_leaves_audio_under_the_ceiling_untouched() -> None:
    quiet = sine(1000, 1.0, 0.1)
    out, reduction_db = limit(quiet, SR, ceiling_dbtp=-1.0)
    np.testing.assert_allclose(out, quiet, atol=1e-6)
    assert reduction_db == 0.0


def test_limit_holds_the_true_peak_ceiling() -> None:
    out, _ = limit(song() * np.float32(4.0), SR, ceiling_dbtp=-1.0)
    assert measure(out, SR).true_peak_dbtp <= -1.0


def test_limit_turns_a_loud_tone_down_without_distorting_it() -> None:
    loud = sine(1000, 1.0, 2.0)  # 6 dB over full scale
    out, reduction_db = limit(loud, SR, ceiling_dbtp=-1.0)
    middle = out[0, SR // 4 : 3 * SR // 4]
    spectrum = np.abs(np.fft.rfft(middle * np.hanning(len(middle))))
    freqs = np.fft.rfftfreq(len(middle), 1 / SR)
    harmonics_db = 20 * np.log10(spectrum[freqs > 1500].max() / spectrum.max())
    assert harmonics_db < -60
    assert reduction_db == pytest.approx(7.0, abs=0.2)


def test_limit_moves_both_channels_together() -> None:
    loud = sine(1000, 1.0, 2.0)
    loud[1] *= 0.25  # right is 12 dB down, and on its own would need no limiting
    out, _ = limit(loud, SR, ceiling_dbtp=-1.0)
    assert rms_dbfs(out[:1]) - rms_dbfs(out[1:]) == pytest.approx(12.04, abs=0.1)


def test_limit_does_not_mutate_input() -> None:
    loud = sine(1000, 0.5, 2.0)
    before = loud.copy()
    limit(loud, SR, ceiling_dbtp=-1.0)
    np.testing.assert_array_equal(loud, before)


def test_master_reports_the_gain_it_applied() -> None:
    result = master(song(), SR, target_lufs=-20.0)  # a cut, so the limiter has nothing to do
    assert result.gain_db == pytest.approx(result.after.lufs - result.before.lufs, abs=0.1)
    assert result.limiting_db == 0.0


def test_master_lands_on_target_when_the_limiter_is_working() -> None:
    result = master(song(), SR, target_lufs=-14.0)
    assert result.limiting_db > 1.0
    assert result.after.lufs == pytest.approx(-14.0, abs=0.15)


def test_master_says_so_whenever_it_lands_under_target() -> None:
    result = master(song(), SR, target_lufs=-9.0)
    assert result.after.lufs < -9.5
    assert result.capped


def test_measure_reads_audio_too_short_for_loudness_as_unmeasurable() -> None:
    assert measure(sine(1000, 0.2, 0.5), SR).lufs == -np.inf


def test_master_rejects_a_silent_reference() -> None:
    silence = np.zeros((2, SR), dtype=np.float32)
    with pytest.raises(ValueError, match="reference"):
        master(song(), SR, target_lufs=-14.0, reference=silence)
