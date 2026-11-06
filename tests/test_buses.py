import numpy as np
import numpy.typing as npt
import pytest

from automix.buses import (
    BUS_BY_ROLE,
    Bus,
    make_returns,
    mono_low_end,
    parallel_compress,
    stereo_correlation,
)
from automix.eq import Role
from automix.gain import rms_dbfs

from .conftest import SR, sine


def anti_phase(freq: float, seconds: float = 1.0) -> npt.NDArray[np.float32]:
    tone = sine(freq, seconds, 0.5)
    tone[1] *= -1
    return tone


def test_every_role_is_routed_to_a_bus() -> None:
    assert set(BUS_BY_ROLE) == set(Role)
    assert BUS_BY_ROLE[Role.KICK] is Bus.DRUMS
    assert BUS_BY_ROLE[Role.RETURN] is Bus.FX


def test_correlation_of_identical_inverted_and_silent_channels() -> None:
    mono = sine(440, 0.5, 0.5)
    assert stereo_correlation(mono) == pytest.approx(1.0)
    assert stereo_correlation(anti_phase(440, 0.5)) == pytest.approx(-1.0)
    assert stereo_correlation(np.zeros((2, 100), dtype=np.float32)) == 0.0


def test_mono_low_end_collapses_wide_bass() -> None:
    out = mono_low_end(anti_phase(45, 2.0), SR, crossover_hz=120)
    settled = out[:, SR // 2 :]
    # Anti-phase below the crossover cancels when summed to mono.
    assert rms_dbfs(settled) < rms_dbfs(anti_phase(45, 2.0)) - 20


def test_mono_low_end_keeps_stereo_width_up_high() -> None:
    out = mono_low_end(anti_phase(3000), SR, crossover_hz=120)
    assert stereo_correlation(out[:, SR // 4 :]) < -0.9


def test_mono_low_end_is_flat_on_mono_material() -> None:
    for freq in (60.0, 120.0, 1000.0):
        tone = sine(freq, 2.0, 0.5)
        out = mono_low_end(tone, SR, crossover_hz=120)
        assert rms_dbfs(out[:, SR // 2 :]) == pytest.approx(rms_dbfs(tone[:, SR // 2 :]), abs=0.3)


def _drum_loop() -> npt.NDArray[np.float32]:
    out = np.zeros((2, 2 * SR), dtype=np.float32)
    t = np.arange(int(0.1 * SR)) / SR
    hit = (0.9 * np.sin(2 * np.pi * 80 * t) * np.exp(-t * 40)).astype(np.float32)
    for start in range(0, 2 * SR - len(hit), SR // 4):
        out[:, start : start + len(hit)] += hit
    return out


def _crest_db(audio: npt.NDArray[np.float32]) -> float:
    return float(20 * np.log10(np.max(np.abs(audio)))) - rms_dbfs(audio)


def test_parallel_compression_keeps_level_and_lifts_the_body() -> None:
    drums = _drum_loop()
    out = parallel_compress(drums, SR, blend=0.3)
    assert rms_dbfs(out) == pytest.approx(rms_dbfs(drums), abs=0.2)
    assert _crest_db(out) < _crest_db(drums) - 0.5


def test_parallel_compression_off_is_a_copy() -> None:
    drums = _drum_loop()
    out = parallel_compress(drums, SR, blend=0.0)
    np.testing.assert_array_equal(out, drums)
    assert out is not drums


def test_no_send_sources_means_no_returns() -> None:
    assert make_returns({Role.KICK: [sine(55, 1.0, 0.5)]}, SR, bpm=120) == {}


def test_vocal_feeds_reverb_and_delay_returns() -> None:
    returns = make_returns({Role.VOCAL: [sine(440, 1.0, 0.5)]}, SR, bpm=120)
    assert set(returns) == {"Reverb Return.wav", "Delay Return.wav"}
    assert all(np.isfinite(rms_dbfs(r)) for r in returns.values())


def test_delay_return_echoes_on_the_dotted_eighth() -> None:
    click = np.zeros((2, 2 * SR), dtype=np.float32)
    click[:, :48] = 0.9
    delay = make_returns({Role.VOCAL: [click]}, SR, bpm=120)["Delay Return.wav"]
    first_echo = int(np.argmax(np.abs(delay[0, :SR])))
    assert first_echo / SR == pytest.approx(0.75 * 60 / 120, abs=0.01)


def test_returns_are_band_limited() -> None:
    returns = make_returns({Role.SYNTH: [sine(60, 2.0, 0.5) + sine(1000, 2.0, 0.5)]}, SR, bpm=None)
    verb = returns["Reverb Return.wav"]
    spectrum = np.abs(np.fft.rfft(verb[0]))
    freqs = np.fft.rfftfreq(verb.shape[1], 1 / SR)
    low = spectrum[(freqs > 50) & (freqs < 70)].max()
    mid = spectrum[(freqs > 950) & (freqs < 1050)].max()
    assert 20 * np.log10(low / mid) < -15
