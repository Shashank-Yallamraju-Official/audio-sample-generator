import numpy as np
import numpy.typing as npt
import pytest

from automix.eq import Role
from automix.gain import peak_dbfs, rms_dbfs
from automix.pipeline import estimate_bpm, ghost_kick, mix_stems
from automix.presets import Genre

from .conftest import SR, sine


def kicks(bpm: float, seconds: float = 4.0) -> npt.NDArray[np.float32]:
    out = np.zeros((2, int(SR * seconds)), dtype=np.float32)
    t = np.arange(int(0.12 * SR)) / SR
    hit = (0.9 * np.sin(2 * np.pi * 55 * t) * np.exp(-t * 35)).astype(np.float32)
    step = 60 / bpm
    for k in range(int(seconds / step)):
        start = int(k * step * SR)
        end = min(start + len(hit), out.shape[1])
        out[:, start:end] += hit[: end - start]
    return out


def stems(bpm: float = 128) -> dict[str, npt.NDArray[np.float32]]:
    return {
        "Kick.wav": kicks(bpm),
        "Bass.wav": sine(55, 4.0, 0.3),
        "Pad.wav": sine(440, 4.0, 0.05),
        "Lead Vox.wav": sine(880, 4.0, 0.2),
    }


@pytest.mark.parametrize("bpm", [90.0, 128.0, 140.0])
def test_estimate_bpm_from_kick_onsets(bpm: float) -> None:
    onsets = np.arange(0, 8 * SR, int(60 / bpm * SR), dtype=np.int64)
    assert estimate_bpm(onsets, SR) == pytest.approx(bpm, rel=0.01)


def test_estimate_bpm_folds_double_time_into_range() -> None:
    onsets = np.arange(0, 8 * SR, int(60 / 260 * SR), dtype=np.int64)  # 260 BPM sixteenths
    bpm = estimate_bpm(onsets, SR)
    assert bpm is not None
    assert 70 <= bpm <= 180


def test_estimate_bpm_needs_two_onsets() -> None:
    assert estimate_bpm(np.array([100], dtype=np.int64), SR) is None


def test_ghost_kick_clicks_at_each_onset_and_is_silent_elsewhere() -> None:
    onsets = np.array([0, SR], dtype=np.int64)
    click = ghost_kick(2 * SR, onsets, SR)
    assert click.shape == (2, 2 * SR)
    assert peak_dbfs(click[:, : SR // 100]) > -12
    assert peak_dbfs(click[:, SR // 2 : SR // 2 + SR // 100]) == -np.inf


def test_mix_assigns_roles_from_filenames() -> None:
    result = mix_stems(stems(), SR, Genre.DANCE)
    assert {name: s.role for name, s in result.stems.items()} == {
        "Kick.wav": Role.KICK,
        "Bass.wav": Role.BASS,
        "Pad.wav": Role.SYNTH,
        "Lead Vox.wav": Role.VOCAL,
    }


def test_mix_detects_tempo_when_not_given() -> None:
    assert mix_stems(stems(bpm=124), SR, Genre.DANCE).bpm == pytest.approx(124, rel=0.02)


def test_given_tempo_wins_over_detection() -> None:
    assert mix_stems(stems(bpm=124), SR, Genre.DANCE, bpm=100).bpm == 100


def test_bass_ducks_under_the_kick_but_vocal_does_not() -> None:
    result = mix_stems(stems(), SR, Genre.DANCE)
    assert result.stems["Bass.wav"].duck is not None
    assert result.stems["Lead Vox.wav"].duck is None
    bass = result.stems["Bass.wav"].audio
    on_kick = bass[:, 200:1000]  # just after the first kick
    between = bass[:, int(0.35 * SR) : int(0.4 * SR)]  # recovered, before the next
    assert rms_dbfs(on_kick) < rms_dbfs(between) - 4


def test_dry_stems_are_only_gain_staged() -> None:
    result = mix_stems(stems(), SR, Genre.POP)
    dry = result.stems["Pad.wav"].dry
    assert rms_dbfs(dry) == pytest.approx(-18.0, abs=0.1)


def test_rough_mix_peaks_at_mix_bus_headroom() -> None:
    result = mix_stems(stems(), SR, Genre.HIPHOP)
    assert peak_dbfs(result.rough_mix) == pytest.approx(-6.0, abs=0.01)


def test_stems_of_different_lengths_are_padded_to_the_longest() -> None:
    short = stems()
    short["Pad.wav"] = short["Pad.wav"][:, :SR]
    result = mix_stems(short, SR, Genre.POP)
    assert {s.audio.shape[1] for s in result.stems.values()} == {4 * SR}


def test_no_kick_means_no_ducking_and_no_tempo() -> None:
    no_kick = {k: v for k, v in stems().items() if k != "Kick.wav"}
    result = mix_stems(no_kick, SR, Genre.DANCE)
    assert result.bpm is None
    assert all(s.duck is None for s in result.stems.values())
    assert result.ghost_kick is None


def test_mix_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="no stems"):
        mix_stems({}, SR, Genre.POP)


def test_estimate_bpm_folds_half_time_into_range() -> None:
    onsets = np.arange(0, 16 * SR, int(60 / 45 * SR), dtype=np.int64)
    assert estimate_bpm(onsets, SR) == pytest.approx(90, rel=0.01)


def test_silent_stems_produce_a_silent_mix() -> None:
    result = mix_stems({"Pad.wav": np.zeros((2, SR), dtype=np.float32)}, SR, Genre.POP)
    assert peak_dbfs(result.rough_mix) == -np.inf
