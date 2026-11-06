import numpy as np
import pytest

from automix.eq import (
    HIGHPASS_HZ,
    Resonance,
    Role,
    apply_cuts,
    find_resonances,
    highpass,
    role_from_filename,
)
from automix.gain import rms_dbfs

from .conftest import SR, sine


@pytest.mark.parametrize(
    ("name", "role"),
    [
        ("Kick.wav", Role.KICK),
        ("01_BD_main.wav", Role.KICK),
        ("808 Bass.wav", Role.BASS),
        ("sub_bass.aif", Role.BASS),
        ("Snare.wav", Role.SNARE),
        ("clap_layer.wav", Role.SNARE),
        ("HiHat open.wav", Role.HATS),
        ("hh_closed.wav", Role.HATS),
        ("perc_shaker.wav", Role.HATS),
        ("Lead Vox.wav", Role.VOCAL),
        ("backing_vocals.wav", Role.VOCAL),
        ("Pad Wide.wav", Role.SYNTH),
        ("keys.wav", Role.SYNTH),
        ("Reverb Return.wav", Role.RETURN),
        ("mystery.wav", Role.OTHER),
    ],
)
def test_role_from_filename(name: str, role: Role) -> None:
    assert role_from_filename(name) == role


def test_bass_wins_over_kick_keyword_in_808() -> None:
    # "808" is a bass in hip-hop naming even when "kick" isn't mentioned.
    assert role_from_filename("808.wav") == Role.BASS


def test_every_role_has_a_highpass_entry() -> None:
    assert set(HIGHPASS_HZ) == set(Role)


def test_highpass_cuts_below_cutoff_at_12_db_per_octave() -> None:
    # Two octaves below a 100 Hz cutoff, a 2nd-order filter is down about 24 dB.
    rumble = sine(25, 2.0, 0.5)
    out = highpass(rumble, SR, 100.0)
    assert rms_dbfs(out) - rms_dbfs(rumble) < -20.0


def test_highpass_passes_content_well_above_cutoff() -> None:
    tone = sine(1000, 1.0, 0.5)
    assert rms_dbfs(highpass(tone, SR, 100.0)) == pytest.approx(rms_dbfs(tone), abs=0.1)


def test_highpass_with_no_cutoff_returns_an_unchanged_copy() -> None:
    tone = sine(50, 0.1, 0.5)
    out = highpass(tone, SR, None)
    np.testing.assert_array_equal(out, tone)
    assert out is not tone


def _noise_with_tone(tone_hz: float, tone_amp: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    noise = (0.05 * rng.standard_normal((2, SR * 2))).astype(np.float32)
    return noise + sine(tone_hz, 2.0, tone_amp)


def test_find_resonances_locates_a_ringing_tone() -> None:
    found = find_resonances(_noise_with_tone(1000, 0.3), SR)
    assert len(found) == 1
    assert found[0].freq_hz == pytest.approx(1000, rel=0.03)
    assert found[0].excess_db > 6.0


def test_find_resonances_ignores_flat_noise() -> None:
    rng = np.random.default_rng(1)
    noise = (0.1 * rng.standard_normal((2, SR * 2))).astype(np.float32)
    assert find_resonances(noise, SR) == []


def test_find_resonances_on_silence_is_empty() -> None:
    assert find_resonances(np.zeros((2, SR), dtype=np.float32), SR) == []


def test_find_resonances_caps_the_number_of_cuts() -> None:
    audio = _noise_with_tone(500, 0.3) + _noise_with_tone(2000, 0.3) + _noise_with_tone(5000, 0.3)
    assert len(find_resonances(audio, SR, max_cuts=2)) == 2


def test_apply_cuts_attenuates_the_resonance_by_up_to_the_cap() -> None:
    tone = sine(1000, 1.0, 0.5)
    out = apply_cuts(tone, SR, [Resonance(1000.0, 20.0)], max_cut_db=6.0)
    assert rms_dbfs(out) - rms_dbfs(tone) == pytest.approx(-6.0, abs=0.5)


def test_apply_cuts_leaves_distant_frequencies_alone() -> None:
    low = sine(200, 1.0, 0.5)
    out = apply_cuts(low, SR, [Resonance(2000.0, 6.0)])
    assert rms_dbfs(out) == pytest.approx(rms_dbfs(low), abs=0.5)


def test_apply_cuts_with_no_resonances_is_a_copy() -> None:
    tone = sine(1000, 0.1, 0.5)
    out = apply_cuts(tone, SR, [])
    np.testing.assert_array_equal(out, tone)
    assert out is not tone
