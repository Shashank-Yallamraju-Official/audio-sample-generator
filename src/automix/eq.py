"""Subtractive cleanup EQ: role-based high-pass filters and automatic resonance cuts.

Only cuts are automated. Tone-shaping boosts are left for mixing by ear.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from pedalboard import PeakFilter
from pedalboard._pedalboard import Pedalboard  # public name, but stubs only export it here
from scipy import ndimage, signal

from automix._types import Audio


class Role(StrEnum):
    """What a stem is, which decides its default processing."""

    KICK = "kick"
    BASS = "bass"
    SNARE = "snare"
    HATS = "hats"
    SYNTH = "synth"
    VOCAL = "vocal"
    RETURN = "return"
    OTHER = "other"


# High-pass cutoffs per role, from the best-practices doc. ``None`` means no filter.
HIGHPASS_HZ: dict[Role, float | None] = {
    Role.KICK: 30.0,
    Role.BASS: 30.0,
    Role.SNARE: 100.0,
    Role.HATS: 300.0,
    Role.SYNTH: 150.0,
    Role.VOCAL: 90.0,
    Role.RETURN: 250.0,
    Role.OTHER: None,
}

# Checked in order: the first matching role wins, so "vocal reverb" is a return, "808 kick"
# is a bass, and "kick sub" or "bass drum" is a kick. Short keywords only count at the start
# of a word, so "trim" is not a rim and "what" is not a hat.
_ROLE_PATTERNS: list[tuple[Role, re.Pattern[str]]] = [
    (Role.RETURN, re.compile(r"reverb|verb|delay|return|\bfx\b")),
    (Role.BASS, re.compile(r"808")),
    (Role.KICK, re.compile(r"kick|\bbd\d*\b|\bkik\b|bass ?drum")),
    (Role.BASS, re.compile(r"bass|\bsub\b")),
    (Role.SNARE, re.compile(r"snare|clap|\bsd\d*\b|\brim")),
    (
        Role.HATS,
        re.compile(
            r"(?:\b|hi ?|open ?|closed? ?)hat|\bhh\d*\b|cymbal|\bride|crash|perc|shaker|tamb"
        ),
    ),
    (Role.VOCAL, re.compile(r"vox|vocal|voice|\bvo\b|adlib")),
    (Role.SYNTH, re.compile(r"synth|pad|keys|piano|chord|lead|pluck|arp|guitar|string")),
]


def role_from_filename(name: str) -> Role:
    """Guess a stem's role from its filename; unknown names map to ``Role.OTHER``."""
    stem = re.sub(r"[_\-.]+", " ", name.lower())
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(stem):
            return role
    return Role.OTHER


def highpass(audio: Audio, sr: int, cutoff_hz: float | None) -> Audio:
    """Apply a 12 dB/octave Butterworth high-pass; ``None`` returns an unchanged copy."""
    if cutoff_hz is None:
        return audio.copy()
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfilt(sos, audio, axis=-1).astype(np.float32)


@dataclass(frozen=True)
class Resonance:
    """A narrow spectral peak standing ``excess_db`` above its surroundings."""

    freq_hz: float
    excess_db: float


_POINTS_PER_OCTAVE = 48
_SMOOTHING_OCTAVES = 2 / 3


def find_resonances(
    audio: Audio,
    sr: int,
    threshold_db: float = 8.0,
    max_cuts: int = 3,
    freq_range_hz: tuple[float, float] = (60.0, 12_000.0),
) -> list[Resonance]:
    """Find peaks in the long-term spectrum that stand out from a smoothed version of it.

    Returns at most ``max_cuts`` resonances, strongest first.
    """
    freqs, psd = signal.welch(audio, fs=sr, nperseg=8192, axis=-1)
    power = np.mean(np.atleast_2d(psd), axis=0)
    if not np.any(power > 0):
        return []

    low, high = freq_range_hz[0], min(freq_range_hz[1], 0.45 * sr)
    n_points = int(np.log2(high / low) * _POINTS_PER_OCTAVE) + 1
    grid = np.geomspace(low, high, n_points)
    level_db = 10 * np.log10(np.interp(grid, freqs, power) + np.finfo(np.float64).tiny)
    smooth_db = ndimage.median_filter(
        level_db, size=int(_POINTS_PER_OCTAVE * _SMOOTHING_OCTAVES), mode="nearest"
    )
    excess = level_db - smooth_db

    peaks, props = signal.find_peaks(excess, height=threshold_db, distance=_POINTS_PER_OCTAVE // 6)
    strongest = np.argsort(props["peak_heights"])[::-1][:max_cuts]
    return [Resonance(float(grid[peaks[i]]), float(excess[peaks[i]])) for i in strongest]


def cut_db(resonance: Resonance, max_cut_db: float = 6.0) -> float:
    """How deep to cut a resonance: half its excess, so it is tamed rather than flattened."""
    return min(resonance.excess_db / 2, max_cut_db)


def apply_cuts(
    audio: Audio,
    sr: int,
    resonances: list[Resonance],
    q: float = 6.0,
    max_cut_db: float = 6.0,
) -> Audio:
    """Notch each resonance with a narrow bell, cutting half its excess up to the cap."""
    if not resonances:
        return audio.copy()
    board = Pedalboard(
        [
            PeakFilter(cutoff_frequency_hz=r.freq_hz, gain_db=-cut_db(r, max_cut_db), q=q)
            for r in resonances
        ]
    )
    return np.asarray(board(audio, sr), dtype=np.float32)
