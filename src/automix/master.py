"""Loudness metering and a first-pass master.

The master is a starting point to compare against Logic's Mastering Assistant or your own
chain: optional reference-track tone matching, gain to a loudness target, and a look-ahead
limiter held to a true-peak ceiling. Limiting is capped, so a dynamic mix lands under target
rather than being crushed; the report says so.
"""

from dataclasses import dataclass, field

import numpy as np
import pyloudnorm as pyln
from scipy import ndimage, signal

from automix._types import Audio
from automix.gain import peak_dbfs

MAX_LIMITING_DB = 6.0
MAX_MATCH_DB = 4.0
MATCH_STRENGTH = 0.5  # move halfway toward the reference: a nudge, not a clone
_OVERSAMPLE = 4
_GAIN_PASSES = 8
_MIN_LOUDNESS_SECONDS = 0.4  # one BS.1770 gating block; shorter audio has no loudness
_LOOKAHEAD_SECONDS = 0.005
_RELEASE_DB_PER_SECOND = 60.0
# The limiter aims this far under the ceiling, so the smoothed gain never lets a peak through.
_CEILING_MARGIN_DB = 0.1
# Octave bands for reference matching, as (centre Hz) of each band.
MATCH_BANDS_HZ = (63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0)


@dataclass(frozen=True)
class Meter:
    """Loudness facts about a piece of audio."""

    lufs: float
    true_peak_dbtp: float
    crest_db: float


def _true_peak(audio: Audio) -> float:
    up = signal.resample_poly(audio, _OVERSAMPLE, 1, axis=-1)
    return peak_dbfs(np.asarray(up, dtype=np.float32))


def measure(audio: Audio, sr: int) -> Meter:
    """Measure integrated loudness (ITU-R BS.1770), 4x-oversampled true peak and crest.

    Audio shorter than one 400 ms gating block has no defined loudness and reads ``-inf`` LUFS.
    """
    peak = peak_dbfs(audio)
    if not np.isfinite(peak):
        return Meter(-np.inf, -np.inf, 0.0)
    lufs = (
        float(pyln.Meter(sr).integrated_loudness(audio.T.astype(np.float64)))
        if audio.shape[1] >= _MIN_LOUDNESS_SECONDS * sr
        else -np.inf
    )
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    return Meter(lufs, _true_peak(audio), peak - float(20 * np.log10(rms)))


@dataclass(frozen=True)
class MasterResult:
    """Outcome of :func:`master`."""

    audio: Audio
    before: Meter
    after: Meter
    target_lufs: float
    gain_db: float
    limiting_db: float
    capped: bool
    match_eq_db: dict[float, float] = field(default_factory=dict)


def _band_levels(audio: Audio, sr: int) -> np.ndarray:
    freqs, psd = signal.welch(audio, fs=sr, nperseg=8192, axis=-1)
    power = np.mean(np.atleast_2d(psd), axis=0)
    levels = []
    for centre in MATCH_BANDS_HZ:
        band = (freqs >= centre / np.sqrt(2)) & (freqs < centre * np.sqrt(2))
        levels.append(10 * np.log10(power[band].sum() + 1e-20))
    return np.array(levels)


def _match_eq(audio: Audio, reference: Audio, sr: int) -> tuple[Audio, dict[float, float]]:
    """Nudge the mix's octave-band balance toward the reference with a linear-phase FIR."""
    diff = _band_levels(reference, sr) - _band_levels(audio, sr)
    diff -= np.median(diff)  # tone only; loudness is handled by the gain stage
    gains_db = np.clip(diff * MATCH_STRENGTH, -MAX_MATCH_DB, MAX_MATCH_DB)
    nyquist = sr / 2
    points = [f for f in MATCH_BANDS_HZ if f < nyquist]
    freq = np.concatenate([[0.0], points, [nyquist]]) / nyquist
    gain = 10 ** (
        np.concatenate([[gains_db[0]], gains_db[: len(points)], [gains_db[len(points) - 1]]]) / 20
    )
    taps = signal.firwin2(4097, freq, gain)
    filtered = signal.fftconvolve(audio, taps[np.newaxis, :], mode="full", axes=-1)
    delay = (len(taps) - 1) // 2
    out = filtered[:, delay : delay + audio.shape[1]].astype(np.float32)
    return out, {f: float(g) for f, g in zip(points, gains_db, strict=False)}


def limit(audio: Audio, sr: int, ceiling_dbtp: float = -1.0) -> tuple[Audio, float]:
    """Hold ``audio`` under ``ceiling_dbtp``; return it with the deepest gain reduction in dB.

    A look-ahead brickwall limiter: the reduction each sample needs comes from the 4x-oversampled
    peak of the loudest channel (so the stereo image holds still), is reached 5 ms early, held
    across the peak, released at 60 dB per second and smoothed. Audio under the ceiling is
    returned unchanged; nothing is ever clipped.
    """
    up = np.abs(signal.resample_poly(audio, _OVERSAMPLE, 1, axis=-1))
    peaks = up.max(axis=0).reshape(-1, _OVERSAMPLE).max(axis=1)
    ceiling = 10 ** ((ceiling_dbtp - _CEILING_MARGIN_DB) / 20)
    needed_db = 20 * np.log10(np.maximum(peaks / ceiling, 1.0))
    if not needed_db.any():
        return audio.copy(), 0.0

    lookahead = max(int(_LOOKAHEAD_SECONDS * sr), 1)
    held = ndimage.maximum_filter1d(needed_db, size=2 * lookahead + 1, mode="constant")
    # Linear-in-dB release, vectorised: the running maximum of a ramped copy, un-ramped.
    ramp = (_RELEASE_DB_PER_SECOND / sr) * np.arange(held.size)
    released = np.maximum.accumulate(held + ramp) - ramp
    reduction_db = ndimage.uniform_filter1d(released, size=lookahead + 1, mode="nearest")
    limited = (audio * 10 ** (-reduction_db / 20)).astype(np.float32)
    return limited, float(reduction_db.max())


def master(
    mix: Audio,
    sr: int,
    target_lufs: float,
    ceiling_dbtp: float = -1.0,
    reference: Audio | None = None,
) -> MasterResult:
    """Master ``mix`` to ``target_lufs`` (or the reference's loudness) under ``ceiling_dbtp``."""
    before = measure(mix, sr)
    if not np.isfinite(before.lufs):
        return MasterResult(mix.copy(), before, before, target_lufs, 0.0, 0.0, capped=False)

    audio: Audio = mix
    match: dict[float, float] = {}
    if reference is not None:
        target_lufs = measure(reference, sr).lufs
        if not np.isfinite(target_lufs):
            msg = "reference is silent or shorter than 0.4 s, so it has no loudness to match"
            raise ValueError(msg)
        audio, match = _match_eq(mix, reference, sr)

    # Limiting depth is how far the boosted true peak rises above the limiter's ceiling.
    unlimited = measure(audio, sr)
    max_gain = MAX_LIMITING_DB + ceiling_dbtp - _CEILING_MARGIN_DB - unlimited.true_peak_dbtp
    gain = min(target_lufs - unlimited.lufs, max_gain)
    # Limiting lowers loudness, so re-aim a few times; it converges in two or three passes.
    for attempt in range(_GAIN_PASSES):  # pragma: no branch
        out, limiting = limit(audio * np.float32(10 ** (gain / 20)), sr, ceiling_dbtp)
        after = measure(out, sr)
        step = min(gain + target_lufs - after.lufs, max_gain) - gain
        if abs(step) < 0.05 or attempt == _GAIN_PASSES - 1:
            break
        gain += step
    return MasterResult(
        audio=out,
        before=before,
        after=after,
        target_lufs=target_lufs,
        gain_db=gain,
        limiting_db=limiting,
        capped=after.lufs < target_lufs - 0.5,
        match_eq_db=match,
    )
