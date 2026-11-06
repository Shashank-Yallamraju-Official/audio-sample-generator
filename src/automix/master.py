"""Loudness metering and a first-pass master.

The master is a starting point to compare against Logic's Mastering Assistant or your own
chain: optional reference-track tone matching, gain to a loudness target, and a limiter held to
a true-peak ceiling. Limiting is capped, so a dynamic mix lands under target rather than
being crushed; the report says so.
"""

from dataclasses import dataclass, field

import numpy as np
import pyloudnorm as pyln
from pedalboard import Limiter
from scipy import signal

from automix._types import Audio
from automix.gain import peak_dbfs

MAX_LIMITING_DB = 6.0
MAX_MATCH_DB = 4.0
MATCH_STRENGTH = 0.5  # move halfway toward the reference: a nudge, not a clone
_OVERSAMPLE = 4
_GAIN_PASSES = 8
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
    """Measure integrated loudness (ITU-R BS.1770), 4x-oversampled true peak and crest."""
    peak = peak_dbfs(audio)
    if not np.isfinite(peak):
        return Meter(-np.inf, -np.inf, 0.0)
    lufs = float(pyln.Meter(sr).integrated_loudness(audio.T.astype(np.float64)))
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


def _limit(audio: Audio, sr: int, gain_db: float, ceiling_dbtp: float) -> Audio:
    boosted = audio * np.float32(10 ** (gain_db / 20))
    limiter = Limiter(threshold_db=ceiling_dbtp - 0.3, release_ms=100.0)
    return np.asarray(limiter(boosted, sr), dtype=np.float32)


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
        audio, match = _match_eq(mix, reference, sr)

    # Limiting depth is how far the boosted sample peak rises above the ceiling.
    max_gain = MAX_LIMITING_DB + ceiling_dbtp - peak_dbfs(audio)
    gain = target_lufs - measure(audio, sr).lufs
    # Limiting lowers loudness, so re-aim a few times; it converges in two or three passes.
    for _ in range(_GAIN_PASSES):  # pragma: no branch
        gain = min(gain, max_gain)
        out = _limit(audio, sr, gain, ceiling_dbtp)
        error = target_lufs - measure(out, sr).lufs
        if abs(error) < 0.1 or (error > 0 and gain >= max_gain):
            break
        gain += error
    capped = gain >= max_gain - 1e-9 and measure(out, sr).lufs < target_lufs - 0.5

    over = measure(out, sr).true_peak_dbtp - ceiling_dbtp
    if over > 0:
        out = (out * np.float32(10 ** (-(over + 0.01) / 20))).astype(np.float32)
    limiting = max(0.0, peak_dbfs(audio) + gain - ceiling_dbtp)
    return MasterResult(
        audio=out,
        before=before,
        after=measure(out, sr),
        target_lufs=target_lufs,
        gain_db=gain,
        limiting_db=limiting,
        capped=capped,
        match_eq_db=match,
    )
