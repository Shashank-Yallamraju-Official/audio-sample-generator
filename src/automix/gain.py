"""Level measurement and gain staging."""

from dataclasses import dataclass

import numpy as np

from automix._types import Audio


def _to_db(linear: float) -> float:
    return float(20 * np.log10(linear)) if linear > 0 else -np.inf


def rms_dbfs(audio: Audio) -> float:
    """Return the RMS level of ``audio`` across all channels, in dBFS (sine reference)."""
    return _to_db(float(np.sqrt(np.mean(np.square(audio, dtype=np.float64)))))


def peak_dbfs(audio: Audio) -> float:
    """Return the sample-peak level of ``audio``, in dBFS."""
    return _to_db(float(np.max(np.abs(audio)))) if audio.size else -np.inf


@dataclass(frozen=True)
class GainResult:
    """Outcome of :func:`gain_stage`."""

    audio: Audio
    gain_db: float
    peak_limited: bool


def gain_stage(
    audio: Audio, target_rms_dbfs: float = -18.0, max_peak_dbfs: float = -6.0
) -> GainResult:
    """Scale ``audio`` so its RMS hits ``target_rms_dbfs`` without peaks above ``max_peak_dbfs``.

    The peak ceiling wins over the RMS target, so spiky material (drums) lands below target
    rather than being pushed into clipping. Silence is returned unchanged with 0 dB gain.
    """
    rms = rms_dbfs(audio)
    if not np.isfinite(rms):
        return GainResult(audio.copy(), 0.0, peak_limited=False)
    gain_db = target_rms_dbfs - rms
    headroom_db = max_peak_dbfs - peak_dbfs(audio)
    peak_limited = gain_db > headroom_db
    gain_db = min(gain_db, headroom_db)
    scaled = (audio * np.float32(10 ** (gain_db / 20))).astype(np.float32)
    return GainResult(scaled, gain_db, peak_limited)
