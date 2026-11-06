"""Compression with thresholds fitted to a target gain reduction.

Setting a threshold by hand until the meter reads "about 4 dB" is the tedious part of
compression, so this module does it by bisection. Attack, release and ratio come from the
role defaults in the best-practices doc; they are the settings to adjust by ear.
"""

import math
from dataclasses import dataclass

import numpy as np
from pedalboard import Compressor

from automix._types import Audio
from automix.eq import Role
from automix.gain import rms_dbfs

MAX_GR_PER_STAGE_DB = 6.0
_SEARCH_RANGE_DB = (-80.0, 0.0)
_SEARCH_STEPS = 20  # 80 dB / 2**20: far finer than any audible difference
_DETECTOR_FLOOR = np.float32(1e-9)  # keeps the gain defined through digital silence


@dataclass(frozen=True)
class CompSettings:
    """A compressor's character plus how hard it should work (average gain reduction)."""

    ratio: float
    attack_ms: float
    release_ms: float
    target_gr_db: float


COMP_BY_ROLE: dict[Role, CompSettings] = {
    Role.KICK: CompSettings(ratio=4.0, attack_ms=20.0, release_ms=75.0, target_gr_db=4.0),
    Role.SNARE: CompSettings(ratio=4.0, attack_ms=10.0, release_ms=115.0, target_gr_db=4.5),
    Role.BASS: CompSettings(ratio=4.0, attack_ms=20.0, release_ms=150.0, target_gr_db=5.0),
    Role.HATS: CompSettings(ratio=3.0, attack_ms=5.0, release_ms=50.0, target_gr_db=2.0),
    Role.VOCAL: CompSettings(ratio=4.0, attack_ms=5.0, release_ms=75.0, target_gr_db=6.0),
    Role.SYNTH: CompSettings(ratio=2.5, attack_ms=20.0, release_ms=150.0, target_gr_db=3.0),
    Role.RETURN: CompSettings(ratio=1.0, attack_ms=10.0, release_ms=100.0, target_gr_db=0.0),
    Role.OTHER: CompSettings(ratio=2.0, attack_ms=20.0, release_ms=150.0, target_gr_db=2.0),
}


def _compressor(threshold_db: float, settings: CompSettings) -> Compressor:
    return Compressor(
        threshold_db=threshold_db,
        ratio=settings.ratio,
        attack_ms=settings.attack_ms,
        release_ms=settings.release_ms,
    )


def _run(audio: Audio, sr: int, threshold_db: float, settings: CompSettings) -> Audio:
    # pedalboard's Compressor follows each channel on its own, which shifts the stereo image
    # whenever one side is louder. So run it on one detector channel (the louder side of each
    # sample) and apply the gain it chose to every channel.
    detector = np.maximum(np.max(np.abs(audio), axis=0, keepdims=True), _DETECTOR_FLOOR)
    squeezed = np.asarray(_compressor(threshold_db, settings)(detector, sr), dtype=np.float32)
    return (audio * (squeezed / detector)).astype(np.float32)


def measure_gr_db(audio: Audio, sr: int, threshold_db: float, settings: CompSettings) -> float:
    """Return the average gain reduction (RMS in minus RMS out) at ``threshold_db``."""
    return rms_dbfs(audio) - rms_dbfs(_run(audio, sr, threshold_db, settings))


def fit_threshold(audio: Audio, sr: int, settings: CompSettings) -> float:
    """Find the threshold at which the compressor averages ``settings.target_gr_db`` of GR."""
    low, high = _SEARCH_RANGE_DB
    for _ in range(_SEARCH_STEPS):
        mid = (low + high) / 2
        if measure_gr_db(audio, sr, mid, settings) > settings.target_gr_db:
            low = mid  # too much reduction: raise the threshold
        else:
            high = mid
    return (low + high) / 2


@dataclass(frozen=True)
class Stage:
    """One compressor in the chain, as configured."""

    threshold_db: float
    gr_db: float


@dataclass(frozen=True)
class CompResult:
    """Outcome of :func:`compress`. ``gr_db`` is the total before make-up gain."""

    audio: Audio
    stages: tuple[Stage, ...]
    gr_db: float
    makeup_db: float


def compress(audio: Audio, sr: int, settings: CompSettings) -> CompResult:
    """Compress to the target GR, then apply make-up gain so loudness is unchanged.

    Targets above :data:`MAX_GR_PER_STAGE_DB` are split evenly across serial stages, which
    sounds cleaner than one compressor working hard.
    """
    if settings.target_gr_db <= 0 or not np.isfinite(rms_dbfs(audio)):
        return CompResult(audio.copy(), (), 0.0, 0.0)

    n_stages = math.ceil(settings.target_gr_db / MAX_GR_PER_STAGE_DB)
    per_stage = CompSettings(
        settings.ratio, settings.attack_ms, settings.release_ms, settings.target_gr_db / n_stages
    )
    out = audio
    stages: list[Stage] = []
    for _ in range(n_stages):
        threshold = fit_threshold(out, sr, per_stage)
        compressed = _run(out, sr, threshold, per_stage)
        stages.append(Stage(threshold, rms_dbfs(out) - rms_dbfs(compressed)))
        out = compressed

    gr_db = rms_dbfs(audio) - rms_dbfs(out)
    makeup = (out * np.float32(10 ** (gr_db / 20))).astype(np.float32)
    return CompResult(makeup, tuple(stages), gr_db, gr_db)
