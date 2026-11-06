"""Bus processing: routing, parallel drum compression, low-end mono and effect returns.

These are the moves that make separate stems sound like one record. Settings are deliberately
light; how wet, how glued and how wide stay decisions for the ear.
"""

from enum import StrEnum
from typing import Literal

import numpy as np
from pedalboard import Delay, Reverb
from scipy import signal

from automix._types import Audio
from automix.compression import CompSettings, compress
from automix.eq import Role, highpass
from automix.gain import rms_dbfs


class Bus(StrEnum):
    """Summing groups, matching a Logic Track Stack per group."""

    DRUMS = "Drums"
    BASS = "Bass"
    MUSIC = "Music"
    VOCALS = "Vocals"
    FX = "FX"


BUS_BY_ROLE: dict[Role, Bus] = {
    Role.KICK: Bus.DRUMS,
    Role.SNARE: Bus.DRUMS,
    Role.HATS: Bus.DRUMS,
    Role.BASS: Bus.BASS,
    Role.SYNTH: Bus.MUSIC,
    Role.OTHER: Bus.MUSIC,
    Role.VOCAL: Bus.VOCALS,
    Role.RETURN: Bus.FX,
}


def stereo_correlation(audio: Audio) -> float:
    """Return the left/right correlation: 1 is mono, 0 is unrelated, -1 is out of phase."""
    left, right = audio[0].astype(np.float64), audio[1].astype(np.float64)
    denom = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
    return float(np.sum(left * right) / denom) if denom > 0 else 0.0


def _linkwitz_riley(
    audio: Audio, sr: int, cutoff_hz: float, btype: Literal["lowpass", "highpass"]
) -> Audio:
    # Two cascaded 2nd-order Butterworths: a 4th-order Linkwitz-Riley band, whose low and high
    # halves sum back to a flat magnitude response.
    sos = signal.butter(2, cutoff_hz, btype=btype, fs=sr, output="sos")
    once = signal.sosfilt(sos, audio, axis=-1)
    return signal.sosfilt(sos, once, axis=-1).astype(np.float32)


def mono_low_end(audio: Audio, sr: int, crossover_hz: float = 120.0) -> Audio:
    """Sum everything below ``crossover_hz`` to mono, leaving the stereo image above it."""
    low = _linkwitz_riley(audio, sr, crossover_hz, "lowpass")
    high = _linkwitz_riley(audio, sr, crossover_hz, "highpass")
    low_mono = np.broadcast_to(low.mean(axis=0, keepdims=True), low.shape)
    return (low_mono + high).astype(np.float32)


# Heavy "New York" squash blended under the dry drums.
PARALLEL_SETTINGS = CompSettings(ratio=8.0, attack_ms=1.0, release_ms=80.0, target_gr_db=10.0)


def parallel_compress(audio: Audio, sr: int, blend: float) -> Audio:
    """Blend a heavily compressed copy under ``audio``, then restore the original level."""
    if blend <= 0 or not np.isfinite(rms_dbfs(audio)):
        return audio.copy()
    wet = compress(audio, sr, PARALLEL_SETTINGS).audio
    mixed = audio + np.float32(blend) * wet
    match = 10 ** ((rms_dbfs(audio) - rms_dbfs(mixed)) / 20)
    return (mixed * np.float32(match)).astype(np.float32)


# Send levels per source role, in dB. Low on purpose: depth is set by ear.
REVERB_SENDS_DB: dict[Role, float] = {Role.SNARE: -18.0, Role.SYNTH: -16.0, Role.VOCAL: -18.0}
DELAY_SENDS_DB: dict[Role, float] = {Role.VOCAL: -20.0, Role.SYNTH: -26.0}
RETURN_BAND_HZ = (250.0, 7000.0)
DOTTED_EIGHTH_BEATS = 0.75
_DEFAULT_DELAY_S = 0.375


def _send(sources: dict[Role, list[Audio]], levels: dict[Role, float]) -> Audio | None:
    feeds = [
        audio * np.float32(10 ** (db / 20))
        for role, db in levels.items()
        for audio in sources.get(role, [])
    ]
    return np.sum(feeds, axis=0).astype(np.float32) if feeds else None


def _band_limit(audio: Audio, sr: int) -> Audio:
    low, high = RETURN_BAND_HZ
    sos = signal.butter(2, high, btype="lowpass", fs=sr, output="sos")
    return signal.sosfilt(sos, highpass(audio, sr, low), axis=-1).astype(np.float32)


def make_returns(sources: dict[Role, list[Audio]], sr: int, bpm: float | None) -> dict[str, Audio]:
    """Render the reverb and delay returns fed by ``sources`` (role -> processed stems)."""
    returns: dict[str, Audio] = {}
    verb_send = _send(sources, REVERB_SENDS_DB)
    if verb_send is not None:
        verb = Reverb(room_size=0.6, damping=0.5, wet_level=1.0, dry_level=0.0, width=1.0)
        returns["Reverb Return.wav"] = _band_limit(np.asarray(verb(verb_send, sr)), sr)
    delay_send = _send(sources, DELAY_SENDS_DB)
    if delay_send is not None:
        seconds = _DEFAULT_DELAY_S if bpm is None else DOTTED_EIGHTH_BEATS * 60 / bpm
        delay = Delay(delay_seconds=seconds, feedback=0.3, mix=1.0)
        returns["Delay Return.wav"] = _band_limit(np.asarray(delay(delay_send, sr)), sr)
    return returns
