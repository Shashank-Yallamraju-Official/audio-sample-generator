"""The first-pass mix: every automated stage applied to a set of stems, in order.

Per stem: gain staging, role high-pass, resonance cuts, compression, then sidechain ducking
from the kick. The stems are summed into a rough mix with mix-bus headroom.
"""

from dataclasses import dataclass

import numpy as np

from automix._types import Audio, SampleIndices
from automix.compression import CompResult, CompSettings, compress
from automix.eq import HIGHPASS_HZ, Resonance, Role, apply_cuts, find_resonances, highpass
from automix.eq import role_from_filename as _role
from automix.gain import gain_stage, peak_dbfs
from automix.presets import Genre, comp_settings, duck_settings
from automix.sidechain import DuckSettings, apply_ducking, detect_onsets, duck_curve

MIX_BUS_PEAK_DBFS = -6.0
TEMPO_RANGE_BPM = (70.0, 180.0)
# Tonal parts have strong spectral peaks that are notes, not resonances, so they skip
# automatic notching.
_NO_RESONANCE_CUTS = frozenset({Role.BASS, Role.SYNTH})
_MAX_CUT_DB = 6.0


@dataclass(frozen=True)
class _Context:
    """Song-wide facts every stem's processing needs."""

    sr: int
    genre: Genre
    onsets: SampleIndices
    bpm: float | None


@dataclass(frozen=True)
class StemResult:
    """One stem after processing, plus every setting that produced it."""

    role: Role
    audio: Audio
    dry: Audio
    gain_db: float
    highpass_hz: float | None
    cuts: tuple[Resonance, ...]
    comp_settings: CompSettings
    comp: CompResult
    duck: DuckSettings | None


@dataclass(frozen=True)
class MixResult:
    """Everything the Logic handoff needs."""

    stems: dict[str, StemResult]
    sr: int
    bpm: float | None
    ghost_kick: Audio | None
    rough_mix: Audio


def estimate_bpm(onsets: SampleIndices, sr: int) -> float | None:
    """Estimate tempo from kick onsets via the median gap, folded into 70-180 BPM."""
    if len(onsets) < 2:
        return None
    bpm = 60 * sr / float(np.median(np.diff(onsets)))
    low, high = TEMPO_RANGE_BPM
    while bpm > high:
        bpm /= 2
    while bpm < low:
        bpm *= 2
    return bpm


def ghost_kick(n_samples: int, onsets: SampleIndices, sr: int) -> Audio:
    """Render a short 1 kHz click at each onset: a clean sidechain key for Logic."""
    t = np.arange(int(0.005 * sr)) / sr
    click = (0.5 * np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 600)).astype(np.float32)
    out = np.zeros((2, n_samples), dtype=np.float32)
    for onset in onsets:
        start = int(onset)
        end = min(start + len(click), n_samples)
        out[:, start:end] = click[: end - start]
    return out


def _pad(audio: Audio, length: int) -> Audio:
    return np.pad(audio, ((0, 0), (0, length - audio.shape[1]))).astype(np.float32)


def mix_stems(
    stems: dict[str, Audio], sr: int, genre: Genre, bpm: float | None = None
) -> MixResult:
    """Run the full first-pass mix on ``stems`` (filename -> audio)."""
    if not stems:
        msg = "no stems to mix"
        raise ValueError(msg)
    length = max(a.shape[1] for a in stems.values())
    padded = {name: _pad(audio, length) for name, audio in stems.items()}
    roles = {name: _role(name) for name in padded}

    kick_names = [n for n, r in roles.items() if r is Role.KICK]
    onsets = (
        detect_onsets(np.sum([padded[n] for n in kick_names], axis=0), sr)
        if kick_names
        else np.array([], dtype=np.int64)
    )
    if bpm is None:
        bpm = estimate_bpm(onsets, sr)

    ctx = _Context(sr, genre, onsets, bpm)
    results = {name: _process(audio, roles[name], ctx) for name, audio in padded.items()}
    mix = np.sum([r.audio for r in results.values()], axis=0).astype(np.float32)
    peak = peak_dbfs(mix)
    if np.isfinite(peak):
        mix = (mix * np.float32(10 ** ((MIX_BUS_PEAK_DBFS - peak) / 20))).astype(np.float32)
    return MixResult(
        stems=results,
        sr=sr,
        bpm=bpm,
        ghost_kick=ghost_kick(length, onsets, sr) if len(onsets) else None,
        rough_mix=mix,
    )


def _process(audio: Audio, role: Role, ctx: _Context) -> StemResult:
    sr, onsets, bpm = ctx.sr, ctx.onsets, ctx.bpm
    staged = gain_stage(audio)
    cutoff = HIGHPASS_HZ[role]
    out = highpass(staged.audio, sr, cutoff)

    found = [] if role in _NO_RESONANCE_CUTS else find_resonances(out, sr)
    cuts = tuple(Resonance(r.freq_hz, min(r.excess_db, _MAX_CUT_DB)) for r in found)
    out = apply_cuts(out, sr, list(cuts), max_cut_db=_MAX_CUT_DB)

    settings = comp_settings(ctx.genre, role)
    comp = compress(out, sr, settings)
    out = comp.audio

    duck = duck_settings(ctx.genre, role, bpm) if bpm is not None and len(onsets) else None
    if duck is not None:
        out = apply_ducking(out, duck_curve(out.shape[1], onsets, sr, duck))

    return StemResult(
        role=role,
        audio=out,
        dry=staged.audio,
        gain_db=staged.gain_db,
        highpass_hz=cutoff,
        cuts=cuts,
        comp_settings=settings,
        comp=comp,
        duck=duck,
    )
