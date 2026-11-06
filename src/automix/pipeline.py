"""The first-pass mix: every automated stage applied to a set of stems, in order.

Per stem: gain staging, role high-pass, resonance cuts, compression, then sidechain ducking
from the kick. The stems are summed into a rough mix with mix-bus headroom.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage, signal

from automix._types import Audio, SampleIndices
from automix.buses import (
    BUS_BY_ROLE,
    Bus,
    make_returns,
    mono_low_end,
    parallel_compress,
    stereo_correlation,
)
from automix.compression import CompResult, CompSettings, compress
from automix.eq import HIGHPASS_HZ, Resonance, Role, apply_cuts, find_resonances, highpass
from automix.eq import role_from_filename as _role
from automix.gain import gain_stage, peak_dbfs, rms_dbfs
from automix.presets import PRESETS, Genre, comp_settings, duck_settings
from automix.sidechain import DuckSettings, apply_ducking, detect_onsets, duck_curve

MIX_BUS_PEAK_DBFS = -6.0
TEMPO_RANGE_BPM = (70.0, 180.0)
# Tonal parts have strong spectral peaks that are notes, not resonances, so they skip
# automatic notching.
_NO_RESONANCE_CUTS = frozenset({Role.BASS, Role.SYNTH})
_MAX_CUT_DB = 6.0
# Starting fader balance: each processed stem's RMS level, so the rough mix is already roughly
# balanced (a level, not a loudness standard; tune by ear in Logic).
FADER_RMS_DBFS: dict[Role, float] = {
    Role.KICK: -20.0,
    Role.BASS: -20.0,
    Role.SNARE: -22.0,
    Role.HATS: -30.0,
    Role.SYNTH: -25.0,
    Role.VOCAL: -19.0,
    Role.RETURN: -30.0,
    Role.OTHER: -26.0,
}
_DRUM_ROLES = frozenset({Role.KICK, Role.SNARE, Role.HATS})
_TEMPO_PRIOR_BPM = {Genre.DANCE: 124.0, Genre.HIPHOP: 90.0, Genre.POP: 110.0}


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
    fader_db: float


@dataclass(frozen=True)
class MixResult:
    """Everything the Logic handoff needs."""

    stems: dict[str, StemResult]
    sr: int
    bpm: float | None
    ghost_kick: Audio | None
    rough_mix: Audio
    buses: dict[Bus, Audio]
    bus_comp: dict[Bus, CompResult]
    returns: dict[str, Audio]
    mix_comp: CompResult
    correlation: float


_ONSET_FPS = 100  # onset-train frame rate for tempo analysis
_PRIOR_OCTAVES = 0.5  # width of the tempo prior, in octaves (one standard deviation)


def estimate_bpm(onsets: SampleIndices, sr: int, prior_bpm: float = 110.0) -> float | None:
    """Estimate tempo from drum onsets by autocorrelating an onset train.

    Each candidate beat period is scored on the autocorrelation at that period plus half its
    double and half its half (so a groove that only repeats every two beats still counts),
    weighted by a log-normal prior around ``prior_bpm`` to settle half/double-time ties.
    """
    frames = np.round(np.asarray(onsets) * _ONSET_FPS / sr).astype(np.int64)
    if len(np.unique(frames)) < 2:
        return None
    train = np.zeros(int(frames.max()) + 1)
    np.add.at(train, frames, 1.0)
    train = ndimage.gaussian_filter1d(train, sigma=1.0)
    train -= train.mean()
    ac = signal.correlate(train, train, mode="full")[len(train) - 1 :]
    ac = np.clip(ac / ac[0], 0.0, None)

    def at(lag: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.asarray(np.interp(lag, np.arange(len(ac)), ac, right=0.0), dtype=np.float64)

    low, high = TEMPO_RANGE_BPM
    candidates = np.arange(low, high + 0.05, 0.1)
    lag = 60 * _ONSET_FPS / candidates
    prior = np.exp(-0.5 * (np.log2(candidates / prior_bpm) / _PRIOR_OCTAVES) ** 2)
    score = (at(lag) + 0.5 * at(2 * lag) + 0.5 * at(lag / 2)) * prior
    return float(candidates[int(np.argmax(score))])


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
        drum_names = [n for n, r in roles.items() if r in _DRUM_ROLES]
        drum_onsets = (
            np.concatenate([detect_onsets(padded[n], sr) for n in drum_names])
            if drum_names
            else np.array([], dtype=np.int64)
        )
        bpm = (
            estimate_bpm(np.sort(drum_onsets), sr, _TEMPO_PRIOR_BPM[genre]) if len(onsets) else None
        )

    ctx = _Context(sr, genre, onsets, bpm)
    results = {name: _process(audio, roles[name], ctx) for name, audio in padded.items()}

    returns = _returns(results, ctx)
    buses, bus_comp = _buses(results, returns, ctx)
    settings = PRESETS[genre].buses
    mix_comp = compress(
        np.sum(list(buses.values()), axis=0).astype(np.float32), sr, settings.mix_glue
    )
    mix = mono_low_end(mix_comp.audio, sr, settings.low_mono_hz)
    peak = peak_dbfs(mix)
    if np.isfinite(peak):
        mix = (mix * np.float32(10 ** ((MIX_BUS_PEAK_DBFS - peak) / 20))).astype(np.float32)
    return MixResult(
        stems=results,
        sr=sr,
        bpm=bpm,
        ghost_kick=ghost_kick(length, onsets, sr) if len(onsets) else None,
        rough_mix=mix,
        buses=buses,
        bus_comp=bus_comp,
        returns=returns,
        mix_comp=mix_comp,
        correlation=stereo_correlation(mix),
    )


def _returns(results: dict[str, StemResult], ctx: _Context) -> dict[str, Audio]:
    """Render reverb/delay returns from the processed stems, ducked like any return."""
    sources: dict[Role, list[Audio]] = {}
    for stem in results.values():
        sources.setdefault(stem.role, []).append(stem.audio)
    returns = make_returns(sources, ctx.sr, ctx.bpm)
    duck = (
        duck_settings(ctx.genre, Role.RETURN, ctx.bpm)
        if ctx.bpm is not None and len(ctx.onsets)
        else None
    )
    if duck is None:
        return returns
    return {
        name: apply_ducking(audio, duck_curve(audio.shape[1], ctx.onsets, ctx.sr, duck))
        for name, audio in returns.items()
    }


def _buses(
    results: dict[str, StemResult], returns: dict[str, Audio], ctx: _Context
) -> tuple[dict[Bus, Audio], dict[Bus, CompResult]]:
    """Sum stems into buses; glue and parallel-compress the drum bus."""
    groups: dict[Bus, list[Audio]] = {}
    for stem in results.values():
        groups.setdefault(BUS_BY_ROLE[stem.role], []).append(stem.audio)
    groups.setdefault(Bus.FX, []).extend(returns.values())
    buses = {
        bus: np.sum(parts, axis=0).astype(np.float32) for bus, parts in groups.items() if parts
    }

    comps: dict[Bus, CompResult] = {}
    if Bus.DRUMS in buses:
        settings = PRESETS[ctx.genre].buses
        glued = compress(buses[Bus.DRUMS], ctx.sr, settings.drum_glue)
        comps[Bus.DRUMS] = glued
        buses[Bus.DRUMS] = parallel_compress(glued.audio, ctx.sr, settings.parallel_blend)
    return buses, comps


def _process(audio: Audio, role: Role, ctx: _Context) -> StemResult:
    sr, onsets, bpm = ctx.sr, ctx.onsets, ctx.bpm
    staged = gain_stage(audio)
    cutoff = HIGHPASS_HZ[role]
    out = highpass(staged.audio, sr, cutoff)

    cuts: tuple[Resonance, ...] = (
        () if role in _NO_RESONANCE_CUTS else tuple(find_resonances(out, sr))
    )
    out = apply_cuts(out, sr, list(cuts), max_cut_db=_MAX_CUT_DB)

    settings = comp_settings(ctx.genre, role)
    comp = compress(out, sr, settings)
    out = comp.audio

    duck = duck_settings(ctx.genre, role, bpm) if bpm is not None and len(onsets) else None
    if duck is not None:
        out = apply_ducking(out, duck_curve(out.shape[1], onsets, sr, duck))

    level = rms_dbfs(out)
    fader_db = FADER_RMS_DBFS[role] - level if np.isfinite(level) else 0.0
    out = (out * np.float32(10 ** (fader_db / 20))).astype(np.float32)

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
        fader_db=fader_db,
    )
