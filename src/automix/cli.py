"""Command line: ``automix STEMS_DIR -o OUT --genre dance``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from automix._types import Audio
from automix.master import master, measure
from automix.pipeline import mix_stems
from automix.presets import PRESETS, parse_genre
from automix.report import to_json, to_markdown

AUDIO_SUFFIXES = frozenset({".wav", ".aif", ".aiff", ".flac"})
# Logic leaves a downmix of the whole project beside the exported stems when Smart Tempo is
# on; mixing it in would double every part.
_SMART_TEMPO_PREFIX = "smart tempo multitrack set"


class InputError(Exception):
    """A problem with the stems folder that the user needs to fix."""


def _load(folder: Path) -> tuple[dict[str, Audio], int]:
    audio_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    files = [p for p in audio_files if not p.name.lower().startswith(_SMART_TEMPO_PREFIX)]
    for path in audio_files:
        if path not in files:
            print(f"automix: skipped {path.name} (Logic's Smart Tempo downmix)", file=sys.stderr)
    if not files:
        msg = f"no audio files in {folder}"
        raise InputError(msg)
    stems: dict[str, Audio] = {}
    rates: dict[str, int] = {}
    for path in files:
        data, sr = sf.read(path, always_2d=True, dtype="float32")
        audio = np.ascontiguousarray(data.T)
        stems[path.name] = np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio[:2]
        rates[path.name] = int(sr)
    if len(set(rates.values())) > 1:
        detail = ", ".join(f"{n}: {r} Hz" for n, r in rates.items())
        msg = f"stems must share one sample rate ({detail})"
        raise InputError(msg)
    return stems, next(iter(rates.values()))


STREAMING_LUFS = -14.0


def _load_reference(path: Path, sr: int) -> Audio:
    data, ref_sr = sf.read(path, always_2d=True, dtype="float32")
    if ref_sr != sr:
        msg = f"reference {path.name} is {ref_sr} Hz; stems are {sr} Hz"
        raise InputError(msg)
    audio = np.ascontiguousarray(data.T)
    reference = np.repeat(audio, 2, axis=0) if audio.shape[0] == 1 else audio[:2]
    if not np.isfinite(measure(reference, sr).lufs):
        msg = f"reference {path.name} is silent or shorter than 0.4 s"
        raise InputError(msg)
    return reference


def _write(path: Path, audio: Audio, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio.T, sr, subtype="FLOAT")


def main(argv: Sequence[str] | None = None) -> int:
    """Run automix; returns a process exit code."""
    parser = argparse.ArgumentParser(prog="automix", description=__doc__)
    parser.add_argument("stems", type=Path, help="folder of stems, all starting at bar 1")
    parser.add_argument("-o", "--out", type=Path, required=True, help="output folder")
    parser.add_argument("--genre", default="pop", help="dance, hip-hop or pop (default pop)")
    parser.add_argument("--bpm", type=float, help="tempo; detected from the kick if omitted")
    parser.add_argument("--reference", type=Path, help="track to match for tone and loudness")
    parser.add_argument(
        "--streaming", action="store_true", help="master to -14 LUFS instead of the genre target"
    )
    parser.add_argument("--no-master", action="store_true", help="skip the first-pass master")
    args = parser.parse_args(argv)

    try:
        genre = parse_genre(args.genre)
        stems, sr = _load(args.stems)
        reference = None if args.reference is None else _load_reference(args.reference, sr)
    except (InputError, ValueError) as err:
        print(f"automix: {err}", file=sys.stderr)
        return 2

    result = mix_stems(stems, sr, genre, bpm=args.bpm)
    out: Path = args.out
    for name, stem in result.stems.items():
        wav = Path(name).with_suffix(".wav").name
        _write(out / "processed" / wav, stem.audio, sr)
        _write(out / "dry" / wav, stem.dry, sr)
    for name, audio in result.returns.items():
        _write(out / "processed" / name, audio, sr)
    for bus, audio in result.buses.items():
        _write(out / "buses" / f"{bus.value}.wav", audio, sr)
    if result.ghost_kick is not None:
        _write(out / "ghost_kick.wav", result.ghost_kick, sr)
    _write(out / "rough_mix.wav", result.rough_mix, sr)
    mastered = None
    if not args.no_master:
        target = STREAMING_LUFS if args.streaming else PRESETS[genre].master_lufs
        mastered = master(result.rough_mix, sr, target, reference=reference)
        _write(out / "master.wav", mastered.audio, sr)
    ref_name = None if args.reference is None else args.reference.name
    (out / "settings.md").write_text(to_markdown(result, genre, mastered, ref_name))
    (out / "settings.json").write_text(to_json(result, genre, mastered))
    print(f"automix: wrote {len(result.stems)} stems to {out}")
    return 0
