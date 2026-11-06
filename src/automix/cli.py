"""Command line: ``automix STEMS_DIR -o OUT --genre dance``."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from automix._types import Audio
from automix.pipeline import mix_stems
from automix.presets import parse_genre
from automix.report import to_json, to_markdown

AUDIO_SUFFIXES = frozenset({".wav", ".aif", ".aiff", ".flac"})


class InputError(Exception):
    """A problem with the stems folder that the user needs to fix."""


def _load(folder: Path) -> tuple[dict[str, Audio], int]:
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
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
    args = parser.parse_args(argv)

    try:
        genre = parse_genre(args.genre)
        stems, sr = _load(args.stems)
    except (InputError, ValueError) as err:
        print(f"automix: {err}", file=sys.stderr)
        return 2

    result = mix_stems(stems, sr, genre, bpm=args.bpm)
    out: Path = args.out
    for name, stem in result.stems.items():
        wav = Path(name).with_suffix(".wav").name
        _write(out / "processed" / wav, stem.audio, sr)
        _write(out / "dry" / wav, stem.dry, sr)
    if result.ghost_kick is not None:
        _write(out / "ghost_kick.wav", result.ghost_kick, sr)
    _write(out / "rough_mix.wav", result.rough_mix, sr)
    (out / "settings.md").write_text(to_markdown(result, genre))
    (out / "settings.json").write_text(to_json(result, genre))
    print(f"automix: wrote {len(result.stems)} stems to {out}")
    return 0
