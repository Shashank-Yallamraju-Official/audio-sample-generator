# automix

Automated first-pass mixing for stems: gain staging, cleanup EQ, kick-triggered sidechain
ducking and starting-point compression, tuned for dance, hip-hop and pop. It does the
measurable, tedious work; you finish the mix by ear in Logic Pro.

Design notes and the rules behind every default:
[mix best-practices doc](https://claude.ai/artifact/MZ3SsDsHNx3F6vU1NgwVbH).

## Status

| Stage | State |
| --- | --- |
| Gain staging (`automix.gain`) | done |
| Kick onset detection and sidechain ducking (`automix.sidechain`) | done |
| Role detection, high-pass and resonance cuts (`automix.eq`) | done |
| Compression fitted to target gain reduction (`automix.compression`) | done |
| Genre presets for dance, hip-hop and pop (`automix.presets`) | done |
| Pipeline, Logic handoff and CLI (`automix.pipeline`, `automix.report`, `automix.cli`) | done |
| Bus processing (drum glue, low-end mono, send returns), loudness metering | planned |

## Usage

Export every track from Logic as a stem starting at bar 1 (File > Export > All Tracks as Audio
Files), with names that say what each stem is, such as `Kick.wav`, `808.wav`, `Lead Vox.wav`
or `Pad.wav`. Then run:

```sh
uv run automix path/to/stems -o path/to/out --genre hip-hop   # --bpm 92 to skip detection
```

The output folder holds:

| Path | What it is |
| --- | --- |
| `processed/` | Stems with gain, EQ, compression and ducking applied. Drag into Logic at bar 1. |
| `dry/` | Stems with gain staging only, to redo any move with your own plugins. |
| `ghost_kick.wav` | A click on every kick. Mute it and use it as the Compressor's Side Chain. |
| `rough_mix.wav` | All processed stems summed, peaking at -6 dBFS. |
| `settings.md` / `settings.json` | Every chosen setting, written in Logic Channel EQ and Compressor terms. |

Processed stems include a starting fader balance (kick and bass loudest, hats lowest), so
the rough mix is already roughly balanced; the settings sheet lists each fader move.

Stem roles come from filenames; anything unrecognised is treated as `other` and gets
gentle defaults.

## Development

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                      # install deps + dev tools from the lockfile
uv run pre-commit install    # format, lint and typecheck on every commit
uv run pytest                # tests + coverage (fails under 90%)
uv run ruff format           # format
uv run ruff check            # lint
uv run mypy                  # strict type check
```

CI runs the same four checks on Python 3.12 and 3.13 for every push and PR.

### Conventions

- **Test-driven:** write a failing test, make it pass, then refactor. Every PR adds tests first.
- **Audio arrays** are `float32`, shaped `(channels, samples)`, matching pedalboard.
- **Pure functions:** processing functions never mutate their input.
- **No audio in git:** `.wav`, `.aif`, `.flac` and `.mp3` are ignored; tests synthesise signals.
