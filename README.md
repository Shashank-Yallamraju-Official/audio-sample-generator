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
| Buses, genre presets, Logic handoff, CLI | planned |

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
