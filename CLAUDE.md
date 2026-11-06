# automix: notes for Claude

- Work test-first: add or change a failing test before touching `src/`.
- Before every commit run: `uv run ruff format && uv run ruff check && uv run mypy && uv run pytest`.
- mypy is `--strict`; ruff rules are in `pyproject.toml`. Fix findings rather than adding ignores.
- Audio is `automix._types.Audio` (`float32`, `(channels, samples)`). Never commit audio files.
- Defaults must match the mix best-practices doc linked in README.md; change both together.
- Target DAW is Logic Pro; genres are dance, hip-hop and pop.
