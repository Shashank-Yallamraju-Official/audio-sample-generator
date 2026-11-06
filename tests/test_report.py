import json

import numpy as np

from automix.pipeline import mix_stems
from automix.presets import Genre
from automix.report import to_json, to_markdown

from .conftest import SR
from .test_pipeline import stems


def test_markdown_report_uses_logic_plugin_terms() -> None:
    text = to_markdown(mix_stems(stems(), SR, Genre.DANCE), Genre.DANCE)
    assert "Channel EQ" in text
    assert "Compressor" in text
    assert "Side Chain: ghost_kick.wav" in text
    assert "## Bass.wav" in text
    assert "-7.0 LUFS" in text


def test_json_report_round_trips_every_stem() -> None:
    result = mix_stems(stems(), SR, Genre.POP)
    data = json.loads(to_json(result, Genre.POP))
    assert set(data["stems"]) == set(result.stems)
    assert data["genre"] == "pop"
    bass = data["stems"]["Bass.wav"]
    assert bass["role"] == "bass"
    assert bass["highpass_hz"] == 30.0
    assert isinstance(bass["compressor"]["stages"], list)
    assert np.isfinite(bass["gain_db"])


def test_markdown_report_lists_unprocessed_stages_as_none() -> None:
    audio = stems()
    extra = {"mystery.wav": audio["Pad.wav"], "Reverb Return.wav": audio["Pad.wav"]}
    text = to_markdown(mix_stems(extra, SR, Genre.POP), Genre.POP)
    assert "- Channel EQ: none" not in text.split("## Reverb Return.wav")[1]
    assert "- Compressor: none" in text
    assert "Tempo: not detected" in text


def test_markdown_report_describes_buses_in_logic_terms() -> None:
    text = to_markdown(mix_stems(stems(), SR, Genre.HIPHOP), Genre.HIPHOP)
    buses = text.split("## Buses")[1]
    assert "Track Stack" in buses
    assert "Drums" in buses
    assert "Parallel" in buses
    assert "Channel EQ in Side mode: High Pass 100 Hz" in buses
    assert "Reverb Return.wav" in buses
    assert "Stereo correlation" in buses


def test_json_report_lists_buses_and_returns() -> None:
    data = json.loads(to_json(mix_stems(stems(), SR, Genre.DANCE), Genre.DANCE))
    assert "Drums" in data["buses"]
    assert data["buses"]["Drums"]["parallel_blend"] > 0
    assert "Reverb Return.wav" in data["returns"]
    assert -1.0 <= data["mix"]["correlation"] <= 1.0


def test_markdown_report_notes_a_silent_drum_bus() -> None:
    silent = {"Kick.wav": np.zeros((2, SR), dtype=np.float32), "Pad.wav": stems()["Pad.wav"]}
    text = to_markdown(mix_stems(silent, SR, Genre.POP), Genre.POP)
    assert "Drums bus: Compressor bypassed" in text
