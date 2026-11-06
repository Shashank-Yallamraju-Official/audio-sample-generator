import json
from dataclasses import replace

import numpy as np

from automix.master import master
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


def test_markdown_report_describes_the_master() -> None:
    result = mix_stems(stems(), SR, Genre.POP)
    mastered = master(result.rough_mix, SR, target_lufs=-14.0)
    text = to_markdown(result, Genre.POP, mastered)
    section = text.split("## Master")[1]
    assert "Adaptive Limiter" in section
    assert "Out Ceiling -1.0 dB" in section
    assert "LUFS" in section
    assert "dBTP" in section


def test_json_report_includes_master_meter() -> None:
    result = mix_stems(stems(), SR, Genre.POP)
    mastered = master(result.rough_mix, SR, target_lufs=-14.0)
    data = json.loads(to_json(result, Genre.POP, mastered))
    assert data["master"]["target_lufs"] == -14.0
    assert data["master"]["after"]["true_peak_dbtp"] <= -0.95


def test_report_without_master_has_no_master_section() -> None:
    result = mix_stems(stems(), SR, Genre.POP)
    assert "## Master" not in to_markdown(result, Genre.POP)
    assert json.loads(to_json(result, Genre.POP))["master"] is None


def test_markdown_report_flags_capped_limiting_and_reference_eq() -> None:
    result = mix_stems(stems(), SR, Genre.POP)
    mastered = replace(
        master(result.rough_mix, SR, target_lufs=-14.0),
        capped=True,
        match_eq_db={1000.0: 1.5},
    )
    section = to_markdown(result, Genre.POP, mastered, reference_name="ref.wav").split("## Master")[
        1
    ]
    assert "capped at 6 dB" in section
    assert "1000 Hz +1.5 dB" in section
    assert "ref.wav" in section
