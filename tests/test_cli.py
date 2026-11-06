from pathlib import Path

import pytest
import soundfile as sf

from automix.cli import main

from .conftest import SR
from .test_pipeline import stems


def write_stems(folder: Path) -> None:
    folder.mkdir()
    for name, audio in stems().items():
        sf.write(folder / name, audio.T, SR, subtype="FLOAT")


def test_cli_writes_the_logic_handoff(tmp_path: Path) -> None:
    write_stems(tmp_path / "in")
    out = tmp_path / "out"
    assert main([str(tmp_path / "in"), "-o", str(out), "--genre", "dance"]) == 0
    for name in stems():
        assert (out / "processed" / name).exists()
        assert (out / "dry" / name).exists()
    for name in ["ghost_kick.wav", "rough_mix.wav", "settings.md", "settings.json"]:
        assert (out / name).exists()
    audio, sr = sf.read(out / "processed" / "Bass.wav", always_2d=True)
    assert sr == SR
    assert audio.shape == (4 * SR, 2)


def test_cli_rejects_mixed_sample_rates(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_stems(tmp_path / "in")
    sf.write(tmp_path / "in" / "Odd.wav", stems()["Pad.wav"].T, 44_100, subtype="FLOAT")
    assert main([str(tmp_path / "in"), "-o", str(tmp_path / "out")]) == 2
    assert "sample rate" in capsys.readouterr().err


def test_cli_rejects_a_folder_without_audio(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "empty").mkdir()
    assert main([str(tmp_path / "empty"), "-o", str(tmp_path / "out")]) == 2
    assert "no audio" in capsys.readouterr().err


def test_cli_upmixes_mono_stems(tmp_path: Path) -> None:
    write_stems(tmp_path / "in")
    sf.write(tmp_path / "in" / "Keys.wav", stems()["Pad.wav"][0], SR, subtype="FLOAT")
    assert main([str(tmp_path / "in"), "-o", str(tmp_path / "out"), "--bpm", "128"]) == 0
    audio, _ = sf.read(tmp_path / "out" / "processed" / "Keys.wav", always_2d=True)
    assert audio.shape[1] == 2


def test_cli_skips_the_ghost_kick_without_a_kick_stem(tmp_path: Path) -> None:
    (tmp_path / "in").mkdir()
    sf.write(tmp_path / "in" / "Pad.wav", stems()["Pad.wav"].T, SR, subtype="FLOAT")
    assert main([str(tmp_path / "in"), "-o", str(tmp_path / "out")]) == 0
    assert not (tmp_path / "out" / "ghost_kick.wav").exists()


def test_cli_writes_bus_prints_and_returns(tmp_path: Path) -> None:
    write_stems(tmp_path / "in")
    out = tmp_path / "out"
    assert main([str(tmp_path / "in"), "-o", str(out), "--genre", "pop"]) == 0
    assert (out / "buses" / "Drums.wav").exists()
    assert (out / "processed" / "Reverb Return.wav").exists()
