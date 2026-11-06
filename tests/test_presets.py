import pytest

from automix.compression import COMP_BY_ROLE, MAX_GR_PER_STAGE_DB
from automix.eq import Role
from automix.presets import PRESETS, Genre, comp_settings, duck_settings, parse_genre


@pytest.mark.parametrize(
    ("text", "genre"),
    [
        ("dance", Genre.DANCE),
        ("EDM", Genre.DANCE),
        ("Hip-Hop", Genre.HIPHOP),
        ("hiphop", Genre.HIPHOP),
        ("rap", Genre.HIPHOP),
        ("pop", Genre.POP),
    ],
)
def test_parse_genre_accepts_common_spellings(text: str, genre: Genre) -> None:
    assert parse_genre(text) == genre


def test_parse_genre_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="polka"):
        parse_genre("polka")


def test_every_genre_has_a_preset() -> None:
    assert set(PRESETS) == set(Genre)


def test_dance_bass_release_is_a_sixteenth_note() -> None:
    settings = duck_settings(Genre.DANCE, Role.BASS, bpm=128)
    assert settings is not None
    assert settings.release_s == pytest.approx(60 / 128 / 4)


def test_hiphop_bass_release_is_short_and_tempo_independent() -> None:
    slow = duck_settings(Genre.HIPHOP, Role.BASS, bpm=80)
    fast = duck_settings(Genre.HIPHOP, Role.BASS, bpm=160)
    assert slow is not None
    assert slow == fast
    assert 0.05 <= slow.release_s <= 0.08


def test_dance_pumps_pads_harder_than_pop() -> None:
    dance = duck_settings(Genre.DANCE, Role.SYNTH, bpm=124)
    pop = duck_settings(Genre.POP, Role.SYNTH, bpm=124)
    assert dance is not None
    assert pop is not None
    assert dance.depth_db > pop.depth_db


@pytest.mark.parametrize("genre", list(Genre))
@pytest.mark.parametrize("role", [Role.KICK, Role.VOCAL, Role.SNARE])
def test_trigger_and_lead_parts_are_never_ducked(genre: Genre, role: Role) -> None:
    assert duck_settings(genre, role, bpm=120) is None


def test_duck_settings_rejects_non_positive_tempo() -> None:
    with pytest.raises(ValueError, match="bpm"):
        duck_settings(Genre.DANCE, Role.BASS, bpm=0)


def test_hiphop_vocals_compress_in_two_stages() -> None:
    assert comp_settings(Genre.HIPHOP, Role.VOCAL).target_gr_db > MAX_GR_PER_STAGE_DB


def test_comp_settings_fall_back_to_role_defaults() -> None:
    assert comp_settings(Genre.POP, Role.KICK) == COMP_BY_ROLE[Role.KICK]


def test_dance_masters_louder_than_pop() -> None:
    assert PRESETS[Genre.DANCE].master_lufs > PRESETS[Genre.POP].master_lufs
