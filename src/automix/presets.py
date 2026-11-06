"""Genre presets: the settings that differ between dance, hip-hop and pop.

Everything not listed here uses the role defaults in :mod:`automix.compression` and
:mod:`automix.eq`. Values follow the genre table in the best-practices doc.
"""

from dataclasses import dataclass, field
from enum import StrEnum

from automix.compression import COMP_BY_ROLE, CompSettings
from automix.eq import Role
from automix.sidechain import DuckSettings


class Genre(StrEnum):
    """Supported genres."""

    DANCE = "dance"
    HIPHOP = "hiphop"
    POP = "pop"


_ALIASES: dict[str, Genre] = {
    "dance": Genre.DANCE,
    "edm": Genre.DANCE,
    "house": Genre.DANCE,
    "techno": Genre.DANCE,
    "hiphop": Genre.HIPHOP,
    "rap": Genre.HIPHOP,
    "trap": Genre.HIPHOP,
    "pop": Genre.POP,
}


def parse_genre(text: str) -> Genre:
    """Map a user-typed genre name ("Hip-Hop", "EDM", ...) to a :class:`Genre`."""
    key = "".join(ch for ch in text.lower() if ch.isalnum())
    try:
        return _ALIASES[key]
    except KeyError:
        msg = f"unknown genre {text!r}; expected one of {sorted(_ALIASES)}"
        raise ValueError(msg) from None


@dataclass(frozen=True)
class DuckRule:
    """How one role ducks under the kick. Release is in beats when tempo-synced."""

    depth_db: float
    release_beats: float | None = None
    release_s: float = 0.1


@dataclass(frozen=True)
class BusSettings:
    """Drum-bus and mix-bus processing for a genre."""

    drum_glue: CompSettings
    parallel_blend: float  # 0 turns parallel drum compression off
    mix_glue: CompSettings
    low_mono_hz: float


@dataclass(frozen=True)
class Preset:
    """Genre-specific overrides."""

    ducking: dict[Role, DuckRule]
    master_lufs: float
    buses: BusSettings
    comp_overrides: dict[Role, CompSettings] = field(default_factory=dict)


SIXTEENTH, EIGHTH = 0.25, 0.5

PRESETS: dict[Genre, Preset] = {
    Genre.DANCE: Preset(
        ducking={
            Role.BASS: DuckRule(8.0, release_beats=SIXTEENTH),
            Role.SYNTH: DuckRule(8.0, release_beats=EIGHTH),
            Role.RETURN: DuckRule(4.5, release_beats=EIGHTH),
        },
        master_lufs=-7.0,
        buses=BusSettings(
            drum_glue=CompSettings(ratio=4.0, attack_ms=10.0, release_ms=100.0, target_gr_db=3.0),
            parallel_blend=0.3,
            mix_glue=CompSettings(ratio=2.0, attack_ms=30.0, release_ms=200.0, target_gr_db=2.0),
            low_mono_hz=120.0,
        ),
    ),
    Genre.HIPHOP: Preset(
        ducking={
            # 808 tucks under the kick for one short moment, whatever the tempo.
            Role.BASS: DuckRule(4.5, release_s=0.065),
            Role.SYNTH: DuckRule(2.5, release_beats=EIGHTH),
            Role.RETURN: DuckRule(3.0, release_beats=EIGHTH),
        },
        master_lufs=-8.0,
        buses=BusSettings(
            drum_glue=CompSettings(ratio=4.0, attack_ms=30.0, release_ms=100.0, target_gr_db=3.0),
            parallel_blend=0.3,
            mix_glue=CompSettings(ratio=2.0, attack_ms=30.0, release_ms=200.0, target_gr_db=1.5),
            # The 808 owns the sub; keep only that mono so wide synths keep their low mids.
            low_mono_hz=100.0,
        ),
        comp_overrides={
            Role.VOCAL: CompSettings(ratio=4.0, attack_ms=5.0, release_ms=75.0, target_gr_db=8.0),
        },
    ),
    Genre.POP: Preset(
        ducking={
            Role.BASS: DuckRule(3.0, release_beats=SIXTEENTH),
            Role.SYNTH: DuckRule(3.0, release_beats=EIGHTH),
            Role.RETURN: DuckRule(3.0, release_beats=EIGHTH),
        },
        master_lufs=-9.0,
        buses=BusSettings(
            drum_glue=CompSettings(ratio=2.0, attack_ms=30.0, release_ms=100.0, target_gr_db=2.0),
            parallel_blend=0.0,
            mix_glue=CompSettings(ratio=1.5, attack_ms=30.0, release_ms=200.0, target_gr_db=1.5),
            low_mono_hz=120.0,
        ),
    ),
}


def duck_settings(genre: Genre, role: Role, bpm: float) -> DuckSettings | None:
    """Return the ducking for ``role`` in ``genre`` at ``bpm``, or ``None`` if it isn't ducked."""
    if bpm <= 0:
        msg = f"bpm must be positive, got {bpm}"
        raise ValueError(msg)
    rule = PRESETS[genre].ducking.get(role)
    if rule is None:
        return None
    release = rule.release_s if rule.release_beats is None else rule.release_beats * 60 / bpm
    return DuckSettings(depth_db=rule.depth_db, release_s=release)


def comp_settings(genre: Genre, role: Role) -> CompSettings:
    """Return compression for ``role`` in ``genre``, falling back to the role default."""
    return PRESETS[genre].comp_overrides.get(role, COMP_BY_ROLE[role])
