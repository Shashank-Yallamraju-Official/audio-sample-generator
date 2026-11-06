"""Settings reports: Markdown in Logic Pro's stock-plugin terms, and JSON for tooling."""

import json

from automix.eq import cut_db
from automix.pipeline import MixResult, StemResult
from automix.presets import PRESETS, Genre

_GHOST = "ghost_kick.wav"


def _eq_lines(stem: StemResult) -> list[str]:
    lines = []
    if stem.highpass_hz is not None:
        lines.append(f"- Channel EQ: High Pass {stem.highpass_hz:.0f} Hz, 12 dB/Oct")
    lines.extend(
        f"- Channel EQ: bell {c.freq_hz:.0f} Hz, -{cut_db(c):.1f} dB, Q 6.0" for c in stem.cuts
    )
    return lines or ["- Channel EQ: none"]


def _comp_lines(stem: StemResult) -> list[str]:
    s = stem.comp_settings
    if not stem.comp.stages:
        return ["- Compressor: none"]
    lines = [
        f"- Compressor {i}: Ratio {s.ratio:g}:1, Attack {s.attack_ms:g} ms, "
        f"Release {s.release_ms:g} ms, Threshold {st.threshold_db:.1f} dB "
        f"(aim for about {st.gr_db:.1f} dB on the gain reduction meter)"
        for i, st in enumerate(stem.comp.stages, start=1)
    ]
    lines.append(f"- Make Up: +{stem.comp.makeup_db:.1f} dB")
    return lines


def _duck_lines(stem: StemResult) -> list[str]:
    if stem.duck is None:
        return []
    d = stem.duck
    return [
        f"- Ducking: Compressor with Side Chain: {_GHOST} (muted track), "
        f"about {d.depth_db:g} dB of reduction, Attack {d.attack_s * 1000:g} ms, "
        f"Release {d.release_s * 1000:.0f} ms"
    ]


def to_markdown(result: MixResult, genre: Genre) -> str:
    """Render the settings sheet a person copies into Logic Pro."""
    tempo = f"{result.bpm:.1f} BPM" if result.bpm is not None else "not detected"
    out = [
        "# automix settings",
        "",
        f"Genre: {genre.value} · Tempo: {tempo} · "
        f"Master target: {PRESETS[genre].master_lufs:.1f} LUFS integrated, -1 dBTP "
        "(-14 LUFS for streaming-only releases)",
        "",
        "Thresholds are measured on automix's compressor; Logic's will differ slightly, so "
        "set them by the gain reduction figure.",
    ]
    for name, stem in result.stems.items():
        out += ["", f"## {name}", "", f"Role: {stem.role.value} · Trim: {stem.gain_db:+.1f} dB"]
        out += ["", *_eq_lines(stem), *_comp_lines(stem), *_duck_lines(stem)]
        out.append(f"- Fader: {stem.fader_db:+.1f} dB (already applied to the processed stem)")
    return "\n".join(out) + "\n"


def to_json(result: MixResult, genre: Genre) -> str:
    """Serialise every chosen setting."""
    stems = {
        name: {
            "role": s.role.value,
            "gain_db": s.gain_db,
            "highpass_hz": s.highpass_hz,
            "cuts": [{"freq_hz": c.freq_hz, "cut_db": cut_db(c)} for c in s.cuts],
            "fader_db": s.fader_db,
            "compressor": {
                "ratio": s.comp_settings.ratio,
                "attack_ms": s.comp_settings.attack_ms,
                "release_ms": s.comp_settings.release_ms,
                "stages": [
                    {"threshold_db": t.threshold_db, "gr_db": t.gr_db} for t in s.comp.stages
                ],
                "makeup_db": s.comp.makeup_db,
            },
            "duck": None
            if s.duck is None
            else {
                "depth_db": s.duck.depth_db,
                "attack_s": s.duck.attack_s,
                "hold_s": s.duck.hold_s,
                "release_s": s.duck.release_s,
            },
        }
        for name, s in result.stems.items()
    }
    data = {
        "genre": genre.value,
        "bpm": result.bpm,
        "sample_rate": result.sr,
        "master_lufs": PRESETS[genre].master_lufs,
        "stems": stems,
    }
    return json.dumps(data, indent=2) + "\n"
