"""Settings reports: Markdown in Logic Pro's stock-plugin terms, and JSON for tooling."""

import json

from automix.buses import DELAY_SENDS_DB, RETURN_BAND_HZ, REVERB_SENDS_DB, Bus
from automix.compression import CompResult, CompSettings
from automix.eq import cut_db
from automix.master import MasterResult, Meter
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


def _glue(label: str, settings: CompSettings, comp: CompResult) -> str:
    if not comp.stages:
        return f"- {label}: Compressor bypassed (bus is silent)"
    stage = comp.stages[0]
    return (
        f"- {label}: Compressor (Vintage VCA), Ratio {settings.ratio:g}:1, "
        f"Attack {settings.attack_ms:g} ms, Release {settings.release_ms:g} ms, "
        f"Threshold {stage.threshold_db:.1f} dB (about {comp.gr_db:.1f} dB of gain reduction)"
    )


def _send_lines(result: MixResult) -> list[str]:
    lines = []
    for name, stem in result.stems.items():
        for target, sends in (
            ("Reverb Return.wav", REVERB_SENDS_DB),
            ("Delay Return.wav", DELAY_SENDS_DB),
        ):
            if stem.role in sends and target in result.returns:
                lines.append(f"- Send: {name} to {target} at {sends[stem.role]:g} dB")
    return lines


def _bus_lines(result: MixResult, genre: Genre) -> list[str]:
    settings = PRESETS[genre].buses
    low, high = RETURN_BAND_HZ
    names = ", ".join(bus.value for bus in result.buses)
    lines = [
        "",
        "## Buses",
        "",
        f"Group stems into one Track Stack per bus: {names}. Bus prints are in `buses/`.",
        "",
    ]
    if Bus.DRUMS in result.bus_comp:
        lines.append(_glue("Drums bus", settings.drum_glue, result.bus_comp[Bus.DRUMS]))
        if settings.parallel_blend > 0:
            lines.append(
                "- Drums bus: Parallel compression, a send to an aux with Compressor at "
                "Ratio 8:1, Attack 1 ms, Release 80 ms, about 10 dB of gain reduction, "
                f"blended at {settings.parallel_blend:.0%} of the dry level"
            )
    for name in result.returns:
        effect = (
            "Space Designer or ChromaVerb"
            if name.startswith("Reverb")
            else "Tape Delay, dotted 1/8"
        )
        lines.append(
            f"- {name}: aux with {effect}, 100% wet, then Channel EQ High Pass {low:.0f} Hz "
            f"and Low Pass {high / 1000:g} kHz"
        )
    lines += _send_lines(result)
    lines += [
        _glue("Stereo Out", settings.mix_glue, result.mix_comp),
        f"- Stereo Out: Channel EQ in Side mode: High Pass {settings.low_mono_hz:.0f} Hz "
        "(keeps the low end mono)",
        f"- Stereo correlation of the rough mix: {result.correlation:+.2f} "
        "(above 0 is mono-safe; near +1 is narrow)",
    ]
    return lines


def _meter_text(meter: Meter) -> str:
    return (
        f"{meter.lufs:.1f} LUFS integrated, {meter.true_peak_dbtp:.1f} dBTP true peak, "
        f"{meter.crest_db:.1f} dB crest"
    )


def _master_lines(mastered: MasterResult, reference_name: str | None) -> list[str]:
    lines = ["", "## Master", ""]
    lines.append(f"Target: {mastered.target_lufs:.1f} LUFS, ceiling -1.0 dBTP")
    if reference_name is not None:
        lines.append(f"Reference: {reference_name} (loudness and tonal balance matched)")
    lines += [
        "",
        f"- Before: {_meter_text(mastered.before)}",
        f"- After: {_meter_text(mastered.after)}",
    ]
    if mastered.match_eq_db:
        bands = ", ".join(f"{f:g} Hz {g:+.1f} dB" for f, g in mastered.match_eq_db.items())
        lines.append(f"- Stereo Out: Linear Phase EQ toward the reference: {bands}")
    lines.append(
        f"- Stereo Out: Adaptive Limiter, Gain {mastered.gain_db:+.1f} dB, Out Ceiling -1.0 dB "
        f"(about {mastered.limiting_db:.1f} dB of limiting); or Mastering Assistant set to "
        f"{mastered.target_lufs:.0f} LUFS"
    )
    if mastered.capped:
        lines.append(
            "- Limiting was capped at 6 dB, so the master is quieter than the target. Get the "
            "rest from arrangement and saturation, or push the limiter only if it still sounds "
            "right."
        )
    return lines


def to_markdown(
    result: MixResult,
    genre: Genre,
    mastered: MasterResult | None = None,
    reference_name: str | None = None,
) -> str:
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
    out += _bus_lines(result, genre)
    if mastered is not None:
        out += _master_lines(mastered, reference_name)
    return "\n".join(out) + "\n"


def _meter_json(meter: Meter) -> dict[str, float]:
    return {"lufs": meter.lufs, "true_peak_dbtp": meter.true_peak_dbtp, "crest_db": meter.crest_db}


def to_json(result: MixResult, genre: Genre, mastered: MasterResult | None = None) -> str:
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
    bus_settings = PRESETS[genre].buses
    buses = {
        bus.value: {
            "glue_gr_db": result.bus_comp[bus].gr_db if bus in result.bus_comp else 0.0,
            "parallel_blend": bus_settings.parallel_blend if bus is Bus.DRUMS else 0.0,
        }
        for bus in result.buses
    }
    data = {
        "genre": genre.value,
        "buses": buses,
        "returns": sorted(result.returns),
        "mix": {
            "glue_gr_db": result.mix_comp.gr_db,
            "low_mono_hz": bus_settings.low_mono_hz,
            "correlation": result.correlation,
        },
        "bpm": result.bpm,
        "sample_rate": result.sr,
        "master_lufs": PRESETS[genre].master_lufs,
        "stems": stems,
        "master": None
        if mastered is None
        else {
            "target_lufs": mastered.target_lufs,
            "gain_db": mastered.gain_db,
            "limiting_db": mastered.limiting_db,
            "capped": mastered.capped,
            "match_eq_db": {str(f): g for f, g in mastered.match_eq_db.items()},
            "before": _meter_json(mastered.before),
            "after": _meter_json(mastered.after),
        },
    }
    return json.dumps(data, indent=2) + "\n"
