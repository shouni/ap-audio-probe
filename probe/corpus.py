"""Line up several songs by composing mode, to tell whether one song is off or the whole mode leans that way.

    python -m probe.corpus audio/*.wav
    python -m probe.corpus audio/*.wav --balance    # also voice − backing (runs demucs on unseparated songs)

Recipes come from recipes/<audio name>.json; songs without one are skipped.
Gain reduction is worked back from the threshold recorded in assessments/<audio name>.json, and left empty
without a record. The generator's treble compressor is designed to "engage only in the chorus and pass the
verse", so these two columns per mode are the material for tuning the threshold offset (masterer.go's
trebleThresholdOffsetDB). Amounts are listed instead of time over the threshold (spectrum's "over") because
time alone cannot tell brief peaks from sustained compression (see the comment before spectrum.BandLevels).

A single-song measurement (loudness / spectrum / balance) cannot say whether a value is normal. The low
punch of sparkle_rock on 2026-10-28 only showed up after lining it up by hand against 20 edm_pop songs; this
fixes that lineup.

Chorus values average the sections whose name contains Chorus (the same rule as spectrum.render's summary).
There is no pass line; use it to pick songs far from their mode's median as candidates to listen to again.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import recipe as recipe_mod
from .balance import analyse_balance
from .loudness import measure as measure_loudness
from .separate import resolve_device, separate_vocals
from .spectrum import analyse as analyse_spectrum
from .spectrum import recorded_threshold
from .table import cell


@dataclass(frozen=True)
class TrackSummary:
    name: str
    mode: str
    integrated_lufs: float
    true_peak_dbtp: float
    # the chorus 40-120Hz punch (spectrum.LOW_PUNCH_PERCENTILES) and 6-10kHz average level
    chorus_low_punch: float | None
    chorus_treble: float | None
    # average 6-10kHz reduction in dB (spectrum.BandLevels.gain_reduction)
    chorus_gain_reduction: float | None
    verse_gain_reduction: float | None
    # the chorus "voice − backing"; only with --balance
    chorus_vocal_over_backing: float | None


def _section_mean(values: list[tuple[str, float | None]], kind: str = "Chorus") -> float | None:
    picked = [v for name, v in values if kind in name and v is not None]
    return float(np.mean(picked)) if picked else None


def summarise(
    audio: Path, recipe_path: Path, *, balance: bool, out_dir: Path, device: str | None,
    assessments: Path = Path("assessments"),
) -> TrackSummary:
    rec = recipe_mod.load(recipe_path)
    loudness = measure_loudness(audio)
    levels = analyse_spectrum(audio, rec, recorded_threshold(audio, assessments))

    vocal_over_backing = None
    if balance:
        vocals = separate_vocals(audio, out_dir, device=device)
        balances = analyse_balance(vocals, vocals.with_name("no_vocals.wav"), rec)
        vocal_over_backing = _section_mean([(b.section.name, b.vocal_over_backing_db) for b in balances])

    return TrackSummary(
        name=audio.stem,
        mode=rec.compose_mode or "(unknown)",
        integrated_lufs=loudness.integrated_lufs,
        true_peak_dbtp=loudness.true_peak_dbtp,
        chorus_low_punch=_section_mean([(lv.section.name, lv.low_punch) for lv in levels]),
        chorus_treble=_section_mean([(lv.section.name, lv.treble) for lv in levels]),
        chorus_gain_reduction=_section_mean([(lv.section.name, lv.gain_reduction) for lv in levels]),
        verse_gain_reduction=_section_mean([(lv.section.name, lv.gain_reduction) for lv in levels], "Verse"),
        chorus_vocal_over_backing=vocal_over_backing,
    )


# Column widths; table.cell aligns by display width in case headings contain wide characters.
NAME_WIDTH = 38
VALUE_WIDTH = 9


def _fmt(value: float | None, signed: bool = False, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _median(values: list[float | None]) -> float | None:
    picked = [v for v in values if v is not None]
    return float(np.median(picked)) if picked else None


def _row(label: str, cells: list[str]) -> str:
    return cell(label, NAME_WIDTH) + "".join(cell(c, VALUE_WIDTH, right=True) for c in cells)


def _values(lufs, tp, punch, treble, chorus_gr, verse_gr, vob) -> list[str]:
    return [
        _fmt(lufs), _fmt(tp), _fmt(punch), _fmt(treble),
        _fmt(chorus_gr, digits=2), _fmt(verse_gr, digits=2), _fmt(vob, signed=True),
    ]


def _cells(t: TrackSummary) -> tuple:
    return (
        t.integrated_lufs, t.true_peak_dbtp, t.chorus_low_punch, t.chorus_treble,
        t.chorus_gain_reduction, t.verse_gain_reduction, t.chorus_vocal_over_backing,
    )


def render(tracks: list[TrackSummary]) -> str:
    rule = "-" * (NAME_WIDTH + VALUE_WIDTH * 7)
    lines = [
        _row("track", ["LUFS", "TP", "punch", "6-10k", "GR", "GR", "voice-bk"]),
        _row("", ["", "dBTP", "(chorus)", "(chorus)", "(chorus)", "(verse)", "(chorus)"]),
        rule,
    ]

    for mode in sorted({t.mode for t in tracks}):
        group = [t for t in tracks if t.mode == mode]
        for t in group:
            lines.append(_row(t.name, _values(*_cells(t))))
        columns = zip(*(_cells(t) for t in group))
        lines.append(_row(f"  ↳ {mode} median ({len(group)} song{'' if len(group) == 1 else 's'})", _values(*(_median(list(c)) for c in columns))))
        lines.append(rule)

    lines += [
        "punch: high percentile − median of the 40-120Hz short-term level. Smaller means a flatter kick.",
        "GR: average 6-10kHz gain reduction in dB (worked back from the recorded threshold). Empty without a record.",
        "voice-bk: median over voiced frames. Negative means the backing is louder.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe.corpus", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+", help="audio (master.wav recommended)")
    parser.add_argument("--recipes", type=Path, default=Path("recipes"), help="where the recipe.json files are")
    parser.add_argument("--assessments", type=Path, default=Path("assessments"),
                        help="where the assessment.json files are (the compression threshold is read from them)")
    parser.add_argument("--balance", action="store_true", help="also show the chorus voice − backing (separates with demucs)")
    parser.add_argument("--out", type=Path, default=Path("out"), help="where to write the stems")
    parser.add_argument("--device", help="device for demucs (default: picks mps / cuda / cpu)")
    args = parser.parse_args()

    device = resolve_device(args.device) if args.balance else None
    tracks = []
    for audio in args.audio:
        recipe_path = args.recipes / f"{audio.stem}.json"
        if not recipe_path.exists():
            print(f"no recipe, skipping: {audio.name}", file=sys.stderr)
            continue
        print(f"measuring: {audio.name} ...", file=sys.stderr)
        tracks.append(summarise(audio, recipe_path, balance=args.balance, out_dir=args.out,
                                device=device, assessments=args.assessments))

    if not tracks:
        raise SystemExit("nothing to measure (no recipes found)")
    print()
    print(render(tracks))


if __name__ == "__main__":
    main()
