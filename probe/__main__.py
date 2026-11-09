"""ap-audio-probe: measure whether a generated song plays as its recipe instructs.

It gives a verdict but never blocks. It lays out the numbers and the reasons; a person decides whether to fix anything.

    python -m probe audio/foo.mp3 recipes/foo.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import recipe as recipe_mod
from .separate import resolve_device, separate_vocals
from .vocals import ACTIVE_RATIO_THRESHOLD, VOCAL_MARGIN_DB, Report, analyse


def _render(report: Report) -> str:
    lines = [
        f"{report.recipe.title}  /  {report.recipe.tempo} BPM",
        f"length {report.audio_seconds:.1f}s  (recipe {report.recipe.declared_duration:.0f}s)",
        "",
        f"{'section':<10} {'type':<6} {'span':<16} {'vocal RMS':>10} {'peak':>8} {'voiced':>7}  verdict",
        "-" * 76,
    ]
    for s in report.stats:
        kind = "instr" if s.section.instrumental else "sung"
        span = f"{s.section.start:.0f}-{s.clamped_end:.1f}s"
        if s.section.instrumental:
            verdict = "★vocals" if report.has_vocals(s) else "clean"
        else:
            verdict = "-"
        notes = []
        if s.bleed:
            notes.append(f"edge bleed {s.bleed:.1f}s excluded")
        if s.truncated:
            notes.append(f"short by {s.section.end - s.clamped_end:.1f}s")
        if notes:
            verdict += f" ({', '.join(notes)})"
        lines.append(
            f"{s.section.name:<10} {kind:<6} {span:<16} "
            f"{s.rms_dbfs:>9.1f}dB {s.peak_dbfs:>7.1f}dB {s.active_ratio:>6.0%}  {verdict}"
        )

    lines += [
        "-" * 76,
        f"reference level of the sung sections: {report.sung_reference_dbfs:.1f} dBFS "
        f"(\"vocals\" when within {VOCAL_MARGIN_DB:.0f}dB of it, "
        f"or voiced for more than {ACTIVE_RATIO_THRESHOLD:.0%})",
    ]
    if any(s.bleed for s in report.stats):
        lines.append("* \"edge bleed\" is voice reaching the edge from a neighbouring sung section (pickups and held endings).")
    lines.append("")
    if report.violations:
        names = ", ".join(s.section.name for s in report.violations)
        lines.append(f"⚠ vocals in sections marked instrumental: {names}")
    else:
        lines.append("✓ every section marked instrumental is clean")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe", description=__doc__)
    parser.add_argument("audio", type=Path, help="audio (mp3/wav)")
    parser.add_argument("recipe", type=Path, help="recipe.json")
    parser.add_argument("--out", type=Path, default=Path("out"), help="where to write the stems")
    parser.add_argument("--device", help="device for demucs (default: picks mps / cuda / cpu)")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[1/2] separating the vocal stem ({device}) ...")
    vocals = separate_vocals(args.audio, args.out, device=device)

    print("[2/2] measuring each section ...\n")
    print(_render(analyse(vocals, recipe_mod.load(args.recipe))))


if __name__ == "__main__":
    main()
