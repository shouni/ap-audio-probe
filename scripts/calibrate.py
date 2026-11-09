"""Build positive controls to check detection power and find where the verdict flips.

    .venv/bin/python scripts/calibrate.py audio/foo.wav recipes/foo.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe import recipe as recipe_mod
from probe.inject import inject_vocal
from probe.separate import separate_vocals
from probe.vocals import analyse

GAINS_DB = [0.0, -6.0, -12.0, -18.0, -24.0, -30.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--section", default="Interlude", help="instrumental section to inject into")
    parser.add_argument("--source-start", type=float, default=40.0, help="second at which to take the voice")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--device", help="device for demucs (default: picks mps / cuda / cpu)")
    args = parser.parse_args()

    rec = recipe_mod.load(args.recipe)
    target = next(s for s in rec.sections if s.name == args.section)
    if not target.instrumental:
        raise SystemExit(f"{target.name} is not marked instrumental")

    print("separating the original audio ...")
    clean_stem = separate_vocals(args.audio, args.out, device=args.device)

    baseline = analyse(clean_stem, rec)
    base = next(s for s in baseline.stats if s.section.name == target.name)
    print(
        f"\nbaseline: {target.name} in the original is {base.rms_dbfs:.1f}dB / voiced {base.active_ratio:.0%} "
        f"(sung reference {baseline.sung_reference_dbfs:.1f}dB)\n"
    )

    rows = []
    for gain in GAINS_DB:
        tag = f"inject{int(abs(gain)):02d}"
        contaminated = args.audio.parent / f"{args.audio.stem}-{tag}.wav"
        print(f"[{gain:+.0f}dB] mixing and separating ...")
        inject_vocal(args.audio, clean_stem, target, args.source_start, gain, contaminated)
        stem = separate_vocals(contaminated, args.out, device=args.device)

        report = analyse(stem, rec)
        stat = next(s for s in report.stats if s.section.name == target.name)
        rows.append((gain, stat.rms_dbfs, stat.active_ratio, report.has_vocals(stat)))

    print(f"\ninjected  {target.name} vocal RMS   voiced    verdict")
    print("-" * 52)
    for gain, rms, active, flagged in rows:
        print(f"{gain:+6.0f}dB {rms:>14.1f}dB {active:>8.0%}    {'★detected' if flagged else 'clean (missed)'}")
    print("-" * 52)

    detected = [g for g, _, _, f in rows if f]
    missed = [g for g, _, _, f in rows if not f]
    if not detected:
        print("⚠ nothing detected at any level. The verdict logic is not working.")
    elif missed:
        print(f"boundary: detected down to {min(detected):+.0f}dB, missed at {max(missed):+.0f}dB and below")
    else:
        print("detected at every level; smaller levels are worth trying.")


if __name__ == "__main__":
    main()
