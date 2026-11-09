"""Measure the loudness of several songs and see how well they match.

The generator's checks score one song at a time, so level differences across a series are structurally
invisible to it. That is what matters on streaming services, so this compares across songs.

    python -m probe.loudness audio/*.wav
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Allowed level spread for songs released together; a common mastering rule of thumb.
SPREAD_TOLERANCE_LU = 1.0

# True-peak ceiling for distribution; the usual guideline to avoid distortion in lossy encoding.
TRUE_PEAK_CEILING_DBTP = -1.0


@dataclass(frozen=True)
class Loudness:
    path: Path
    integrated_lufs: float
    true_peak_dbtp: float
    lra: float

    @property
    def peak_is_hot(self) -> bool:
        return self.true_peak_dbtp > TRUE_PEAK_CEILING_DBTP


def measure(path: Path) -> Loudness:
    """Measure BS.1770 values with ffmpeg's loudnorm."""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    # loudnorm prints its measurements as JSON at the end of stderr.
    blocks = re.findall(r"\{[^{}]*\}", result.stderr, re.DOTALL)
    if not blocks:
        raise RuntimeError(f"cannot parse the loudnorm output: {path}")
    data = json.loads(blocks[-1])

    return Loudness(
        path=path,
        integrated_lufs=float(data["input_i"]),
        true_peak_dbtp=float(data["input_tp"]),
        lra=float(data["input_lra"]),
    )


def render(tracks: list[Loudness]) -> str:
    lines = [
        f"{'track':<34} {'LUFS':>9} {'True Peak':>11} {'LRA':>7}",
        "-" * 64,
    ]
    for t in tracks:
        flag = "  ⚠ peak too hot" if t.peak_is_hot else ""
        lines.append(
            f"{t.path.stem[:33]:<34} {t.integrated_lufs:>8.1f} "
            f"{t.true_peak_dbtp:>10.1f} {t.lra:>6.1f}{flag}"
        )
    lines.append("-" * 64)

    levels = [t.integrated_lufs for t in tracks]
    spread = max(levels) - min(levels)
    lines.append(f"level spread: {spread:.1f} LU (guideline {SPREAD_TOLERANCE_LU:.1f} LU)")

    if len(tracks) < 2:
        lines.append("Comparing needs two or more songs.")
    elif spread > SPREAD_TOLERANCE_LU:
        loudest = max(tracks, key=lambda t: t.integrated_lufs)
        quietest = min(tracks, key=lambda t: t.integrated_lufs)
        lines.append(
            f"⚠ not matched: the loudest {loudest.path.stem[:24]} and "
            f"the quietest {quietest.path.stem[:24]} are {spread:.1f} LU apart."
        )
    else:
        lines.append("✓ within range to sit together as a series.")

    hot = [t for t in tracks if t.peak_is_hot]
    if hot:
        lines.append(
            f"⚠ {len(hot)} song(s) exceed {TRUE_PEAK_CEILING_DBTP:.1f} dBTP true peak "
            f"(may distort in lossy encoding)."
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path, nargs="+", help="audio to measure (master.wav recommended)")
    args = parser.parse_args()

    tracks = []
    for path in args.tracks:
        print(f"measuring: {path.name} ...", file=sys.stderr)
        tracks.append(measure(path))

    print()
    print(render(sorted(tracks, key=lambda t: t.integrated_lufs)))


if __name__ == "__main__":
    main()
