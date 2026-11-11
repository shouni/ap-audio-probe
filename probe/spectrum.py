"""Measure the band balance per section.

The generator's mastering compresses 6-10kHz only. This measures, across songs, how those constants behave
in choruses and verses. The threshold is read from assessment.json (the "over" and "GR" columns are
explained in the comment before BandLevels).

Only the processed audio can be measured. How much the compressor did would need the unprocessed audio for
comparison. It can show that compression "is working", never that it "is needed".

    python -m probe.spectrum audio/foo.wav recipes/foo.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from . import recipe as recipe_mod
from .assessment import load as load_assessment
from .recipe import Recipe, Section
from .vocals import SILENCE_DBFS, dbfs

# Bands matching the mastering's acrossover=split=6000 10000. Only the 6-10kHz between them is compressed;
# the generator calls the three bands body / treble / air.
#
# The lower edge was 4000 until 2026-09-05. The compressor works on the full mix and cannot tell voice from
# instruments, so including 4-6kHz, where consonants and presence sit, made the voice itself cross the
# threshold the harder it sang in the chorus (the cause of the "muffled vocal"). That day the generator moved
# both the compressed band and the threshold measurement to 6000. This must match, or uncompressed bands
# would be read as gain reduction.
CROSSOVER_HZ = (6000.0, 10000.0)

# Boundary that shows 4-6kHz, no longer compressed, separately from what lies below it.
#
# The generator's chain treats everything below 6kHz as one band (body) and has no such line. It is shown
# separately because this exact band returned to unprocessed on 2026-09-05, and a column is needed to see
# whether the core of the voice came back. It is not used for the "over" judgement.
PRESENCE_HZ = 4000.0

# The band where kick and bass sit. Only used to measure low-end punch (LOW_PUNCH_PERCENTILES); the
# mastering has no matching split.
LOW_PUNCH_HZ = (40.0, 120.0)

# Reports the gap between the high percentile and the median of the low band's short-term level as "punch".
# A standing kick lifts only the hits and widens the gap; bass or pads filling the low end narrow it.
#
# Needed on 2026-10-28 to compare songs built from the same instructions with a Flow Music version. Over whole
# songs (-40dB floor), three sparkle_rock songs measured 3.8–5.3dB, the median of 20 edm_pop songs 8.1dB,
# and the Flow Music version 8.0dB. Running the generator's mastering on the Flow Music version moved it only
# 7.7 → 7.6dB, so the gap is decided by generation, not mastering. There is no pass line.
LOW_PUNCH_PERCENTILES = (95, 50)


def _band(x: np.ndarray, sr: int, low: float | None, high: float | None) -> np.ndarray:
    nyq = sr / 2.0
    if low and high:
        sos = butter(4, [low / nyq, min(high, nyq * 0.99) / nyq], btype="bandpass", output="sos")
    elif high:
        sos = butter(4, min(high, nyq * 0.99) / nyq, btype="lowpass", output="sos")
    else:
        sos = butter(4, low / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, x)


# Short-time window matching the compressor's detector, the time scale that the mastering's attack=5ms /
# release=80ms follows. A whole-section RMS averages 36 seconds and cannot tell whether compression engages.
DETECTOR_WINDOW_SECONDS = 0.05
DETECTOR_HOP_SECONDS = 0.01


# The "over" column counts against the threshold the generator recorded in assessment.json
# (mastering.treble_threshold_dbfs) plus the loudness-matching gain, i.e. the line as it falls on master.wav.
#
# It used to count against the mastering's fixed value (acompressor=threshold=0.025, -32.0dBFS). Across 11
# songs the chorus time over it varied fourfold (11–43 %), so on 2026-08-15 the generator made the threshold
# relative to each song (spread on the same 11 songs fell from 31 pt to 18 pt; both figures are for the
# 4-10kHz band of the time and do not compare with today's 6-10kHz). From then on, counting against the fixed
# value no longer showed what was actually applied. The per-song threshold was only in the logs, so on
# 2026-10-29 the generator started recording it, and this just reads it. Songs without the record get no
# column: counting against a guessed threshold would be mistaken for the real amount.
#
# A compressor's output only exceeds the threshold when its input did (the static curve is monotonic), so time
# counted on the processed audio approximates the time compression was applied. The band split differs from
# acrossover and is off by 0.35dB on average (masterer.go's trebleThresholdOffsetDB).
#
# The "GR" column works back from the static curve how loud each frame over the threshold was before
# processing, and averages the dB removed over the section; frames under the threshold count as 0. The time
# share alone does not give the amount: on 24 songs from 2026-10-28, verses spent a median 7 % over the
# threshold, above the 5 % the design assumed, but the average reduction was only 0.35dB (1.84dB in choruses);
# what crossed it were brief peaks such as cymbals and consonants. Knee and attack/release are ignored, so it
# is not precise enough to tune in 0.5dB steps.

# The mastering's acompressor=ratio=4. Only used to work back the gain reduction.
COMPRESSOR_RATIO = 4.0


@dataclass
class BandLevels:
    section: Section
    low: float
    # presence is 4-6kHz; not compressed since 2026-09-05.
    presence: float
    # treble is the compressed 6-10kHz; air is above 10kHz.
    treble: float
    air: float
    # treble_peak is the high percentile of the 6-10kHz short-term level; over_ratio is the share of time it
    # exceeds the threshold. None when the threshold is unknown.
    treble_peak: float
    over_ratio: float | None
    # gain_reduction is the section average of the 6-10kHz reduction (dB). None when the threshold is unknown.
    gain_reduction: float | None
    # low_punch is the gap between the high percentile and the median of the 40-120Hz short-term level (dB);
    # see LOW_PUNCH_PERCENTILES.
    low_punch: float


def analyse(audio_path: Path, recipe: Recipe, threshold_dbfs: float | None = None) -> list[BandLevels]:
    """Measure band levels per section.

    threshold_dbfs is the compression threshold as seen on master.wav (MasteringSettings.master_threshold_dbfs).
    Without it, over_ratio and gain_reduction are None.
    """
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    seconds = len(mono) / sr

    lo_cut, hi_cut = CROSSOVER_HZ
    bands = {
        "low": _band(mono, sr, None, PRESENCE_HZ),
        "presence": _band(mono, sr, PRESENCE_HZ, lo_cut),
        "treble": _band(mono, sr, lo_cut, hi_cut),
        "air": _band(mono, sr, hi_cut, None),
        "kick": _band(mono, sr, *LOW_PUNCH_HZ),
    }

    levels = []
    for section in recipe.sections:
        start, end = int(section.start * sr), int(min(section.end, seconds) * sr)
        if end - start <= 0:
            continue
        short_term = _short_term_dbfs(bands["treble"][start:end], sr)
        levels.append(
            BandLevels(
                section=section,
                low=dbfs(bands["low"][start:end]),
                presence=dbfs(bands["presence"][start:end]),
                treble=dbfs(bands["treble"][start:end]),
                air=dbfs(bands["air"][start:end]),
                treble_peak=float(np.percentile(short_term, 95)) if short_term.size else SILENCE_DBFS,
                over_ratio=None
                if threshold_dbfs is None
                else float(np.mean(short_term > threshold_dbfs)) if short_term.size else 0.0,
                gain_reduction=None if threshold_dbfs is None else gain_reduction(short_term, threshold_dbfs),
                low_punch=low_punch(bands["kick"][start:end], sr),
            )
        )
    return levels


def gain_reduction(short_term: np.ndarray, threshold_dbfs: float) -> float:
    """Work back the average reduction (dB) applied by the compressor from the processed short-term level.

    With a static ratio R, an input d over the threshold comes out d / R over. A frame e over after processing
    was e * R over before, so the reduction is e * (R - 1).
    """
    if not short_term.size:
        return 0.0
    excess = np.maximum(short_term - threshold_dbfs, 0.0)
    return float(np.mean(excess * (COMPRESSOR_RATIO - 1.0)))


def low_punch(kick_band: np.ndarray, sr: int) -> float:
    """From the 40-120Hz signal, return the gap between the high percentile and the median of the short-term level."""
    short_term = _short_term_dbfs(kick_band, sr)
    if not short_term.size:
        return 0.0
    high, mid = np.percentile(short_term, LOW_PUNCH_PERCENTILES)
    return float(high - mid)


def _short_term_dbfs(x: np.ndarray, sr: int) -> np.ndarray:
    """RMS per short window in dBFS, the quantity a compressor's detector sees."""
    window = int(DETECTOR_WINDOW_SECONDS * sr)
    hop = int(DETECTOR_HOP_SECONDS * sr)
    if x.size < window:
        return np.array([])

    starts = np.arange(0, x.size - window, hop)
    frames = np.stack([x[s : s + window] for s in starts])
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def render(title: str, levels: list[BandLevels], threshold_dbfs: float | None = None) -> str:
    lines = [
        f"\n{title}",
        f"{'section':<11}{'<4k':>9}{'4-6k':>9}{'6-10k':>9}{'>10k':>9}"
        f"{'6-10k short':>11}{'over':>8}{'GR':>9}{'punch':>8}",
        "-" * 86,
    ]
    for lv in levels:
        lines.append(
            f"{lv.section.name:<11}{lv.low:>7.1f}dB{lv.presence:>7.1f}dB"
            f"{lv.treble:>7.1f}dB{lv.air:>7.1f}dB"
            f"{lv.treble_peak:>11.1f}dB"
            f"{'-' if lv.over_ratio is None else f'{lv.over_ratio:.0%}':>8}"
            f"{'-' if lv.gain_reduction is None else f'{lv.gain_reduction:.2f}dB':>9}"
            f"{lv.low_punch:>8.1f}dB"
        )

    sung = [lv for lv in levels if not lv.section.instrumental]
    chorus = [lv for lv in sung if "Chorus" in lv.section.name]
    verse = [lv for lv in sung if "Verse" in lv.section.name]
    lines.append("-" * 86)
    if chorus and verse:
        c = float(np.mean([lv.treble for lv in chorus]))
        v = float(np.mean([lv.treble for lv in verse]))
        cp = float(np.mean([lv.treble_peak for lv in chorus]))
        lines.append(f"6-10k chorus {c:.1f}dB / verse {v:.1f}dB / diff {c - v:.1f}dB")
        if threshold_dbfs is None:
            lines.append(
                f"chorus short-term level {cp:.1f}dB / threshold unknown"
                " (assessment.json has no mastering record; the over column is left out too)"
            )
        else:
            lines.append(
                f"chorus short-term level {cp:.1f}dB / threshold {threshold_dbfs:.1f}dB (on master.wav)"
                f"  → {'compressed' if cp > threshold_dbfs else 'below the threshold'}"
            )
        # 4-6kHz left the compressed band on 2026-09-05. This line tracks across songs whether the core of the
        # voice survives; it is not used for any judgement.
        cpr = float(np.mean([lv.presence for lv in chorus]))
        vpr = float(np.mean([lv.presence for lv in verse]))
        lines.append(
            f"4-6k (unprocessed) chorus {cpr:.1f}dB / verse {vpr:.1f}dB / diff {cpr - vpr:.1f}dB"
        )

    # Sung sections other than Chorus / Verse (e.g. the Bridge of an 8-section layout). The two summary lines
    # check the "tame only the chorus" design, so these are listed separately rather than mixed in.
    others = [lv for lv in sung if "Chorus" not in lv.section.name and "Verse" not in lv.section.name]
    if others:
        lines.append(
            "other sung sections: "
            + " / ".join(f"{lv.section.name} {lv.treble:.1f}dB" for lv in others)
        )
    return "\n".join(lines)


def recorded_threshold(audio: Path, assessments: Path) -> float | None:
    """Return the threshold recorded in assessments/<audio name>.json, as a value on master.wav."""
    path = assessments / f"{audio.stem}.json"
    if not path.exists():
        return None
    mastering = load_assessment(path).mastering
    return None if mastering is None else mastering.master_threshold_dbfs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, nargs="+", help="audio and recipe, alternating")
    parser.add_argument("--assessments", type=Path, default=Path("assessments"),
                        help="where the assessment.json files are (the compression threshold is read from them)")
    args = parser.parse_args()

    if len(args.pairs) % 2:
        raise SystemExit("pass audio and recipe in pairs")

    for audio, recipe_path in zip(args.pairs[::2], args.pairs[1::2], strict=False):
        rec = recipe_mod.load(recipe_path)
        threshold = recorded_threshold(audio, args.assessments)
        print(render(f"{rec.title}  ({audio.stem})", analyse(audio, rec, threshold), threshold))


if __name__ == "__main__":
    main()
