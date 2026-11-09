"""Compare the assessment.json the generator left with the local audio.

The generator writes `music/<job_id>/assessment.json` per job: the values it measured at generation time and
the review of the model that listened to it. This lays that record next to the same quantities measured again
locally.

    python -m probe.assessment audio/foo.wav assessments/foo.json

**They measure different things.** The generator measures the mp3 served on the web (`Inspect(mastered.Web)`;
true peak and LUFS only come from the wav via `InspectMaster` when `loudness_from_master` is true), while this
measures the distribution master wav. Differences are differences between those two, not errors on either
side. The point is to separate the differences that have an explanation (the PADDING / peak_amplitude notes
below) from those that do not.

The measurement is reimplemented to match the generator's inspector.go, so the same numbers come out of a
different tool. If the implementations drifted, it would be unclear whether a difference is "mp3 vs wav" or
"a different method".
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt

from .loudness import measure as measure_loudness
from .table import cell

# Constants from the generator's inspector.go, with the same names and values.
ANALYSIS_WINDOW_SECONDS = 0.05
SILENCE_THRESHOLD_RMS = 0.005
CLIPPING_THRESHOLD = 0.999
MIN_AUDIO_DBFS = -120.0

# The Inspector's bands. Not the mastering's compressed band (spectrum.py: CROSSOVER_HZ); after compression
# moved to start at 6kHz on 2026-09-05, this record stayed at 4-10kHz.
MID_BAND_HZ = (200.0, 4000.0)
HIGH_BAND_HZ = (4000.0, 10000.0)

# Padding added by the mp3 decoder: 2,304 samples = two MP3 frames, so the generator reports a length longer
# by this much. 0.052 s at 44.1kHz.
MP3_PADDING_SAMPLES = 2304

# Tolerances for flagging unexplained differences. Encoding differences between mp3 and wav measured around
# 0.1dB, so these sit an order above. Length is judged after subtracting the padding.
LEVEL_TOLERANCE_DB = 0.5
DURATION_TOLERANCE_SECONDS = 0.2
RATIO_TOLERANCE = 0.02


@dataclass(frozen=True)
class Measurements:
    """The same fields as inspector.go's AudioMeasurements."""

    duration_seconds: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    longest_internal_silence_seconds: float
    silent_ratio: float
    peak_amplitude: float
    clipped_sample_ratio: float
    rms_dbfs: float
    mid_band_dbfs: float
    high_band_dbfs: float
    true_peak_dbtp: float
    integrated_lufs: float

    @property
    def treble_headroom_db(self) -> float:
        return self.mid_band_dbfs - self.high_band_dbfs


@dataclass(frozen=True)
class MasteringSettings:
    """The mastering constants the generator applied to the song (MasteringSettings).

    Records before 2026-10-29 lack them, and then Record.mastering is None.
    """

    # 6-10kHz average measured on the unprocessed input; None when the generator could not measure it and fell back to a fixed threshold.
    treble_dbfs: float | None
    treble_threshold_dbfs: float
    loudness_gain_db: float

    @property
    def master_threshold_dbfs(self) -> float:
        """The threshold as seen on master.wav.

        The compressor acts before loudness matching, so drawing the line on the processed audio needs the gain added.
        """
        return self.treble_threshold_dbfs + self.loudness_gain_db


@dataclass(frozen=True)
class Record:
    """The parts of assessment.json (the generator's AudioAssessmentRecord) used here."""

    job_id: str
    created_at: str
    score: float
    summary: str
    passed: bool
    mastered: bool
    measurements: Measurements
    issues: list[tuple[str, str]]
    review: dict | None
    lyrics: dict | None
    mastering: MasteringSettings | None = None


def _bandpass_sos(sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    """The same second-order band-pass as inspector.go's newBandFilter (RBJ Cookbook).

    This is used instead of spectrum.py's fourth-order Butterworth to reproduce the mid/high band numbers the
    generator recorded. A single-stage band-pass has gentle skirts, and plenty of kick and bass leaks into the
    main band (inspector.go notes this). Use it only to compare with the record, not as an absolute band level.
    """
    center = np.sqrt(low_hz * high_hz)
    q = center / (high_hz - low_hz)
    w0 = 2.0 * np.pi * center / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    return np.array([[alpha / a0, 0.0, -alpha / a0, 1.0, -2.0 * np.cos(w0) / a0, (1.0 - alpha) / a0]])


def _dbfs(amplitude: float) -> float:
    return max(MIN_AUDIO_DBFS, 20.0 * np.log10(amplitude)) if amplitude > 0 else MIN_AUDIO_DBFS


def _silence(windows: np.ndarray) -> tuple[float, float, float, float]:
    """From the sequence of silent windows, return leading, trailing, longest internal, and the overall share.

    Counted as in inspector.go's summariseSilence (the longest internal run looks only inside, excluding the
    leading and trailing runs; the share is over all windows, including those).
    """
    if windows.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    silent = windows.tolist()
    leading = 0
    for value in silent:
        if not value:
            break
        leading += 1
    if leading == len(silent):
        return float(leading), 0.0, 0.0, 1.0

    trailing = 0
    for value in reversed(silent):
        if not value:
            break
        trailing += 1

    run = longest = internal = 0
    for value in silent[leading : len(silent) - trailing]:
        if value:
            run += 1
            longest = max(longest, run)
            internal += 1
            continue
        run = 0

    return float(leading), float(trailing), float(longest), (internal + leading + trailing) / len(silent)


def measure(path: Path) -> Measurements:
    """Measure the audio the same way as the generator's Inspector.

    The key is measuring one waveform with the channels averaged. The generator folds all channels with
    decodeFrame → meanSample before taking the amplitude peak, so `peak_amplitude` always comes out lower than
    each channel's peak by the stereo difference (same mp3: 0.903 per channel / 0.867 averaged). This folds the
    same way so like is compared with like. True peak and loudness are defined per channel by their standards,
    so those use loudness.py (ffmpeg).
    """
    data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)

    window = max(int(sample_rate * ANALYSIS_WINDOW_SECONDS), 1)
    full = mono[: mono.size // window * window].reshape(-1, window)
    window_rms = np.sqrt(np.mean(np.square(full), axis=1)) if full.size else np.empty(0)
    # include the partial last window in the silence check (so a trailing fade-out is not dropped)
    remainder = mono[full.size :]
    if remainder.size:
        window_rms = np.append(window_rms, np.sqrt(np.mean(np.square(remainder))))

    leading, trailing, longest, silent_ratio = _silence(window_rms < SILENCE_THRESHOLD_RMS)
    window_seconds = window / sample_rate

    mid = sosfilt(_bandpass_sos(sample_rate, *MID_BAND_HZ), mono)
    high = sosfilt(_bandpass_sos(sample_rate, *HIGH_BAND_HZ), mono)
    loudness = measure_loudness(path)

    return Measurements(
        duration_seconds=mono.size / sample_rate,
        leading_silence_seconds=leading * window_seconds,
        trailing_silence_seconds=trailing * window_seconds,
        longest_internal_silence_seconds=longest * window_seconds,
        silent_ratio=silent_ratio,
        peak_amplitude=float(np.abs(mono).max()) if mono.size else 0.0,
        clipped_sample_ratio=float(np.mean(np.abs(mono) >= CLIPPING_THRESHOLD)) if mono.size else 0.0,
        rms_dbfs=_dbfs(float(np.sqrt(np.mean(np.square(mono))))),
        mid_band_dbfs=_dbfs(float(np.sqrt(np.mean(np.square(mid))))),
        high_band_dbfs=_dbfs(float(np.sqrt(np.mean(np.square(high))))),
        true_peak_dbtp=loudness.true_peak_dbtp,
        integrated_lufs=loudness.integrated_lufs,
    )


def load(path: Path) -> Record:
    data = json.loads(path.read_text(encoding="utf-8"))
    assessment = data.get("assessment") or {}
    quality = assessment.get("quality") or {}
    m = quality.get("measurements") or {}

    return Record(
        job_id=data.get("job_id", ""),
        created_at=data.get("created_at", ""),
        score=float(data.get("score", 0.0)),
        summary=data.get("summary", ""),
        passed=bool(data.get("passed", False)),
        mastered=bool(assessment.get("mastered", False)),
        measurements=Measurements(
            duration_seconds=float(m.get("duration_seconds", 0.0)),
            leading_silence_seconds=float(m.get("leading_silence_seconds", 0.0)),
            trailing_silence_seconds=float(m.get("trailing_silence_seconds", 0.0)),
            longest_internal_silence_seconds=float(m.get("longest_internal_silence_seconds", 0.0)),
            silent_ratio=float(m.get("silent_ratio", 0.0)),
            peak_amplitude=float(m.get("peak_amplitude", 0.0)),
            clipped_sample_ratio=float(m.get("clipped_sample_ratio", 0.0)),
            rms_dbfs=float(m.get("rms_dbfs", 0.0)),
            mid_band_dbfs=float(m.get("mid_band_dbfs", 0.0)),
            high_band_dbfs=float(m.get("high_band_dbfs", 0.0)),
            true_peak_dbtp=float(m.get("true_peak_dbtp", 0.0)),
            integrated_lufs=float(m.get("integrated_lufs", 0.0)),
        ),
        issues=[(i.get("kind", ""), i.get("detail", "")) for i in (quality.get("issues") or [])],
        review=assessment.get("review"),
        lyrics=assessment.get("lyrics"),
        mastering=_mastering(assessment.get("mastering")),
    )


def _mastering(data: dict | None) -> MasteringSettings | None:
    if not data or "treble_threshold_dbfs" not in data:
        return None
    treble = data.get("treble_dbfs")
    return MasteringSettings(
        treble_dbfs=None if treble is None else float(treble),
        treble_threshold_dbfs=float(data["treble_threshold_dbfs"]),
        loudness_gain_db=float(data.get("loudness_gain_db", 0.0)),
    )


@dataclass(frozen=True)
class Comparison:
    label: str
    recorded: float
    measured: float
    unit: str
    # note is a remark shown on the row, for the reader; it does not change the verdict.
    note: str = ""
    tolerance: float = 0.0
    # explained means "this difference is already accounted for". It is kept apart from note so that items whose
    # remark always shows (how peak_amplitude is measured, for example) still flag a large difference. Silencing
    # the warning whenever a remark exists would let a method note excuse any difference.
    explained: bool = False

    @property
    def diff(self) -> float:
        return self.measured - self.recorded

    @property
    def unexplained(self) -> bool:
        return not self.explained and abs(self.diff) > self.tolerance


def compare(record: Record, measured: Measurements, sample_rate: int = 44100) -> list[Comparison]:
    """Line up the record and the remeasurement per item, with a remark where the difference has an explanation."""
    r, m = record.measurements, measured
    padding = MP3_PADDING_SAMPLES / sample_rate

    padded = abs((r.duration_seconds - m.duration_seconds) - padding) < DURATION_TOLERANCE_SECONDS

    return [
        Comparison(
            "length", r.duration_seconds, m.duration_seconds, "s",
            f"explained by the mp3 padding of {padding * 1000:.0f}ms" if padded else "",
            DURATION_TOLERANCE_SECONDS, explained=padded,
        ),
        # The tolerance covers the overshoot from mp3 encoding (channel-averaged: wav 0.887 / mp3 0.867, a 0.02
        # gap). Doubled; anything beyond it points at the method.
        Comparison(
            "peak_amplitude", r.peak_amplitude, m.peak_amplitude, "",
            "both measured on the channel-averaged waveform", 0.05,
        ),
        Comparison("true peak", r.true_peak_dbtp, m.true_peak_dbtp, "dBTP", "", LEVEL_TOLERANCE_DB),
        Comparison("integrated", r.integrated_lufs, m.integrated_lufs, "LUFS", "", LEVEL_TOLERANCE_DB),
        Comparison("RMS", r.rms_dbfs, m.rms_dbfs, "dBFS", "", LEVEL_TOLERANCE_DB),
        Comparison("200Hz-4kHz", r.mid_band_dbfs, m.mid_band_dbfs, "dBFS", "", LEVEL_TOLERANCE_DB),
        Comparison("4-10kHz", r.high_band_dbfs, m.high_band_dbfs, "dBFS", "", LEVEL_TOLERANCE_DB),
        Comparison(
            "treble headroom", r.treble_headroom_db, m.treble_headroom_db, "dB", "", LEVEL_TOLERANCE_DB
        ),
        Comparison("leading silence", r.leading_silence_seconds, m.leading_silence_seconds, "s", "", DURATION_TOLERANCE_SECONDS),
        Comparison("trailing silence", r.trailing_silence_seconds, m.trailing_silence_seconds, "s", "", DURATION_TOLERANCE_SECONDS),
        Comparison("longest inner silence", r.longest_internal_silence_seconds, m.longest_internal_silence_seconds, "s", "", DURATION_TOLERANCE_SECONDS),
        Comparison("silent share", r.silent_ratio, m.silent_ratio, "", "", RATIO_TOLERANCE),
        Comparison("clipped share", r.clipped_sample_ratio, m.clipped_sample_ratio, "", "", RATIO_TOLERANCE),
    ]


def render(record: Record, measured: Measurements, audio_name: str) -> str:
    lines = [
        f"\n{record.job_id or '(no job_id)'}  recorded {record.created_at}",
        f"score {record.score:.2f} / measured {'pass' if record.passed else 'fail'}"
        f" / mastering {'applied' if record.mastered else 'not applied'}",
        f"summary: {record.summary}",
        "",
        cell("item", 18) + cell("generator", 14, right=True)
        + cell(f"local({audio_name})", 16, right=True) + cell("diff", 10, right=True),
        "-" * 76,
    ]

    unexplained = []
    for c in compare(record, measured):
        mark = " ⚠" if c.unexplained else ""
        lines.append(
            cell(c.label, 18)
            + f"{c.recorded:>14.3f}{c.measured:>16.3f}{c.diff:>+10.3f}{mark}"
            + (f"   {c.note}" if c.note else "")
        )
        if c.unexplained:
            unexplained.append(c)

    lines.append("-" * 76)
    if record.mastering:
        mst = record.mastering
        source = "fell back to the fixed value" if mst.treble_dbfs is None else f"from the input 6-10kHz {mst.treble_dbfs:.1f}dBFS"
        lines.append(
            f"mastering: 6-10kHz threshold {mst.treble_threshold_dbfs:.1f}dBFS ({source})"
            f" / loudness gain {mst.loudness_gain_db:+.2f}dB"
        )
    if record.issues:
        lines.append("issues raised by the generator:")
        lines.extend(f"  [{kind}] {detail}" for kind, detail in record.issues)

    if record.review:
        r = record.review
        lines.append(
            f"review: lyrics {r.get('lyrics_intelligibility', 0)}/5 / structure {r.get('structure_match', 0)}/5"
            f" / pacing {r.get('pacing', 0)}/5 / vocal {r.get('vocal_balance', 0)}/5"
        )
        lines.extend(f"  - {f}" for f in (r.get("findings") or []))

    lyrics = record.lyrics
    if lyrics and (lyrics.get("missing_lines") or lyrics.get("extra_lines")):
        lines.append(
            f"score lines: missing {lyrics.get('missing_lines', 0)} / extra {lyrics.get('extra_lines', 0)}"
            "  * compares the score the model returned, not the audio (that is check_lyrics.py\'s job)"
        )

    lines.append("")
    if unexplained:
        lines.append(f"⚠ {len(unexplained)} unexplained difference(s):")
        lines.extend(f"   {c.label} {c.diff:+.3f}{c.unit}" for c in unexplained)
        lines.append("  mp3 vs wav differences measure around 0.1dB. Beyond that, suspect the method or the target.")
    else:
        lines.append("✓ the record and the local measurement agree within what mp3 vs wav explains.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="audio to measure (master.wav recommended)")
    parser.add_argument("assessment", type=Path, help="the generator's assessment.json")
    args = parser.parse_args()

    record = load(args.assessment)
    print(render(record, measure(args.audio), args.audio.suffix.lstrip(".") or "local"))


if __name__ == "__main__":
    main()
