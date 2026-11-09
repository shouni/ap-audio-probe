"""Cut the vocal stem at section boundaries and measure how much voice each section has."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .recipe import Recipe, Section

# Every verdict is relative to the song's own sung sections. demucs never removes the backing completely;
# a little leaks, and neither the leak nor the song's level is consistent, so absolute thresholds fail
# across songs. The reference is sung_reference().

# A section whose RMS is within this margin of the reference is taken to carry voice.
VOCAL_MARGIN_DB = 12.0

# Frames louder than the reference minus this are counted as voiced.
FRAME_FLOOR_DB = 20.0
FRAME_SECONDS = 0.05

# More voiced frames than this share also means "vocals". The RMS condition alone only reaches injections
# 12dB below the singing; this is what catches -18dB (scripts/calibrate.py).
ACTIVE_RATIO_THRESHOLD = 0.20

# A line can start before the boundary (a pickup) and a held ending can run past it. Up to this length it is
# treated as bleed from the neighbour and excluded from the instrumental section. The bleed seen in practice
# was 0.9 s (about two beats at 156 BPM); this cap doubles that while staying shorter than a phrase.
BOUNDARY_BLEED_SECONDS = 2.0

SILENCE_DBFS = -120.0


def dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(rms) if rms > 0 else SILENCE_DBFS


def frame_levels(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """Short-time RMS per FRAME_SECONDS in dBFS. A trailing partial frame is dropped."""
    frame = int(FRAME_SECONDS * sample_rate)
    if frame <= 0 or x.size < frame:
        return np.empty(0, dtype=np.float64)
    usable = x[: len(x) // frame * frame].reshape(-1, frame)
    return 20.0 * np.log10(
        np.maximum(np.sqrt(np.mean(np.square(usable), axis=1)), 1e-12)
    )


def sung_reference(mono: np.ndarray, sample_rate: int, recipe: Recipe) -> float:
    """Median RMS of the sung sections: the song's own yardstick that every verdict is relative to."""
    seconds = len(mono) / sample_rate
    levels = []
    for section in recipe.sections:
        if section.instrumental:
            continue
        chunk = mono[int(section.start * sample_rate) : int(min(section.end, seconds) * sample_rate)]
        if chunk.size:
            levels.append(dbfs(chunk))
    return float(np.median(levels)) if levels else SILENCE_DBFS


def _run_length(voiced: np.ndarray, *, from_end: bool) -> int:
    """Number of consecutive voiced frames from the start (or the end)."""
    ordered = voiced[::-1] if from_end else voiced
    silent = np.flatnonzero(~ordered)
    return int(silent[0]) if silent.size else int(ordered.size)


@dataclass
class SectionStats:
    section: Section
    clamped_end: float
    rms_dbfs: float
    peak_dbfs: float
    active_ratio: float
    # seconds excluded from the measurement as bleed from a neighbouring sung section
    head_bleed: float = 0.0
    tail_bleed: float = 0.0

    @property
    def truncated(self) -> bool:
        return self.clamped_end < self.section.end - 0.05

    @property
    def bleed(self) -> float:
        return self.head_bleed + self.tail_bleed


@dataclass
class Report:
    recipe: Recipe
    audio_seconds: float
    stats: list[SectionStats]
    sung_reference_dbfs: float

    @property
    def missing_tail(self) -> float:
        return self.recipe.declared_duration - self.audio_seconds

    def has_vocals(self, s: SectionStats) -> bool:
        """Vocals if either the section average or the time with voice reaches its threshold.

        They play different roles, so neither can go; see ACTIVE_RATIO_THRESHOLD for why.
        """
        return (
            s.rms_dbfs > self.sung_reference_dbfs - VOCAL_MARGIN_DB
            or s.active_ratio > ACTIVE_RATIO_THRESHOLD
        )

    @property
    def violations(self) -> list[SectionStats]:
        """Sections marked instrumental that carry voice."""
        return [s for s in self.stats if s.section.instrumental and self.has_vocals(s)]


def analyse(vocals_wav: Path, recipe: Recipe) -> Report:
    data, sr = sf.read(vocals_wav, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    audio_seconds = len(mono) / sr

    def slice_of(s: Section) -> tuple[np.ndarray, float]:
        end = min(s.end, audio_seconds)
        lo, hi = int(s.start * sr), int(end * sr)
        return mono[lo:hi], end

    # Compute the sung-section reference first; every verdict is measured against it.
    reference = sung_reference(mono, sr, recipe)

    frame = int(FRAME_SECONDS * sr)
    floor = reference - FRAME_FLOOR_DB

    def sung_neighbour(index: int, offset: int) -> bool:
        neighbour = index + offset
        return 0 <= neighbour < len(recipe.sections) and not recipe.sections[neighbour].instrumental

    stats: list[SectionStats] = []
    for index, section in enumerate(recipe.sections):
        chunk, end = slice_of(section)
        frame_db = frame_levels(chunk, sr)
        voiced = frame_db > floor

        # Only the edge next to a sung section is excluded; voice in the middle stays. A run longer than the
        # cap is measured whole as part of the section, because excluding part of it would report only the
        # remainder as a violation and make the numbers unreadable.
        head = tail = 0
        if section.instrumental and voiced.size and not voiced.all():
            limit = int(BOUNDARY_BLEED_SECONDS / FRAME_SECONDS)
            if sung_neighbour(index, -1):
                head = _run_length(voiced, from_end=False)
            if sung_neighbour(index, +1):
                tail = _run_length(voiced, from_end=True)
            head = head if head <= limit else 0
            tail = tail if tail <= limit else 0

        core = chunk[head * frame : len(chunk) - tail * frame] if head or tail else chunk
        core_db = frame_levels(core, sr) if (head or tail) else frame_db

        peak = float(np.max(np.abs(core))) if core.size else 0.0
        stats.append(
            SectionStats(
                section=section,
                clamped_end=end,
                rms_dbfs=dbfs(core),
                peak_dbfs=20.0 * np.log10(peak) if peak > 0 else SILENCE_DBFS,
                active_ratio=float(np.mean(core_db > floor)) if core_db.size else 0.0,
                head_bleed=head * FRAME_SECONDS,
                tail_bleed=tail * FRAME_SECONDS,
            )
        )

    return Report(
        recipe=recipe,
        audio_seconds=audio_seconds,
        stats=stats,
        sung_reference_dbfs=reference,
    )
