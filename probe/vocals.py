"""ボーカル stem をセクション境界で切り、区間ごとの声の量を測る。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from .recipe import Recipe, Section

# 歌唱セクションの中央値からこの幅に収まっていれば、そこに声が乗っていると見なす。
# demucs は伴奏を完全には除去しきれず微量が漏れるため、絶対値ではなく
# 「その曲の歌唱区間と比べてどうか」で判断する。
VOCAL_MARGIN_DB = 12.0

# 有声フレームと見なす下限。同じく歌唱区間からの相対で決める。
FRAME_FLOOR_DB = 20.0
FRAME_SECONDS = 0.05

SILENCE_DBFS = -120.0


def _dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(rms) if rms > 0 else SILENCE_DBFS


@dataclass
class SectionStats:
    section: Section
    clamped_end: float
    rms_dbfs: float
    peak_dbfs: float
    active_ratio: float

    @property
    def truncated(self) -> bool:
        return self.clamped_end < self.section.end - 0.05


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
        return (
            s.rms_dbfs > self.sung_reference_dbfs - VOCAL_MARGIN_DB
            or s.active_ratio > 0.20
        )

    @property
    def violations(self) -> list[SectionStats]:
        """インスト指定なのに声が乗っているセクション。"""
        return [s for s in self.stats if s.section.instrumental and self.has_vocals(s)]


def analyse(vocals_wav: Path, recipe: Recipe) -> Report:
    data, sr = sf.read(vocals_wav, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    audio_seconds = len(mono) / sr

    def slice_of(s: Section) -> tuple[np.ndarray, float]:
        end = min(s.end, audio_seconds)
        lo, hi = int(s.start * sr), int(end * sr)
        return mono[lo:hi], end

    # 先に歌唱セクションの基準値を出す。これが全判定の物差しになる。
    sung = [slice_of(s)[0] for s in recipe.sections if not s.instrumental]
    sung_levels = [_dbfs(x) for x in sung if x.size]
    reference = float(np.median(sung_levels)) if sung_levels else SILENCE_DBFS

    frame = int(FRAME_SECONDS * sr)
    floor = reference - FRAME_FLOOR_DB

    stats: list[SectionStats] = []
    for section in recipe.sections:
        chunk, end = slice_of(section)
        if chunk.size >= frame:
            usable = chunk[: len(chunk) // frame * frame].reshape(-1, frame)
            frame_db = 20.0 * np.log10(
                np.maximum(np.sqrt(np.mean(np.square(usable), axis=1)), 1e-12)
            )
            active = float(np.mean(frame_db > floor))
        else:
            active = 0.0

        peak = float(np.max(np.abs(chunk))) if chunk.size else 0.0
        stats.append(
            SectionStats(
                section=section,
                clamped_end=end,
                rms_dbfs=_dbfs(chunk),
                peak_dbfs=20.0 * np.log10(peak) if peak > 0 else SILENCE_DBFS,
                active_ratio=active,
            )
        )

    return Report(
        recipe=recipe,
        audio_seconds=audio_seconds,
        stats=stats,
        sung_reference_dbfs=reference,
    )
