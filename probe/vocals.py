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

# 歌い出しは境界の手前から始まり(弱起)、歌尾は境界を越えて伸びます。この長さまでは
# 隣からの食い込みと見なし、インスト区間の中身から外して測ります。実測では
# 「ステートレスな夜に」の Interlude(78-92s)で声が立つのは最後の 0.9 秒だけ
# なのに、区間平均が -50.9dB から -30.0dB へ跳ねて違反として上がりました。
# 上限は、区間の大半を占める声まで見逃さないために要ります。
BOUNDARY_BLEED_SECONDS = 2.0

SILENCE_DBFS = -120.0


def _dbfs(x: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(rms) if rms > 0 else SILENCE_DBFS


def frame_levels(x: np.ndarray, sample_rate: int) -> np.ndarray:
    """FRAME_SECONDS ごとの短時間 RMS を dBFS で返す。端数のフレームは捨てる。"""
    frame = int(FRAME_SECONDS * sample_rate)
    if frame <= 0 or x.size < frame:
        return np.empty(0, dtype=np.float64)
    usable = x[: len(x) // frame * frame].reshape(-1, frame)
    return 20.0 * np.log10(
        np.maximum(np.sqrt(np.mean(np.square(usable), axis=1)), 1e-12)
    )


def sung_reference(mono: np.ndarray, sample_rate: int, recipe: Recipe) -> float:
    """歌唱セクションの RMS 中央値。曲ごとに違うレベルを揃えるための物差し。

    絶対値で切ると曲を跨いで機能しないため、判定はすべてこの値からの相対で決めます。
    """
    seconds = len(mono) / sample_rate
    levels = []
    for section in recipe.sections:
        if section.instrumental:
            continue
        chunk = mono[int(section.start * sample_rate) : int(min(section.end, seconds) * sample_rate)]
        if chunk.size:
            levels.append(_dbfs(chunk))
    return float(np.median(levels)) if levels else SILENCE_DBFS


def _run_length(voiced: np.ndarray, *, from_end: bool) -> int:
    """先頭(または末尾)から続く有声フレームの本数。"""
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
    # 隣の歌唱セクションから食い込んだぶんとして、測定から外した秒数。
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

        # 外すのは隣が歌唱セクションである側の端だけで、中ほどに現れる声はそのまま
        # 残ります。上限を超える長さは、途中まで外すと残りだけが違反として上がって
        # 数字が読めなくなるため、まるごと区間の中身として測ります。
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
                rms_dbfs=_dbfs(core),
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
