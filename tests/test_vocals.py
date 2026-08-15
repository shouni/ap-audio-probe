"""判定ロジックの回帰テスト。

demucs を通す実地の陽性対照は scripts/calibrate.py にあり、数分かかります。
ここでは合成した stem を analyse に直接食わせ、判定部分だけを速く確かめます。
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from probe.recipe import Recipe, Section
from probe.vocals import BOUNDARY_BLEED_SECONDS, analyse

SR = 22050
INSTRUMENTAL_PROMPT = "[Instrumental Interlude] Instruments only — no lead vocal."
SUNG_PROMPT = "[Verse] Focus on the lyrics marked [Verse]."

# 食い込みはフレーム単位(FRAME_SECONDS)でしか切れないため、その丸めぶんを許容する。
FRAME_TOLERANCE = 0.06


def _recipe() -> Recipe:
    return Recipe(
        title="テスト",
        tempo=120,
        sections=[
            Section("Intro", 0.0, 4.0, INSTRUMENTAL_PROMPT),
            Section("Verse", 4.0, 12.0, SUNG_PROMPT),
            Section("Interlude", 12.0, 16.0, INSTRUMENTAL_PROMPT),
            Section("Chorus", 16.0, 24.0, SUNG_PROMPT),
        ],
    )


def _write_stem(path, levels: dict[str, float], seconds: float = 24.0):
    """区間ごとの振幅を指定して、ボーカル stem に見立てた wav を書く。"""
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    envelope = np.zeros_like(tone)
    for section in _recipe().sections:
        lo, hi = int(section.start * SR), int(min(section.end, seconds) * SR)
        envelope[lo:hi] = levels[section.name]
    sf.write(path, tone * envelope, SR)
    return path


@pytest.fixture
def stem_path(tmp_path):
    return tmp_path / "vocals.wav"


def test_clean_track_reports_no_violation(stem_path):
    _write_stem(stem_path, {"Intro": 0.001, "Verse": 0.1, "Interlude": 0.001, "Chorus": 0.1})

    report = analyse(stem_path, _recipe())

    assert report.violations == []


def test_vocals_in_instrumental_section_are_detected(stem_path):
    """インスト指定区間に歌唱と同程度の声が乗っていれば検出する。"""
    _write_stem(stem_path, {"Intro": 0.001, "Verse": 0.1, "Interlude": 0.05, "Chorus": 0.1})

    report = analyse(stem_path, _recipe())

    assert [s.section.name for s in report.violations] == ["Interlude"]


def test_quiet_bleed_is_not_flagged(stem_path):
    """分離の漏れ程度の微量は違反にしない。ここを拾うと狼少年になる。"""
    _write_stem(stem_path, {"Intro": 0.001, "Verse": 0.1, "Interlude": 0.0015, "Chorus": 0.1})

    report = analyse(stem_path, _recipe())

    assert report.violations == []


def test_sung_sections_never_count_as_violations(stem_path):
    """歌唱区間に声があるのは当然で、違反にはならない。"""
    _write_stem(stem_path, {"Intro": 0.001, "Verse": 0.1, "Interlude": 0.001, "Chorus": 0.1})

    report = analyse(stem_path, _recipe())

    assert all(s.section.instrumental for s in report.violations)
    assert report.sung_reference_dbfs == pytest.approx(-23.0, abs=1.0)


def _write_stem_with_bleed(path, bleed_seconds: float, *, at_end: bool = True, seconds: float = 24.0):
    """Interlude (12-16s) の端に、歌唱と同じレベルの声が食い込んだ stem を書く。"""
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    envelope = np.full_like(tone, 0.001)
    for section in _recipe().sections:
        if section.instrumental:
            continue
        envelope[int(section.start * SR) : int(section.end * SR)] = 0.1

    if at_end:
        lo, hi = 16.0 - bleed_seconds, 16.0
    else:
        lo, hi = 12.0, 12.0 + bleed_seconds
    envelope[int(lo * SR) : int(hi * SR)] = 0.1

    sf.write(path, tone * envelope, SR)
    return path


def test_pickup_into_the_end_of_an_instrumental_section_is_not_flagged(stem_path):
    """次の歌唱セクションの歌い出しが食い込むのは弱起で、指示違反ではない。"""
    _write_stem_with_bleed(stem_path, 0.9, at_end=True)

    report = analyse(stem_path, _recipe())

    interlude = next(s for s in report.stats if s.section.name == "Interlude")
    assert report.violations == []
    assert interlude.tail_bleed == pytest.approx(0.9, abs=FRAME_TOLERANCE)
    # 食い込みを外した中身は、分離漏れ程度まで下がっている。
    assert interlude.rms_dbfs < report.sung_reference_dbfs - 30


def test_held_note_at_the_start_of_an_instrumental_section_is_not_flagged(stem_path):
    """前の歌唱セクションの歌尾が伸びる場合も同じ。"""
    _write_stem_with_bleed(stem_path, 0.9, at_end=False)

    report = analyse(stem_path, _recipe())

    interlude = next(s for s in report.stats if s.section.name == "Interlude")
    assert report.violations == []
    assert interlude.head_bleed == pytest.approx(0.9, abs=FRAME_TOLERANCE)


def test_bleed_longer_than_the_limit_is_still_a_violation(stem_path):
    """上限を超える長さは食い込みではなく、区間の中身として違反に上げる。"""
    _write_stem_with_bleed(stem_path, BOUNDARY_BLEED_SECONDS + 0.5, at_end=True)

    report = analyse(stem_path, _recipe())

    interlude = next(s for s in report.stats if s.section.name == "Interlude")
    assert [s.section.name for s in report.violations] == ["Interlude"]
    assert interlude.bleed == 0.0  # 途中まで外して数字を濁らせない


def test_voice_in_the_middle_is_still_a_violation(stem_path):
    """境界に接していない声は、短くても混入として扱う。"""
    _write_stem(stem_path, {"Intro": 0.001, "Verse": 0.1, "Interlude": 0.001, "Chorus": 0.1})
    data, _ = sf.read(stem_path, dtype="float32")
    data[int(13.5 * SR) : int(14.4 * SR)] *= 100.0  # Interlude の中ほどだけ持ち上げる
    sf.write(stem_path, data, SR)

    report = analyse(stem_path, _recipe())

    assert [s.section.name for s in report.violations] == ["Interlude"]


def test_short_audio_is_clamped_and_reported(stem_path):
    """音源がレシピの宣言尺より短いのは普通に起きる(Lyria が Outro を畳む)。"""
    _write_stem(
        stem_path,
        {"Intro": 0.001, "Verse": 0.1, "Interlude": 0.001, "Chorus": 0.1},
        seconds=20.0,
    )

    report = analyse(stem_path, _recipe())

    chorus = next(s for s in report.stats if s.section.name == "Chorus")
    assert chorus.truncated
    assert chorus.clamped_end == pytest.approx(20.0, abs=0.01)
    assert report.missing_tail == pytest.approx(4.0, abs=0.01)
