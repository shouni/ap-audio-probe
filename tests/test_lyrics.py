"""歌詞照合の回帰テスト。ASR は通さず、突き合わせの部分だけを速く確かめます。"""

from __future__ import annotations

import numpy as np
import pytest

from probe.lyrics import (
    CLIP_PAD_SECONDS,
    COVERAGE_THRESHOLD,
    LineCheck,
    _coverage,
    check,
    to_kana,
    trim_to_voice,
)
from probe.recipe import Recipe, Section, parse_lyrics

# 実際に Lyria の音源を書き起こした結果。英語のフックは一貫して聞き違えられ、
# 「境界」は「妖怪」になりました。どちらも行そのものは歌われています。
HEARD = (
    "Speed the road 届かない場所 そぎ落としたノイズを捨てろ Speed the road 研ぎ澄ます光 "
    "妖怪よ ここに行け 闇よ 先解き放て Speed the road 打ち鳴らせ鼓動 余計なものは持ち込む"
)


def _cov(line: str) -> float:
    return _coverage(to_kana(line), to_kana(HEARD))


def test_parse_lyrics_splits_on_section_tags():
    parsed = parse_lyrics("[Verse]\n一行目\n二行目\n\n[Chorus]\nサビ\n")

    assert parsed == {"Verse": ["一行目", "二行目"], "Chorus": ["サビ"]}


def test_to_kana_drops_latin():
    """日本語 ASR は混ざった英語を安定して聞き取れないため、比較から外す。"""
    assert to_kana("Split the role 届かない場所") == to_kana("届かない場所")


def test_english_only_line_is_not_verifiable():
    line = LineCheck("Split the role", 0.0)

    assert not line.verifiable
    assert line.sung  # 確かめられない行を「落ちた」と報告してはいけない


@pytest.mark.parametrize("line", [
    "削ぎ落としたノイズを捨てろ",
    "闇を裂き解き放て",
    "境界をここに引け",  # 「妖怪よここに行け」と誤認識されても歌われている
])
def test_sung_lines_clear_the_threshold(line):
    assert _cov(line) >= COVERAGE_THRESHOLD


@pytest.mark.parametrize("line", [
    "夏の終わりに手を振った",
    "遠い街の冬を数えて",
    "アスファルトの向こうへ走れ",
])
def test_absent_lines_stay_below_the_threshold(line):
    """歌われていない行が閾値を超えると、行落ちを見逃す。"""
    assert _cov(line) < COVERAGE_THRESHOLD


def test_threshold_sits_between_the_two_groups():
    """陽性と陰性が閾値を挟んで分かれていることを、余裕込みで固定する。"""
    sung = min(_cov(l) for l in ["削ぎ落としたノイズを捨てろ", "境界をここに引け", "闇を裂き解き放て"])
    absent = max(_cov(l) for l in ["夏の終わりに手を振った", "遠い街の冬を数えて"])

    assert absent < COVERAGE_THRESHOLD < sung
    assert sung - absent > 0.3


def test_check_reports_missing_lines_per_section():
    chorus = Section("Chorus", 0.0, 30.0, "[Chorus] Focus on the lyrics.")
    recipe = Recipe(
        title="テスト",
        tempo=126,
        sections=[chorus],
        lyrics={"Chorus": ["削ぎ落としたノイズを捨てろ", "夏の終わりに手を振った"]},
    )

    result = check([(chorus, HEARD)], recipe)

    assert len(result) == 1
    assert [l.line for l in result[0].missing] == ["夏の終わりに手を振った"]


SR = 16000
# 歌唱が -20dBFS、分離漏れが -50dBFS。実測(-18dB / -50dB)に合わせた値です。
LOUD, QUIET = 0.1, 0.003
FLOOR = -40.0


def _clip(loud_from: float, loud_to: float, seconds: float = 12.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    envelope = np.full_like(tone, QUIET)
    envelope[int(loud_from * SR) : int(loud_to * SR)] = LOUD
    return tone * envelope


def test_a_voiceless_clip_yields_nothing_to_transcribe():
    """声が無い区間は空を返す。これが trim_to_voice の主目的。

    渡せば whisper が幻聴を返し(実測では分離漏れだけの Outro が
    「ありがとうございました。」になった)、行が丸ごと落ちたセクションを
    見逃すことになります。
    """
    silent = _clip(0.0, 0.0, seconds=12.0)

    assert trim_to_voice(silent, FLOOR, SR).size == 0


def test_trailing_silence_is_trimmed():
    trimmed = trim_to_voice(_clip(0.0, 10.0, seconds=12.0), FLOOR, SR)

    assert len(trimmed) / SR == pytest.approx(10.0 + CLIP_PAD_SECONDS, abs=0.06)


def test_leading_silence_is_trimmed():
    trimmed = trim_to_voice(_clip(2.0, 12.0, seconds=12.0), FLOOR, SR)

    assert len(trimmed) / SR == pytest.approx(10.0 + CLIP_PAD_SECONDS, abs=0.06)


def test_padding_keeps_the_onset():
    """余白を残さないと子音の立ち上がりを切ってしまう。"""
    clip = _clip(2.0, 10.0, seconds=12.0)
    trimmed = trim_to_voice(clip, FLOOR, SR)

    assert np.max(np.abs(trimmed[: int(CLIP_PAD_SECONDS * SR)])) < LOUD


def test_a_fully_voiced_clip_is_left_alone():
    clip = _clip(0.0, 12.0, seconds=12.0)

    assert len(trim_to_voice(clip, FLOOR, SR)) == len(clip)


def test_instrumental_sections_without_lyrics_are_skipped():
    intro = Section("Intro", 0.0, 8.0, "[Instrumental Intro] Instruments only.")
    recipe = Recipe(title="テスト", tempo=126, sections=[intro], lyrics={})

    assert check([(intro, "")], recipe) == []
