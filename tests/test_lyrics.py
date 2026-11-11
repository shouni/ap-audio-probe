"""Regression tests for lyric matching. No ASR; only the comparison is checked, quickly."""

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

# An actual transcript of Lyria audio. The English hook was misheard every time and 「境界」 became 「妖怪」;
# both lines were in fact sung.
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
    """Japanese ASR cannot reliably hear mixed-in English, so it is left out of the comparison."""
    assert to_kana("Split the role 届かない場所") == to_kana("届かない場所")


def test_english_only_line_is_not_verifiable():
    line = LineCheck("Split the role", 0.0)

    assert not line.verifiable
    assert line.sung  # a line that cannot be checked must not be reported as dropped


@pytest.mark.parametrize("line", [
    "削ぎ落としたノイズを捨てろ",
    "闇を裂き解き放て",
    "境界をここに引け",  # sung, even though it was heard as 「妖怪よここに行け」
])
def test_sung_lines_clear_the_threshold(line):
    assert _cov(line) >= COVERAGE_THRESHOLD


@pytest.mark.parametrize("line", [
    "夏の終わりに手を振った",
    "遠い街の冬を数えて",
    "アスファルトの向こうへ走れ",
])
def test_absent_lines_stay_below_the_threshold(line):
    """If unsung lines cross the threshold, dropped lines are missed."""
    assert _cov(line) < COVERAGE_THRESHOLD


def test_threshold_sits_between_the_two_groups():
    """Pin, with margin, that the positive and negative cases sit on either side of the threshold."""
    sung = min(_cov(line) for line in ["削ぎ落としたノイズを捨てろ", "境界をここに引け", "闇を裂き解き放て"])
    absent = max(_cov(line) for line in ["夏の終わりに手を振った", "遠い街の冬を数えて"])

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
    assert [check.line for check in result[0].missing] == ["夏の終わりに手を振った"]


SR = 16000
# singing at -20dBFS, separation leakage at -50dBFS, matching measurements (-18dB / -50dB)
LOUD, QUIET = 0.1, 0.003
FLOOR = -40.0


def _clip(loud_from: float, loud_to: float, seconds: float = 12.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    envelope = np.full_like(tone, QUIET)
    envelope[int(loud_from * SR) : int(loud_to * SR)] = LOUD
    return tone * envelope


def test_a_voiceless_clip_yields_nothing_to_transcribe():
    """A section without voice returns nothing; that is trim_to_voice's main purpose.

    Given it, whisper hallucinates (an outro of only separation leakage became 「ありがとうございました。」),
    and a section that lost all its lines would be missed.
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
    """Without padding, consonant onsets would be cut."""
    clip = _clip(2.0, 10.0, seconds=12.0)
    trimmed = trim_to_voice(clip, FLOOR, SR)

    assert np.max(np.abs(trimmed[: int(CLIP_PAD_SECONDS * SR)])) < LOUD


def test_padding_reaches_outside_the_section():
    """Even when the voice starts right at the section head, the padding is taken from outside the section.

    With zero padding, whisper flattens the whole section into a hallucination (seen on a 120 BPM Verse).
    """
    audio = _clip(2.0, 12.0, seconds=12.0)
    start = 2 * SR
    trimmed = trim_to_voice(audio, FLOOR, SR, start=start, end=len(audio))

    assert len(trimmed) / SR == pytest.approx(10.0 + CLIP_PAD_SECONDS, abs=0.06)
    assert np.max(np.abs(trimmed[: int(CLIP_PAD_SECONDS * SR) - 1])) < LOUD


def test_a_neighbours_voice_does_not_make_a_silent_section_voiced():
    """Voice is judged inside the section only.

    If a neighbour's held ending counted as voice, a fully dropped section would be missed."""
    audio = _clip(0.0, 2.0, seconds=12.0)  # 0-2 s is the neighbour's singing; from 2 s is the section under test

    assert trim_to_voice(audio, FLOOR, SR, start=2 * SR, end=len(audio)).size == 0


def test_a_fully_voiced_clip_is_left_alone():
    clip = _clip(0.0, 12.0, seconds=12.0)

    assert len(trim_to_voice(clip, FLOOR, SR)) == len(clip)


def test_instrumental_sections_without_lyrics_are_skipped():
    intro = Section("Intro", 0.0, 8.0, "[Instrumental Intro] Instruments only.")
    recipe = Recipe(title="テスト", tempo=126, sections=[intro], lyrics={})

    assert check([(intro, "")], recipe) == []
