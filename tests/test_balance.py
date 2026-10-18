"""声と伴奏の音量差（probe.balance）の回帰テスト。

demucs は通さず、合成した 2 本の stem を直接渡して計算部分だけを確かめます。
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from probe.balance import analyse_balance
from probe.recipe import Recipe, Section

SR = 22050
INSTRUMENTAL_PROMPT = "[Instrumental Interlude] Instruments only — no lead vocal."
SUNG_PROMPT = "[Verse] Focus on the lyrics marked [Verse]."


def _recipe() -> Recipe:
    return Recipe(
        title="テスト",
        tempo=120,
        sections=[
            Section("Intro", 0.0, 4.0, INSTRUMENTAL_PROMPT),
            Section("Verse", 4.0, 12.0, SUNG_PROMPT),
            Section("Chorus", 12.0, 20.0, SUNG_PROMPT),
        ],
    )


def _write(path, levels: dict[str, float], freq: float, seconds: float = 20.0, gaps: bool = False):
    t = np.arange(int(seconds * SR)) / SR
    tone = np.sin(2 * np.pi * freq * t).astype(np.float32)
    env = np.zeros_like(tone)
    for s in _recipe().sections:
        lo, hi = int(s.start * SR), int(min(s.end, seconds) * SR)
        env[lo:hi] = levels[s.name]
    if gaps:
        # 1 秒ごとに 0.5 秒黙る。行間の無音を模したもの。
        env[(t % 1.0) >= 0.5] = 0.0
    sf.write(path, tone * env, SR)
    return path


@pytest.fixture
def stems(tmp_path):
    return tmp_path / "vocals.wav", tmp_path / "no_vocals.wav"


def _by_name(balances):
    return {b.section.name: b.vocal_over_backing_db for b in balances}


def test_reports_backing_louder_in_the_chorus(stems):
    vocals, backing = stems
    _write(vocals, {"Intro": 0.0, "Verse": 0.1, "Chorus": 0.1}, 440.0)
    # Chorus の伴奏を声の 2 倍（+6dB）にする。Verse は同じ大きさ。
    _write(backing, {"Intro": 0.1, "Verse": 0.1, "Chorus": 0.2}, 110.0)

    got = _by_name(analyse_balance(vocals, backing, _recipe()))

    assert "Intro" not in got, "器楽区間は測らない"
    assert got["Verse"] == pytest.approx(0.0, abs=0.2)
    assert got["Chorus"] == pytest.approx(-6.0, abs=0.2)


def test_ignores_frames_where_nobody_sings(stems):
    """行間の無音を含めると、声の大きさと無関係に差がマイナスへ引っ張られる。"""
    vocals, backing = stems
    _write(vocals, {"Intro": 0.0, "Verse": 0.1, "Chorus": 0.1}, 440.0, gaps=True)
    _write(backing, {"Intro": 0.1, "Verse": 0.1, "Chorus": 0.1}, 110.0)

    got = _by_name(analyse_balance(vocals, backing, _recipe()))

    assert got["Verse"] == pytest.approx(0.0, abs=0.5)


def test_returns_none_when_the_section_is_never_sung(stems):
    vocals, backing = stems
    _write(vocals, {"Intro": 0.0, "Verse": 0.1, "Chorus": 0.0}, 440.0)
    _write(backing, {"Intro": 0.1, "Verse": 0.1, "Chorus": 0.1}, 110.0)

    got = _by_name(analyse_balance(vocals, backing, _recipe()))

    assert got["Chorus"] is None
