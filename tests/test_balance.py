"""Regression tests for the voice vs backing level (probe.balance).

No demucs: two synthesised stems are passed directly to check only the calculation.
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
        # silent for 0.5 s every second, imitating the gaps between lines
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
    # the Chorus backing is twice the voice (+6dB); the Verse is equal
    _write(backing, {"Intro": 0.1, "Verse": 0.1, "Chorus": 0.2}, 110.0)

    got = _by_name(analyse_balance(vocals, backing, _recipe()))

    assert "Intro" not in got, "instrumental sections are not measured"
    assert got["Verse"] == pytest.approx(0.0, abs=0.2)
    assert got["Chorus"] == pytest.approx(-6.0, abs=0.2)


def test_ignores_frames_where_nobody_sings(stems):
    """Including the silent gaps between lines would pull the difference negative regardless of the voice."""
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
