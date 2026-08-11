"""帯域計測の回帰テスト。合成した信号で、帯域分離と短時間検出を確かめます。"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from probe.recipe import Recipe, Section
from probe.spectrum import COMPRESSOR_THRESHOLD_DBFS, _short_term_dbfs, analyse

SR = 44100
SUNG_PROMPT = "[Chorus] Focus on the lyrics marked [Chorus]."


def _write(path, freq_hz: float, amplitude: float, seconds: float = 4.0):
    t = np.arange(int(seconds * SR)) / SR
    sf.write(path, (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32), SR)
    return path


def _recipe(seconds: float = 4.0) -> Recipe:
    return Recipe(
        title="テスト",
        tempo=126,
        sections=[Section("Chorus", 0.0, seconds, SUNG_PROMPT)],
    )


@pytest.mark.parametrize("freq,loudest", [(500.0, "low"), (6000.0, "mid"), (15000.0, "high")])
def test_energy_lands_in_the_expected_band(tmp_path, freq, loudest):
    """acrossover と同じ 4k/10k で切れていることを、単一の正弦波で確かめる。"""
    levels = analyse(_write(tmp_path / "t.wav", freq, 0.5), _recipe())[0]

    bands = {"low": levels.low, "mid": levels.mid, "high": levels.high}
    assert max(bands, key=bands.get) == loudest


def test_short_term_level_tracks_the_signal():
    t = np.arange(SR) / SR
    quiet = 0.01 * np.sin(2 * np.pi * 6000.0 * t)

    levels = _short_term_dbfs(quiet.astype(np.float32), SR)

    # 振幅 0.01 の正弦波の RMS は -43dBFS 前後。
    assert levels.mean() == pytest.approx(-43.0, abs=1.0)


def test_over_ratio_is_zero_when_the_band_is_quiet(tmp_path):
    """圧縮の閾値に届かない素材で「圧縮がかかる」と出てはいけない。"""
    levels = analyse(_write(tmp_path / "t.wav", 6000.0, 0.005), _recipe())[0]

    assert levels.mid_peak < COMPRESSOR_THRESHOLD_DBFS
    assert levels.over_ratio == 0.0


def test_over_ratio_is_high_when_the_band_is_loud(tmp_path):
    levels = analyse(_write(tmp_path / "t.wav", 6000.0, 0.5), _recipe())[0]

    assert levels.mid_peak > COMPRESSOR_THRESHOLD_DBFS
    assert levels.over_ratio == pytest.approx(1.0, abs=0.05)


def test_sections_beyond_the_audio_are_dropped(tmp_path):
    recipe = Recipe(
        title="テスト",
        tempo=126,
        sections=[
            Section("Chorus", 0.0, 2.0, SUNG_PROMPT),
            Section("Outro", 10.0, 20.0, "[Instrumental Outro] Instruments only."),
        ],
    )

    levels = analyse(_write(tmp_path / "t.wav", 6000.0, 0.2, seconds=2.0), recipe)

    assert [lv.section.name for lv in levels] == ["Chorus"]
