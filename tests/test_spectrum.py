"""帯域計測の回帰テスト。合成した信号で、帯域分離と短時間検出を確かめます。"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from probe.assessment import load as load_assessment
from probe.recipe import Recipe, Section
from probe.spectrum import _short_term_dbfs, analyse

# 2026-08-15 より前の固定閾値。テストでは閾値を明示して渡します。
THRESHOLD_DBFS = -32.0

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


@pytest.mark.parametrize(
    "freq,loudest",
    [(500.0, "low"), (5000.0, "presence"), (8000.0, "treble"), (15000.0, "air")],
)
def test_energy_lands_in_the_expected_band(tmp_path, freq, loudest):
    """acrossover と同じ 6k/10k で切れていることを、単一の正弦波で確かめる。

    4-6kHz は 2026-09-05 に圧縮対象から外れた帯域です。表示だけの分割なので
    acrossover には現れませんが、圧縮域の列へ混ざっていないことはここで押さえます。
    """
    levels = analyse(_write(tmp_path / "t.wav", freq, 0.5), _recipe())[0]

    bands = {
        "low": levels.low,
        "presence": levels.presence,
        "treble": levels.treble,
        "air": levels.air,
    }
    assert max(bands, key=bands.get) == loudest


def test_short_term_level_tracks_the_signal():
    t = np.arange(SR) / SR
    quiet = 0.01 * np.sin(2 * np.pi * 8000.0 * t)

    levels = _short_term_dbfs(quiet.astype(np.float32), SR)

    # 振幅 0.01 の正弦波の RMS は -43dBFS 前後。
    assert levels.mean() == pytest.approx(-43.0, abs=1.0)


def test_over_ratio_is_zero_when_the_band_is_quiet(tmp_path):
    """圧縮の閾値に届かない素材で「圧縮がかかる」と出てはいけない。"""
    levels = analyse(_write(tmp_path / "t.wav", 8000.0, 0.005), _recipe(), THRESHOLD_DBFS)[0]

    assert levels.treble_peak < THRESHOLD_DBFS
    assert levels.over_ratio == 0.0


def test_over_ratio_is_high_when_the_band_is_loud(tmp_path):
    levels = analyse(_write(tmp_path / "t.wav", 8000.0, 0.5), _recipe(), THRESHOLD_DBFS)[0]

    assert levels.treble_peak > THRESHOLD_DBFS
    assert levels.over_ratio == pytest.approx(1.0, abs=0.05)


def test_over_ratio_is_left_out_without_a_recorded_threshold(tmp_path):
    """閾値の記録が無い曲で、推測の線で数えた値を出してはいけない。"""
    levels = analyse(_write(tmp_path / "t.wav", 8000.0, 0.5), _recipe())[0]

    assert levels.over_ratio is None


def test_recorded_threshold_is_moved_onto_the_master(tmp_path):
    """閾値は音量合わせより前で効くので、master.wav の上ではゲインの分だけ動く。"""
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"assessment": {"mastering": {
        "treble_dbfs": -36.9, "treble_threshold_dbfs": -34.4, "loudness_gain_db": 0.29,
    }}}), encoding="utf-8")

    mastering = load_assessment(path).mastering

    assert mastering.master_threshold_dbfs == pytest.approx(-34.11)


def test_records_without_mastering_settings_still_load(tmp_path):
    """2026-10-29 より前の記録には mastering が無い。"""
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"assessment": {"quality": {"measurements": {}}}}), encoding="utf-8")

    assert load_assessment(path).mastering is None


def test_sections_beyond_the_audio_are_dropped(tmp_path):
    recipe = Recipe(
        title="テスト",
        tempo=126,
        sections=[
            Section("Chorus", 0.0, 2.0, SUNG_PROMPT),
            Section("Outro", 10.0, 20.0, "[Instrumental Outro] Instruments only."),
        ],
    )

    levels = analyse(_write(tmp_path / "t.wav", 8000.0, 0.2, seconds=2.0), recipe)

    assert [lv.section.name for lv in levels] == ["Chorus"]


def _kick_train(seconds: float = 4.0, bpm: float = 120.0, bed: float = 0.0) -> np.ndarray:
    """60Hz のキックを拍ごとに置いた信号。bed を足すと、その振幅の 80Hz が鳴り続けて低域を埋める。"""
    t = np.arange(int(seconds * SR)) / SR
    since_beat = t % (60.0 / bpm)
    kick = 0.5 * np.exp(-since_beat / 0.04) * np.sin(2 * np.pi * 60.0 * t)
    return (kick + bed * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32)


def test_low_punch_separates_a_standing_kick_from_a_filled_low_end(tmp_path):
    """キックだけなら打点で差が開き、持続するベースが低域を埋めると差が詰まる。"""
    sf.write(tmp_path / "kick.wav", _kick_train(), SR)
    sf.write(tmp_path / "filled.wav", _kick_train(bed=0.3), SR)

    kick = analyse(tmp_path / "kick.wav", _recipe())[0].low_punch
    filled = analyse(tmp_path / "filled.wav", _recipe())[0].low_punch

    assert kick > 10.0
    assert filled < kick - 5.0


def test_low_punch_is_flat_for_a_steady_tone(tmp_path):
    levels = analyse(_write(tmp_path / "t.wav", 80.0, 0.5), _recipe())[0]

    assert levels.low_punch == pytest.approx(0.0, abs=0.5)
