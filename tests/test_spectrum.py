"""Regression tests for band measurement: band splitting and short-time detection on synthesised signals."""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from probe.assessment import load as load_assessment
from probe.recipe import Recipe, Section
from probe.spectrum import _short_term_dbfs, analyse, gain_reduction

# the fixed threshold used before 2026-08-15; tests pass the threshold explicitly
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
    """Check with single sines that the split is at 6k/10k, as in acrossover.

    4-6kHz left the compressed band on 2026-09-05. The split exists only for display and does not appear in
    acrossover, but this pins that it does not leak into the "over" column.
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

    # a sine of amplitude 0.01 has an RMS around -43dBFS
    assert levels.mean() == pytest.approx(-43.0, abs=1.0)


def test_over_ratio_is_zero_when_the_band_is_quiet(tmp_path):
    """Material below the compression threshold must not be reported as compressed."""
    levels = analyse(_write(tmp_path / "t.wav", 8000.0, 0.005), _recipe(), THRESHOLD_DBFS)[0]

    assert levels.treble_peak < THRESHOLD_DBFS
    assert levels.over_ratio == 0.0


def test_over_ratio_is_high_when_the_band_is_loud(tmp_path):
    levels = analyse(_write(tmp_path / "t.wav", 8000.0, 0.5), _recipe(), THRESHOLD_DBFS)[0]

    assert levels.treble_peak > THRESHOLD_DBFS
    assert levels.over_ratio == pytest.approx(1.0, abs=0.05)


def test_over_ratio_is_left_out_without_a_recorded_threshold(tmp_path):
    """Without a recorded threshold, no value counted against a guessed line may be shown."""
    levels = analyse(_write(tmp_path / "t.wav", 8000.0, 0.5), _recipe())[0]

    assert levels.over_ratio is None


def test_recorded_threshold_is_moved_onto_the_master(tmp_path):
    """The threshold acts before loudness matching, so on master.wav it moves by the gain."""
    path = tmp_path / "a.json"
    path.write_text(json.dumps({"assessment": {"mastering": {
        "treble_dbfs": -36.9, "treble_threshold_dbfs": -34.4, "loudness_gain_db": 0.29,
    }}}), encoding="utf-8")

    mastering = load_assessment(path).mastering

    assert mastering.master_threshold_dbfs == pytest.approx(-34.11)


def test_records_without_mastering_settings_still_load(tmp_path):
    """Records before 2026-10-29 have no mastering."""
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
    """A 60Hz kick on every beat. With `bed`, an 80Hz tone of that amplitude keeps playing and fills the low end."""
    t = np.arange(int(seconds * SR)) / SR
    since_beat = t % (60.0 / bpm)
    kick = 0.5 * np.exp(-since_beat / 0.04) * np.sin(2 * np.pi * 60.0 * t)
    return (kick + bed * np.sin(2 * np.pi * 80.0 * t)).astype(np.float32)


def test_low_punch_separates_a_standing_kick_from_a_filled_low_end(tmp_path):
    """Kicks alone widen the gap at the hits; a sustained bass filling the low end narrows it."""
    sf.write(tmp_path / "kick.wav", _kick_train(), SR)
    sf.write(tmp_path / "filled.wav", _kick_train(bed=0.3), SR)

    kick = analyse(tmp_path / "kick.wav", _recipe())[0].low_punch
    filled = analyse(tmp_path / "filled.wav", _recipe())[0].low_punch

    assert kick > 10.0
    assert filled < kick - 5.0


def test_low_punch_is_flat_for_a_steady_tone(tmp_path):
    levels = analyse(_write(tmp_path / "t.wav", 80.0, 0.5), _recipe())[0]

    assert levels.low_punch == pytest.approx(0.0, abs=0.5)


def test_gain_reduction_undoes_the_ratio():
    """A frame 2dB over after processing was 8dB over before; the reduction is 6dB."""
    short_term = np.array([-40.0, -30.0, -28.0, -30.0])

    assert gain_reduction(short_term, -30.0) == pytest.approx(6.0 / 4)


def test_gain_reduction_separates_a_spike_from_a_sustained_overshoot():
    """Same time over the threshold, but brief peaks and sustained excess differ in reduction.

    The 2026-10-28 verses looked like this."""
    spike = np.array([-40.0] * 9 + [-29.0])
    sustained = np.array([-40.0] * 9 + [-25.0])

    assert np.mean(spike > -30.0) == np.mean(sustained > -30.0)
    assert gain_reduction(spike, -30.0) < gain_reduction(sustained, -30.0) / 4
