"""Regression tests for comparing with assessment.json.

The one point is "does it measure the same way as the generator". If that drifts, a measurement meant to show
mp3 vs wav differences ends up showing differences in method.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from probe.assessment import (
    MP3_PADDING_SAMPLES,
    Measurements,
    compare,
    load,
    measure,
    render,
)

SR = 44100


def _write(path, left: np.ndarray, right: np.ndarray | None = None):
    data = left if right is None else np.stack([left, right], axis=1)
    sf.write(path, data.astype(np.float32), SR)
    return path


def _sine(freq: float, amplitude: float, seconds: float = 5.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return amplitude * np.sin(2 * np.pi * freq * t)


def _record(**measurements) -> dict:
    """JSON shaped like the generator's AudioAssessmentRecord."""
    base = {
        "duration_seconds": 180.0,
        "leading_silence_seconds": 0.1,
        "trailing_silence_seconds": 2.0,
        "longest_internal_silence_seconds": 0.5,
        "silent_ratio": 0.05,
        "peak_amplitude": 0.867,
        "clipped_sample_ratio": 0.0,
        "rms_dbfs": -14.0,
        "mid_band_dbfs": -20.0,
        "high_band_dbfs": -31.0,
        "true_peak_dbtp": -1.5,
        "integrated_lufs": -12.5,
    }
    base.update(measurements)
    return {
        "job_id": "music-20260904-171931-5364ab06bcef",
        "created_at": "2026-09-04T17:19:31Z",
        "score": 0.92,
        "summary": "OK (180.0s, measured 0.92)",
        "passed": True,
        "treble_headroom_db": base["mid_band_dbfs"] - base["high_band_dbfs"],
        "assessment": {
            "quality": {"measurements": base, "issues": [], "score": 0.92},
            "mastered": True,
            "review": {
                "lyrics_intelligibility": 4,
                "structure_match": 5,
                "pacing": 4,
                "vocal_balance": 5,
                "findings": ["Chorus 2 の複数行が潰れている"],
            },
            "lyrics": {"sung": "[[A1]]\nアイウ", "missing_lines": 1, "extra_lines": 0},
        },
    }


def test_peak_follows_the_channel_average(tmp_path):
    """The generator folds all channels before taking the amplitude peak (inspector.go: meanSample).

    With left and right in opposite phase the average is 0: measured per channel the peak is 0.9, folded it is 0.
    This is why `peak_amplitude` matched neither the wav nor the mp3, so the folding is pinned here.
    """
    tone = _sine(1000.0, 0.9)

    measured = measure(_write(tmp_path / "t.wav", tone, -tone))

    # Not exactly 0: quantisation from writing the wav remains (1.5e-5 ≈ 1/65536).
    assert measured.peak_amplitude == pytest.approx(0.0, abs=1e-3)


def test_peak_and_rms_match_a_known_signal(tmp_path):
    measured = measure(_write(tmp_path / "t.wav", _sine(1000.0, 0.5)))

    assert measured.peak_amplitude == pytest.approx(0.5, abs=0.01)
    # a sine of amplitude 0.5 has an RMS of 0.354 = -9.0dBFS
    assert measured.rms_dbfs == pytest.approx(-9.0, abs=0.2)
    assert measured.duration_seconds == pytest.approx(5.0, abs=0.01)


def test_leading_and_trailing_silence_are_measured_from_the_edges(tmp_path):
    quiet = np.zeros(SR)
    clip = np.concatenate([quiet, _sine(1000.0, 0.5, seconds=3.0), quiet, quiet])

    measured = measure(_write(tmp_path / "t.wav", clip))

    assert measured.leading_silence_seconds == pytest.approx(1.0, abs=0.1)
    assert measured.trailing_silence_seconds == pytest.approx(2.0, abs=0.1)
    assert measured.longest_internal_silence_seconds == 0.0


def test_treble_headroom_is_positive_for_a_dark_signal(tmp_path):
    """With only the main band playing, the recorded headroom comes out large.

    Even a pure 1kHz tone only reaches about 17dB: a single second-order band-pass has gentle skirts and some
    1kHz remains in the 4-10kHz band (as inspector.go notes). This also confirms the value must not be read as
    an absolute band level.
    """
    measured = measure(_write(tmp_path / "t.wav", _sine(1000.0, 0.5)))

    assert measured.mid_band_dbfs > measured.high_band_dbfs
    assert measured.treble_headroom_db > 15


def test_load_reads_the_generated_record(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")

    record = load(path)

    assert record.job_id == "music-20260904-171931-5364ab06bcef"
    assert record.passed is True
    assert record.mastered is True
    assert record.measurements.integrated_lufs == -12.5
    assert record.measurements.treble_headroom_db == pytest.approx(11.0)
    assert record.review["vocal_balance"] == 5
    assert record.lyrics["missing_lines"] == 1


def test_load_survives_a_record_without_review_or_lyrics(tmp_path):
    """The review and the score can both be missing (the generator writes the record even when they fail)."""
    data = _record()
    del data["assessment"]["review"]
    del data["assessment"]["lyrics"]
    path = tmp_path / "a.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    record = load(path)

    assert record.review is None and record.lyrics is None
    # Missing items are left out entirely (writing 0/5 would read as if there were a review).
    assert "review" not in render(record, record.measurements, "wav")


def _measured(**overrides) -> Measurements:
    values = {
        "duration_seconds": 180.0,
        "leading_silence_seconds": 0.1,
        "trailing_silence_seconds": 2.0,
        "longest_internal_silence_seconds": 0.5,
        "silent_ratio": 0.05,
        "peak_amplitude": 0.867,
        "clipped_sample_ratio": 0.0,
        "rms_dbfs": -14.0,
        "mid_band_dbfs": -20.0,
        "high_band_dbfs": -31.0,
        "true_peak_dbtp": -1.5,
        "integrated_lufs": -12.5,
    }
    values.update(overrides)
    return Measurements(**values)


def test_the_mp3_padding_is_not_reported_as_a_difference(tmp_path):
    """The generator's length is longer by the mp3 padding; that must not be called a difference every time."""
    padding = MP3_PADDING_SAMPLES / SR
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_record(duration_seconds=180.0 + padding)), encoding="utf-8")

    duration = next(c for c in compare(load(path), _measured(duration_seconds=180.0)) if c.label == "length")

    assert duration.note
    assert not duration.unexplained


def test_a_level_gap_beyond_the_encoding_difference_is_flagged(tmp_path):
    """mp3 vs wav differ around 0.1dB; a loudness gap beyond that has no explanation."""
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")
    record = load(path)

    flagged = [c for c in compare(record, _measured(integrated_lufs=-11.0)) if c.unexplained]

    assert [c.label for c in flagged] == ["integrated"]
    assert "unexplained difference" in render(record, _measured(integrated_lufs=-11.0), "wav")


def test_a_note_does_not_silence_a_large_gap(tmp_path):
    """Items with a remark still warn when the difference is large.

    peak_amplitude always carries the "measured on the channel average" remark. Using remarks to silence warnings
    would let the description of the method excuse the difference itself.
    """
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")

    peak = next(c for c in compare(load(path), _measured(peak_amplitude=0.5)) if c.label == "peak_amplitude")

    assert peak.note
    assert peak.unexplained


def test_matching_measurements_report_no_gap(tmp_path):
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")
    record = load(path)

    assert not [c for c in compare(record, _measured()) if c.unexplained]
    assert "✓" in render(record, _measured(), "wav")
