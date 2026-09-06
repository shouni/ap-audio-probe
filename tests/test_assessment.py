"""assessment.json との突き合わせの回帰テスト。

要点は「生成側と同じ測り方をしているか」の 1 点です。ここがずれると、mp3 と wav の差を
見るための計測が、測り方の差を見る計測になります。
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
    """生成側 AudioAssessmentRecord と同じ形の JSON。"""
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
    """生成側は全チャンネルを畳んでから振幅ピークを取る（inspector.go: meanSample）。

    左右を逆相にすると平均は 0 になります。チャンネルごとに測っていればピークは 0.9 で、
    畳んでいれば 0 です。`peak_amplitude` が wav とも mp3 とも一致しなかった理由が
    これなので、同じ畳み方をしていることをここで固定します。
    """
    tone = _sine(1000.0, 0.9)

    measured = measure(_write(tmp_path / "t.wav", tone, -tone))

    # 完全な 0 にはなりません。wav へ書いた時点の量子化が残ります(1.5e-5 ≒ 1/65536)。
    assert measured.peak_amplitude == pytest.approx(0.0, abs=1e-3)


def test_peak_and_rms_match_a_known_signal(tmp_path):
    measured = measure(_write(tmp_path / "t.wav", _sine(1000.0, 0.5)))

    assert measured.peak_amplitude == pytest.approx(0.5, abs=0.01)
    # 振幅 0.5 の正弦波の RMS は 0.354 = -9.0dBFS。
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
    """主帯域だけが鳴っていれば、記録される headroom は大きく出る。

    1kHz の純音でも 17dB 程度にしかなりません。2 次バンドパス 1 段は裾が緩く、
    4-10kHz 側にも 1kHz が残るためです(inspector.go の但し書きどおり)。この値を
    帯域の絶対量として読んではいけない、という確認も兼ねています。
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
    """講評と譜面はどちらも欠けうる（生成側は失敗しても記録を書きます）。"""
    data = _record()
    del data["assessment"]["review"]
    del data["assessment"]["lyrics"]
    path = tmp_path / "a.json"
    path.write_text(json.dumps(data), encoding="utf-8")

    record = load(path)

    assert record.review is None and record.lyrics is None
    # 欠けている項目は行ごと出しません（0/5 と書くと講評が付いたように読めます）。
    assert "講評" not in render(record, record.measurements, "wav")


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
    """生成側の尺は mp3 の padding のぶん長い。それを毎回「差」と呼んではいけない。"""
    padding = MP3_PADDING_SAMPLES / SR
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_record(duration_seconds=180.0 + padding)), encoding="utf-8")

    duration = next(c for c in compare(load(path), _measured(duration_seconds=180.0)) if c.label == "尺")

    assert duration.note
    assert not duration.unexplained


def test_a_level_gap_beyond_the_encoding_difference_is_flagged(tmp_path):
    """mp3 と wav の差は 0.1dB 台。それを超えるラウドネス差は説明が付かない。"""
    path = tmp_path / "a.json"
    path.write_text(json.dumps(_record()), encoding="utf-8")
    record = load(path)

    flagged = [c for c in compare(record, _measured(integrated_lufs=-11.0)) if c.unexplained]

    assert [c.label for c in flagged] == ["integrated"]
    assert "説明の付かない差" in render(record, _measured(integrated_lufs=-11.0), "wav")


def test_a_note_does_not_silence_a_large_gap(tmp_path):
    """但し書きが付く項目でも、差が大きければ警告は出す。

    peak_amplitude には「チャンネル平均で測った値」という注記が常に付きます。注記を
    警告の抑止に使うと、測り方の説明が差そのものの免罪符になります。
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
