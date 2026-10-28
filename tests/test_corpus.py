"""曲を跨いだ集計の回帰テスト。分離は走らせず、合成した音源とレシピで確かめます。"""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from probe.corpus import TrackSummary, render, summarise

SR = 44100


def _track(name: str, mode: str, punch: float | None, vob: float | None = None) -> TrackSummary:
    return TrackSummary(
        name=name, mode=mode, integrated_lufs=-12.4, true_peak_dbtp=-2.1,
        chorus_low_punch=punch, chorus_treble=-34.0, chorus_over_ratio=None, verse_over_ratio=None,
        chorus_vocal_over_backing=vob,
    )


def test_tracks_are_grouped_by_mode_with_a_median_row():
    out = render([
        _track("a", "edm_pop", 8.0),
        _track("b", "sparkle_rock", 4.0),
        _track("c", "edm_pop", 6.0),
        _track("d", "edm_pop", 9.0),
    ])

    lines = out.splitlines()
    edm = next(i for i, line in enumerate(lines) if "edm_pop の中央値 (3 曲)" in line)
    rock = next(i for i, line in enumerate(lines) if "sparkle_rock の中央値 (1 曲)" in line)
    # モード名の順に並び、中央値の行はそのモードの曲の直後に来る。
    assert edm < rock
    assert [line.split()[0] for line in lines[edm - 3 : edm]] == ["a", "c", "d"]
    assert "8.0" in lines[edm]


def test_missing_values_are_shown_as_a_dash_and_left_out_of_the_median():
    out = render([_track("a", "edm_pop", 8.0, vob=None), _track("b", "edm_pop", None, vob=-3.0)])

    median = next(line for line in out.splitlines() if "中央値" in line)
    assert median.split()[-4:] == ["-34.0", "-", "-", "-3.0"]
    assert "8.0" in median


def test_summarise_averages_only_the_chorus_sections(tmp_path):
    """サビの値は Chorus を名前に含むセクションだけから出す。Verse の低域は混ぜない。"""
    t = np.arange(4 * SR) / SR
    since_beat = t % 0.5
    kick = 0.5 * np.exp(-since_beat / 0.04) * np.sin(2 * np.pi * 60.0 * t)
    steady = 0.3 * np.sin(2 * np.pi * 80.0 * t)
    # 前半 4 秒は低域を持続音が埋める Verse、後半 4 秒はキックだけの Chorus。
    sf.write(tmp_path / "song.wav", np.concatenate([steady, kick]).astype(np.float32), SR)
    (tmp_path / "song.json").write_text(json.dumps({
        "title": "テスト", "tempo": 120, "compose_mode": "sparkle_rock",
        "sections": [
            {"name": "Verse", "start_seconds": 0, "end_seconds": 4, "prompt": "[Verse]"},
            {"name": "Chorus", "start_seconds": 4, "end_seconds": 8, "prompt": "[Chorus]"},
        ],
    }), encoding="utf-8")

    summary = summarise(tmp_path / "song.wav", tmp_path / "song.json",
                        balance=False, out_dir=tmp_path, device=None, assessments=tmp_path)

    assert summary.mode == "sparkle_rock"
    assert summary.chorus_low_punch > 10.0
    assert summary.chorus_vocal_over_backing is None
    # assessment.json が無いので、圧縮域は数えない。
    assert summary.chorus_over_ratio is None
