"""Regression tests for cross-song summaries, with synthesised audio and recipes and no separation."""

from __future__ import annotations

import json

import numpy as np
import soundfile as sf

from probe.corpus import TrackSummary, render, summarise

SR = 44100


def _track(name: str, mode: str, punch: float | None, vob: float | None = None) -> TrackSummary:
    return TrackSummary(
        name=name, mode=mode, integrated_lufs=-12.4, true_peak_dbtp=-2.1,
        chorus_low_punch=punch, chorus_treble=-34.0, chorus_gain_reduction=None, verse_gain_reduction=None,
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
    edm = next(i for i, line in enumerate(lines) if "edm_pop median (3 songs)" in line)
    rock = next(i for i, line in enumerate(lines) if "sparkle_rock median (1 song)" in line)
    # sorted by mode name; each median row comes right after that mode's songs
    assert edm < rock
    assert [line.split()[0] for line in lines[edm - 3 : edm]] == ["a", "c", "d"]
    assert "8.0" in lines[edm]


def test_missing_values_are_shown_as_a_dash_and_left_out_of_the_median():
    out = render([_track("a", "edm_pop", 8.0, vob=None), _track("b", "edm_pop", None, vob=-3.0)])

    median = next(line for line in out.splitlines() if "median" in line)
    assert median.split()[-4:] == ["-34.0", "-", "-", "-3.0"]
    assert "8.0" in median


def test_summarise_averages_only_the_chorus_sections(tmp_path):
    """Chorus values come only from sections named Chorus; the Verse low end is not mixed in."""
    t = np.arange(4 * SR) / SR
    since_beat = t % 0.5
    kick = 0.5 * np.exp(-since_beat / 0.04) * np.sin(2 * np.pi * 60.0 * t)
    steady = 0.3 * np.sin(2 * np.pi * 80.0 * t)
    # first 4 s: a Verse with a sustained tone filling the low end; last 4 s: a Chorus with only kicks
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
    # no assessment.json, so no gain reduction is computed
    assert summary.chorus_gain_reduction is None
