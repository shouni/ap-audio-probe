"""Measure, per sung section, how far the voice sits in front of the backing.

    python -m probe.balance audio/foo.wav recipes/foo.json

A measurement to check, with numbers, the review model's remark that "synths cover the voice in the chorus".
It compares the two demucs stems (vocals.wav and no_vocals.wav) at the same times and reports the median
"voice − backing" difference in dB over the frames where the voice sounds.

On four songs from 2026-10-17, choruses were -4 to -7dB (backing louder) and verses around -1dB, matching
the passages the review called buried. Songs after the generator's edm_pop chorus instructions were revised
measured -3.2 / -3.6dB. Use the value to compare songs, and before / after a prompt change. There is no
pass threshold (judgement is left to the listener, as everywhere in this tool).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import recipe as recipe_mod
from .recipe import Recipe, Section
from .separate import separate_vocals
from .vocals import FRAME_SECONDS, frame_levels, sung_reference

# Floor for voiced frames. Frames within this margin of the sung sections' median RMS (sung_reference) count
# as singing, which keeps breaths and silent gaps between lines out of the difference. Including frames where
# only the backing plays between lines would pull the difference negative regardless of the voice's level.
VOICED_MARGIN_DB = 12.0

# Sections with fewer voiced frames than this get no value (the median would rest on a few frames).
MIN_VOICED_FRAMES = 20


@dataclass
class SectionBalance:
    section: Section
    # median "voice − backing" (dB) over voiced frames; None when it cannot be computed
    vocal_over_backing_db: float | None
    voiced_frames: int


def _mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def analyse_balance(vocals_wav: Path, backing_wav: Path, recipe: Recipe) -> list[SectionBalance]:
    vocals, sr = _mono(vocals_wav)
    backing, backing_sr = _mono(backing_wav)
    if backing_sr != sr:
        raise ValueError(f"stem sample rates differ: {sr} / {backing_sr}")
    length = min(len(vocals), len(backing))
    vocals, backing = vocals[:length], backing[:length]
    seconds = length / sr

    floor = sung_reference(vocals, sr, recipe) - VOICED_MARGIN_DB

    results: list[SectionBalance] = []
    for section in recipe.sections:
        if section.instrumental:
            continue
        lo, hi = int(section.start * sr), int(min(section.end, seconds) * sr)
        v_db = frame_levels(vocals[lo:hi], sr)
        a_db = frame_levels(backing[lo:hi], sr)
        n = min(len(v_db), len(a_db))
        voiced = v_db[:n] > floor
        count = int(voiced.sum())
        value = float(np.median(v_db[:n][voiced] - a_db[:n][voiced])) if count >= MIN_VOICED_FRAMES else None
        results.append(SectionBalance(section=section, vocal_over_backing_db=value, voiced_frames=count))
    return results


def _render(title: str, balances: list[SectionBalance]) -> str:
    lines = [
        title,
        "",
        f"{'section':<10} {'span':<14} {'voice-back':>10} {'voiced':>8}",
        "-" * 48,
    ]
    for b in balances:
        span = f"{b.section.start:.0f}-{b.section.end:.0f}s"
        value = "-" if b.vocal_over_backing_db is None else f"{b.vocal_over_backing_db:+.1f}dB"
        lines.append(f"{b.section.name:<10} {span:<14} {value:>10} {b.voiced_frames * FRAME_SECONDS:>7.1f}s")
    lines += [
        "-" * 48,
        "Median over voiced frames only. Negative means the backing is louder.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe.balance", description=__doc__)
    parser.add_argument("audio", type=Path, help="audio (mp3/wav)")
    parser.add_argument("recipe", type=Path, help="recipe.json")
    parser.add_argument("--out", type=Path, default=Path("out"), help="where to write the stems")
    parser.add_argument("--device", help="device for demucs (default: picks mps / cuda / cpu)")
    args = parser.parse_args()

    vocals = separate_vocals(args.audio, args.out, device=args.device)
    backing = vocals.with_name("no_vocals.wav")
    recipe = recipe_mod.load(args.recipe)
    print(_render(f"{recipe.title}  /  {recipe.tempo} BPM", analyse_balance(vocals, backing, recipe)))


if __name__ == "__main__":
    main()
