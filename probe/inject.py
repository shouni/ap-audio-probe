"""Synthesise positive controls to check detection power.

Measuring one clean song and getting "no violations" does not prove anything can be detected: an
implementation that returns "clean" for anything would give the same result. So it builds audio with
voice deliberately mixed into an instrumental section and checks that it is detected. Varying the amount
also finds where the verdict flips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .recipe import Section


def inject_vocal(
    mix_path: Path,
    vocals_path: Path,
    target: Section,
    source_start: float,
    gain_db: float,
    out_path: Path,
) -> Path:
    """Write audio with voice taken from source_start mixed into the `target` section at gain_db.

    gain_db is relative to the original voice: at 0dB it mixes in as loud as the chorus.
    """
    mix, sr = sf.read(mix_path, dtype="float32", always_2d=True)
    vocals, vsr = sf.read(vocals_path, dtype="float32", always_2d=True)
    if sr != vsr:
        raise ValueError(f"sample rates differ: mix={sr} vocals={vsr}")

    start = int(target.start * sr)
    end = min(int(target.end * sr), len(mix))
    length = end - start
    if length <= 0:
        raise ValueError(f"section lies outside the audio: {target.name}")

    src = int(source_start * sr)
    snippet = vocals[src : src + length]
    if len(snippet) < length:
        raise ValueError("not enough source voice: move source_start earlier")

    out = mix.copy()
    out[start:end] += snippet * (10.0 ** (gain_db / 20.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written as float32 so the mix does not clip above 0dBFS. Clipping here would leave it unclear whether
    # the measurement shows the injected voice or distortion.
    sf.write(out_path, out, sr, subtype="FLOAT")
    return out_path
