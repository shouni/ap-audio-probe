"""Separate the vocal stem with Demucs.

Telling voice from synths by band energy cannot avoid false positives: saw pads and chords sit in the same
band as the voice, harmonics and all. Actually removing the backing before measuring makes the verdict
straightforward in principle.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODEL = "htdemucs"


def resolve_device(device: str | None) -> str:
    """Choose the device for demucs: the given one, or whatever is available.

    When "mps" was hard-coded as the default, forgetting --device cpu off Apple Silicon crashed demucs right
    at start. Importing torch takes seconds, so it is not touched when a device is given.
    """
    if device:
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def separate_vocals(audio: Path, out_dir: Path, device: str | None = None) -> Path:
    """Create the vocal stem from `audio` and return the wav path.

    Does not run again if already separated (about 20 s for a 3-minute song even on MPS).
    Without `device`, resolve_device picks one.
    """
    # demucs names its output without the extension, so an mp3 and a wav with the same base name would
    # write to the same place, and a swapped measurement target would go unnoticed. Keep them apart with a
    # work directory that includes the extension.
    work = out_dir / audio.suffix.lstrip(".")
    vocals = work / MODEL / audio.stem / "vocals.wav"
    if vocals.exists():
        return vocals

    work.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", MODEL,
        "-d", resolve_device(device),
        "-o", str(work),
        str(audio),
    ]
    subprocess.run(cmd, check=True)

    if not vocals.exists():
        raise FileNotFoundError(f"demucs finished but the stem is missing: {vocals}")
    return vocals
