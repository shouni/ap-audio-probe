"""Demucs でボーカル stem を分離する。

帯域エネルギーで歌とシンセを見分けようとすると、saw 系のパッドやコードが
歌声と同じ帯域に倍音ごと乗るため誤検出が避けられない。伴奏を実際に取り除いて
から測るほうが、判定が原理的に素直になる。
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODEL = "htdemucs"


def separate_vocals(audio: Path, out_dir: Path, device: str = "mps") -> Path:
    """audio からボーカル stem を作り、その wav のパスを返す。

    既に分離済みなら再実行しない(3分の曲で1分前後かかるため)。
    """
    vocals = out_dir / MODEL / audio.stem / "vocals.wav"
    if vocals.exists():
        return vocals

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", MODEL,
        "-d", device,
        "-o", str(out_dir),
        str(audio),
    ]
    subprocess.run(cmd, check=True)

    if not vocals.exists():
        raise FileNotFoundError(f"demucs は完走したが stem が見つからない: {vocals}")
    return vocals
