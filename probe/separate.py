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

    既に分離済みなら再実行しない(MPS でも3分の曲に20秒程度かかるため)。
    """
    # demucs は拡張子を落とした名前で出力するため、同じ基底名の mp3 と wav が
    # 同じ場所に書き込んでしまいます。測る対象が入れ替わっても気づけないので、
    # 拡張子まで含めた作業ディレクトリに分けます。
    work = out_dir / audio.suffix.lstrip(".")
    vocals = work / MODEL / audio.stem / "vocals.wav"
    if vocals.exists():
        return vocals

    out_dir = work

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
