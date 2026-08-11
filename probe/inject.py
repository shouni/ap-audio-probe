"""検出力を確かめるための陽性対照を合成する。

クリーンな曲を1本測って「違反なし」と出ても、それは検出できることの証明には
なりません。何を渡しても「クリーン」を返す実装でも同じ結果になるためです。
そこで、インスト区間に歌声を意図的に混ぜた音源を作り、それを検出できるかで
検出力を確かめます。混ぜる量を変えれば、判定が反転する境目も測れます。
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
    """target 区間に、source_start から取った歌声を gain_db で混ぜた音源を書く。

    gain_db は元の歌声に対する相対量です。0dB ならサビと同じ音量で混ざります。
    """
    mix, sr = sf.read(mix_path, dtype="float32", always_2d=True)
    vocals, vsr = sf.read(vocals_path, dtype="float32", always_2d=True)
    if sr != vsr:
        raise ValueError(f"サンプルレートが違う: mix={sr} vocals={vsr}")

    start = int(target.start * sr)
    end = min(int(target.end * sr), len(mix))
    length = end - start
    if length <= 0:
        raise ValueError(f"区間が音源の外にある: {target.name}")

    src = int(source_start * sr)
    snippet = vocals[src : src + length]
    if len(snippet) < length:
        raise ValueError("元にする歌声が足りない: source_start を前にずらすこと")

    out = mix.copy()
    out[start:end] += snippet * (10.0 ** (gain_db / 20.0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # float32 で書くのは、混ぜた結果が 0dBFS を超えてもクリップさせないためです。
    # ここでクリップすると、測っているのが混入なのか歪みなのか分からなくなります。
    sf.write(out_path, out, sr, subtype="FLOAT")
    return out_path
