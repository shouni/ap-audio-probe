"""複数の曲のラウドネスを測って揃い具合を見る。

ap-comp の検査は1曲ずつしか採点しないため、シリーズとして並べたときの
音量差は構造的に見えません。配信で効くのはそこなので、曲を跨いで比べます。

    python -m probe.loudness audio/*.wav
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# 同じリリースに並べる曲の音量差の許容幅。マスタリングの慣習的な目安です。
SPREAD_TOLERANCE_LU = 1.0

# 配信用の真正ピーク上限。ロッシー変換で歪まないための一般的なガイドラインです。
TRUE_PEAK_CEILING_DBTP = -1.0


@dataclass(frozen=True)
class Loudness:
    path: Path
    integrated_lufs: float
    true_peak_dbtp: float
    lra: float

    @property
    def peak_is_hot(self) -> bool:
        return self.true_peak_dbtp > TRUE_PEAK_CEILING_DBTP


def measure(path: Path) -> Loudness:
    """ffmpeg の loudnorm で BS.1770 の実測値を取る。"""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-nostats", "-i", str(path),
         "-af", "loudnorm=print_format=json", "-f", "null", "-"],
        capture_output=True, text=True, check=True,
    )
    # loudnorm は測定結果を stderr の末尾に JSON で吐きます。
    blocks = re.findall(r"\{[^{}]*\}", result.stderr, re.DOTALL)
    if not blocks:
        raise RuntimeError(f"loudnorm の出力を解析できません: {path}")
    data = json.loads(blocks[-1])

    return Loudness(
        path=path,
        integrated_lufs=float(data["input_i"]),
        true_peak_dbtp=float(data["input_tp"]),
        lra=float(data["input_lra"]),
    )


def render(tracks: list[Loudness]) -> str:
    lines = [
        f"{'track':<34} {'LUFS':>9} {'True Peak':>11} {'LRA':>7}",
        "-" * 64,
    ]
    for t in tracks:
        flag = "  ⚠ ピーク高い" if t.peak_is_hot else ""
        lines.append(
            f"{t.path.stem[:33]:<34} {t.integrated_lufs:>8.1f} "
            f"{t.true_peak_dbtp:>10.1f} {t.lra:>6.1f}{flag}"
        )
    lines.append("-" * 64)

    levels = [t.integrated_lufs for t in tracks]
    spread = max(levels) - min(levels)
    lines.append(f"音量差: {spread:.1f} LU (許容目安 {SPREAD_TOLERANCE_LU:.1f} LU)")

    if len(tracks) < 2:
        lines.append("比較には2曲以上が要ります。")
    elif spread > SPREAD_TOLERANCE_LU:
        loudest = max(tracks, key=lambda t: t.integrated_lufs)
        quietest = min(tracks, key=lambda t: t.integrated_lufs)
        lines.append(
            f"⚠ 揃っていません。最大 {loudest.path.stem[:24]} と "
            f"最小 {quietest.path.stem[:24]} で {spread:.1f} LU 差があります。"
        )
    else:
        lines.append("✓ シリーズとして並べても違和感の出ない範囲です。")

    hot = [t for t in tracks if t.peak_is_hot]
    if hot:
        lines.append(
            f"⚠ 真正ピークが {TRUE_PEAK_CEILING_DBTP:.1f} dBTP を超えている曲が "
            f"{len(hot)} 本あります(ロッシー変換で歪む恐れ)。"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tracks", type=Path, nargs="+", help="測る音源(master.wav 推奨)")
    args = parser.parse_args()

    tracks = []
    for path in args.tracks:
        print(f"測定中: {path.name} ...", file=sys.stderr)
        tracks.append(measure(path))

    print()
    print(render(sorted(tracks, key=lambda t: t.integrated_lufs)))


if __name__ == "__main__":
    main()
