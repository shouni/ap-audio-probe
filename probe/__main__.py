"""ap-audio-probe: 生成された曲がレシピの指示どおりに鳴っているかを測る。

判定を出すが、止めはしない。数字と根拠を並べて、直すかどうかは人が決める。

    python -m probe audio/foo.mp3 recipes/foo.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

from . import recipe as recipe_mod
from .separate import separate_vocals
from .vocals import Report, analyse


def _render(report: Report) -> str:
    lines = [
        f"{report.recipe.title}  /  {report.recipe.tempo} BPM",
        f"実尺 {report.audio_seconds:.1f}s  (レシピ指定 {report.recipe.declared_duration:.0f}s)",
        "",
        f"{'section':<10} {'指定':<6} {'区間':<16} {'vocal RMS':>10} {'peak':>8} {'有声率':>7}  判定",
        "-" * 76,
    ]
    for s in report.stats:
        kind = "インスト" if s.section.instrumental else "歌唱"
        span = f"{s.section.start:.0f}-{s.clamped_end:.1f}s"
        if s.section.instrumental:
            verdict = "★声あり" if report.has_vocals(s) else "クリーン"
        else:
            verdict = "-"
        notes = []
        if s.bleed:
            notes.append(f"食い込み {s.bleed:.1f}s 除外")
        if s.truncated:
            notes.append(f"尺不足 {s.section.end - s.clamped_end:.1f}s")
        if notes:
            verdict += f" ({'、'.join(notes)})"
        lines.append(
            f"{s.section.name:<10} {kind:<6} {span:<16} "
            f"{s.rms_dbfs:>9.1f}dB {s.peak_dbfs:>7.1f}dB {s.active_ratio:>6.0%}  {verdict}"
        )

    lines += [
        "-" * 76,
        f"歌唱セクションの基準値: {report.sung_reference_dbfs:.1f} dBFS "
        f"(この値から 12dB 以内、または有声率 20% 超で「声あり」)",
    ]
    if any(s.bleed for s in report.stats):
        lines.append("※ 「食い込み」は隣の歌唱セクションから端に届いた声(弱起や歌尾)です。")
    lines.append("")
    if report.violations:
        names = "、".join(s.section.name for s in report.violations)
        lines.append(f"⚠ インスト指定なのに声が乗っている: {names}")
    else:
        lines.append("✓ インスト指定セクションはすべてクリーン")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe", description=__doc__)
    parser.add_argument("audio", type=Path, help="音源 (mp3/wav)")
    parser.add_argument("recipe", type=Path, help="ap-comp の recipe.json")
    parser.add_argument("--out", type=Path, default=Path("out"), help="stem の出力先")
    parser.add_argument("--device", default="mps", help="demucs の実行デバイス")
    args = parser.parse_args()

    print(f"[1/2] ボーカル stem を分離中 ({args.device}) ...")
    vocals = separate_vocals(args.audio, args.out, device=args.device)

    print(f"[2/2] セクション別に測定中 ...\n")
    print(_render(analyse(vocals, recipe_mod.load(args.recipe))))


if __name__ == "__main__":
    main()
