"""陽性対照を作って検出力を確かめ、判定が反転する境目を測る。

    .venv/bin/python scripts/calibrate.py audio/foo.wav recipes/foo.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe import recipe as recipe_mod
from probe.inject import inject_vocal
from probe.separate import separate_vocals
from probe.vocals import analyse

GAINS_DB = [0.0, -6.0, -12.0, -18.0, -24.0, -30.0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--section", default="Interlude", help="混入させるインスト区間")
    parser.add_argument("--source-start", type=float, default=40.0, help="歌声を取り出す秒数")
    parser.add_argument("--out", type=Path, default=Path("out"))
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    rec = recipe_mod.load(args.recipe)
    target = next(s for s in rec.sections if s.name == args.section)
    if not target.instrumental:
        raise SystemExit(f"{target.name} はインスト指定ではありません")

    print(f"元の音源を分離中 ...")
    clean_stem = separate_vocals(args.audio, args.out, device=args.device)

    baseline = analyse(clean_stem, rec)
    base = next(s for s in baseline.stats if s.section.name == target.name)
    print(
        f"\n基準: {target.name} は元の音源で {base.rms_dbfs:.1f}dB / 有声率 {base.active_ratio:.0%} "
        f"(歌唱の基準値 {baseline.sung_reference_dbfs:.1f}dB)\n"
    )

    rows = []
    for gain in GAINS_DB:
        tag = f"inject{int(abs(gain)):02d}"
        contaminated = args.audio.parent / f"{args.audio.stem}-{tag}.wav"
        print(f"[{gain:+.0f}dB] 合成して分離中 ...")
        inject_vocal(args.audio, clean_stem, target, args.source_start, gain, contaminated)
        stem = separate_vocals(contaminated, args.out, device=args.device)

        report = analyse(stem, rec)
        stat = next(s for s in report.stats if s.section.name == target.name)
        rows.append((gain, stat.rms_dbfs, stat.active_ratio, report.has_vocals(stat)))

    print(f"\n混入量   {target.name} の vocal RMS   有声率    判定")
    print("-" * 52)
    for gain, rms, active, flagged in rows:
        print(f"{gain:+6.0f}dB {rms:>14.1f}dB {active:>8.0%}    {'★検出' if flagged else 'クリーン(見逃し)'}")
    print("-" * 52)

    detected = [g for g, _, _, f in rows if f]
    missed = [g for g, _, _, f in rows if not f]
    if not detected:
        print("⚠ どの混入量でも検出できていません。判定ロジックが機能していません。")
    elif missed:
        print(f"境目: {min(detected):+.0f}dB までは検出、{max(missed):+.0f}dB 以下は見逃し")
    else:
        print("すべての混入量で検出。より小さい混入量も試す価値があります。")


if __name__ == "__main__":
    main()
