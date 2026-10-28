"""複数の曲を作曲モードごとに並べ、1 曲が外れなのかモード全体の傾向なのかを見る。

    python -m probe.corpus audio/*.wav
    python -m probe.corpus audio/*.wav --balance    # 声 − 伴奏も出す（未分離の曲は demucs が走る）

レシピは recipes/<音源の名前>.json から引きます。見つからない曲は飛ばします。
圧縮量は assessments/<音源の名前>.json に記録された閾値から逆算し、記録が無ければ空欄です。
生成側の高域コンプの設計は「サビだけ踏み、Verse は素通り」なので、この 2 列をモードごとに
並べたものが、閾値のオフセット（masterer.go の trebleThresholdOffsetDB）を詰める材料になります。
超過時間（spectrum の「圧縮域」）ではなく量を並べるのは、時間だけでは瞬間的な山と
持続する圧縮を区別できないためです（spectrum.BandLevels の前のコメント）。

1 曲ずつの計測（loudness / spectrum / balance）は、それだけでは「この値は普通か」に
答えられません。2026-10-28 に sparkle_rock の低域のアタック感が低いと分かったのも、
edm_pop 20 曲と手で並べてからでした。ここではその並べ方を固定します。

サビの値は名前に Chorus を含むセクションの平均です（spectrum.render の要約と同じ規則）。
合否の線は置いていません。モードの中央値から外れた曲を、聴き直す候補として拾う使い方です。
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import recipe as recipe_mod
from .balance import analyse_balance
from .loudness import measure as measure_loudness
from .separate import resolve_device, separate_vocals
from .spectrum import analyse as analyse_spectrum
from .spectrum import recorded_threshold
from .table import cell


@dataclass(frozen=True)
class TrackSummary:
    name: str
    mode: str
    integrated_lufs: float
    true_peak_dbtp: float
    # サビの 40-120Hz の起伏（spectrum.LOW_PUNCH_PERCENTILES）と 6-10kHz の平均レベル。
    chorus_low_punch: float | None
    chorus_treble: float | None
    # 6-10kHz を縮めた量の平均 dB（spectrum.BandLevels.gain_reduction）。
    chorus_gain_reduction: float | None
    verse_gain_reduction: float | None
    # サビの「声 − 伴奏」。--balance を付けたときだけ入ります。
    chorus_vocal_over_backing: float | None


def _section_mean(values: list[tuple[str, float | None]], kind: str = "Chorus") -> float | None:
    picked = [v for name, v in values if kind in name and v is not None]
    return float(np.mean(picked)) if picked else None


def summarise(
    audio: Path, recipe_path: Path, *, balance: bool, out_dir: Path, device: str | None,
    assessments: Path = Path("assessments"),
) -> TrackSummary:
    rec = recipe_mod.load(recipe_path)
    loudness = measure_loudness(audio)
    levels = analyse_spectrum(audio, rec, recorded_threshold(audio, assessments))

    vocal_over_backing = None
    if balance:
        vocals = separate_vocals(audio, out_dir, device=device)
        balances = analyse_balance(vocals, vocals.with_name("no_vocals.wav"), rec)
        vocal_over_backing = _section_mean([(b.section.name, b.vocal_over_backing_db) for b in balances])

    return TrackSummary(
        name=audio.stem,
        mode=rec.compose_mode or "(不明)",
        integrated_lufs=loudness.integrated_lufs,
        true_peak_dbtp=loudness.true_peak_dbtp,
        chorus_low_punch=_section_mean([(lv.section.name, lv.low_punch) for lv in levels]),
        chorus_treble=_section_mean([(lv.section.name, lv.treble) for lv in levels]),
        chorus_gain_reduction=_section_mean([(lv.section.name, lv.gain_reduction) for lv in levels]),
        verse_gain_reduction=_section_mean([(lv.section.name, lv.gain_reduction) for lv in levels], "Verse"),
        chorus_vocal_over_backing=vocal_over_backing,
    )


# 列の幅。見出しに全角が混ざるので、table.cell で表示幅を揃えます。
NAME_WIDTH = 38
VALUE_WIDTH = 9


def _fmt(value: float | None, signed: bool = False, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


def _median(values: list[float | None]) -> float | None:
    picked = [v for v in values if v is not None]
    return float(np.median(picked)) if picked else None


def _row(label: str, cells: list[str]) -> str:
    return cell(label, NAME_WIDTH) + "".join(cell(c, VALUE_WIDTH, right=True) for c in cells)


def _values(lufs, tp, punch, treble, chorus_gr, verse_gr, vob) -> list[str]:
    return [
        _fmt(lufs), _fmt(tp), _fmt(punch), _fmt(treble),
        _fmt(chorus_gr, digits=2), _fmt(verse_gr, digits=2), _fmt(vob, signed=True),
    ]


def _cells(t: TrackSummary) -> tuple:
    return (
        t.integrated_lufs, t.true_peak_dbtp, t.chorus_low_punch, t.chorus_treble,
        t.chorus_gain_reduction, t.verse_gain_reduction, t.chorus_vocal_over_backing,
    )


def render(tracks: list[TrackSummary]) -> str:
    rule = "-" * (NAME_WIDTH + VALUE_WIDTH * 7)
    lines = [
        _row("track", ["LUFS", "TP", "低域起伏", "6-10k", "圧縮量", "圧縮量", "声−伴奏"]),
        _row("", ["", "dBTP", "(サビ)", "(サビ)", "(サビ)", "(Verse)", "(サビ)"]),
        rule,
    ]

    for mode in sorted({t.mode for t in tracks}):
        group = [t for t in tracks if t.mode == mode]
        for t in group:
            lines.append(_row(t.name, _values(*_cells(t))))
        columns = zip(*(_cells(t) for t in group))
        lines.append(_row(f"  ↳ {mode} の中央値 ({len(group)} 曲)", _values(*(_median(list(c)) for c in columns))))
        lines.append(rule)

    lines += [
        "低域起伏: 40-120Hz の短時間レベルの上位値 − 中央値。小さいほどキックが平たい。",
        "圧縮量: 6-10kHz を縮めた量の平均 dB（記録された閾値から逆算）。記録の無い曲は空欄。",
        "声−伴奏: 声が鳴っているフレームでの中央値。マイナスは伴奏のほうが大きい。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe.corpus", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path, nargs="+", help="音源 (master.wav 推奨)")
    parser.add_argument("--recipes", type=Path, default=Path("recipes"), help="recipe.json の置き場所")
    parser.add_argument("--assessments", type=Path, default=Path("assessments"),
                        help="assessment.json の置き場所（圧縮の閾値をここから読みます）")
    parser.add_argument("--balance", action="store_true", help="サビの声 − 伴奏も出す（demucs で分離します）")
    parser.add_argument("--out", type=Path, default=Path("out"), help="stem の出力先")
    parser.add_argument("--device", help="demucs の実行デバイス (省略時は mps / cuda / cpu から自動)")
    args = parser.parse_args()

    device = resolve_device(args.device) if args.balance else None
    tracks = []
    for audio in args.audio:
        recipe_path = args.recipes / f"{audio.stem}.json"
        if not recipe_path.exists():
            print(f"レシピが無いので飛ばします: {audio.name}", file=sys.stderr)
            continue
        print(f"測定中: {audio.name} ...", file=sys.stderr)
        tracks.append(summarise(audio, recipe_path, balance=args.balance, out_dir=args.out,
                                device=device, assessments=args.assessments))

    if not tracks:
        raise SystemExit("測れる曲がありませんでした（レシピが見つからない）")
    print()
    print(render(tracks))


if __name__ == "__main__":
    main()
