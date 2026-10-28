"""セクション別の帯域バランスを測る。

生成側のマスタリング処理は高域だけを圧縮しています。その定数の根拠は「サビの当該
帯域が -27dBFS 前後、Verse は -35dBFS 前後」という実測でしたが、測定そのものは
残っていませんでした。ここで同じ量を曲を跨いで測り直します。

測れるのは処理後の音だけです。コンプレッサがどれだけ効いたかは処理前と比べないと
分かりません。「効いている」ことは言えても「必要である」ことの証明にはなりません。

この計測で 11曲を測った結果、閾値が固定値では曲ごとにばらつくと分かり、生成側は
2026-08-15 に曲ごとの相対へ変えました。その閾値は assessment.json から読みます
（「圧縮域」の列の説明は BandLevels の前のコメントを参照）。

    python -m probe.spectrum audio/foo.wav recipes/foo.json
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

from . import recipe as recipe_mod
from .assessment import load as load_assessment
from .recipe import Recipe, Section
from .vocals import SILENCE_DBFS, dbfs

# マスタリング処理の acrossover=split=6000 10000 に合わせた帯域です。圧縮がかかるのは
# この二つに挟まれた 6-10kHz だけで、生成側は body / treble / air と呼んでいます。
#
# 下限は 2026-09-05 まで 4000 でした。コンプレッサはミックス済みの音にかかるので声と
# 楽器を区別できず、子音と存在感が乗る 4-6kHz を含めると、サビで声が張るほどその声自身が
# 閾値を踏みます（「ボーカルが籠る」の原因）。生成側はこの日、圧縮する側と閾値を測る側の
# 両方を 6000 へ動かしました。ここも同じ値でなければ、圧縮されていない帯域を圧縮量として
# 読むことになります。
CROSSOVER_HZ = (6000.0, 10000.0)

# 圧縮対象から外れた 4-6kHz を、それより下と分けて出すための境界です。
#
# 生成側のチェーンでは 6kHz 以下がひとまとめ（body）で、この線は存在しません。それでも
# 分けて表示するのは、2026-09-05 の変更で無処理に戻ったのがまさにこの帯域で、「声の芯が
# 戻ったか」を見る列が要るためです。判定（圧縮域）には使いません。
PRESENCE_HZ = 4000.0

# キックとベースが乗る帯域。低域のアタック感（LOW_PUNCH_PERCENTILES）を測るためだけの線で、
# マスタリング処理には対応する分割がありません。
LOW_PUNCH_HZ = (40.0, 120.0)

# 低域の短時間レベルの上位値と中央値の差を「アタック感」として出します。キックが立っていれば
# 打点だけが持ち上がって差が開き、ベースやパッドが低域を埋めていれば差が詰まります。
#
# 2026-10-28 に、同じ指示から作った曲を Flow Music 版と比べたときに要った量です。曲全体
# (-40dB 床) で測ると sparkle_rock の 3 曲が 3.8〜5.3dB、edm_pop 20 曲の中央値が 8.1dB、
# Flow Music 版が 8.0dB でした。生成側のマスタリングを Flow Music 版に掛けても 7.7 → 7.6dB
# で、この差はマスタリングではなく生成で決まっています。合否の線は置いていません。
LOW_PUNCH_PERCENTILES = (95, 50)

# 相対化より前、マスタリング処理が根拠にしていた実測値。11曲を測り直したところ
# サビの短時間レベルは -29.3〜-31.4dBFS で、この前提より低いところに集まっていました。
# どちらも当時の 4-10kHz についての数字で、6-10kHz とは別の量です。
ASSUMED_CHORUS_DBFS = -27.0
ASSUMED_VERSE_DBFS = -35.0


def _band(x: np.ndarray, sr: int, low: float | None, high: float | None) -> np.ndarray:
    nyq = sr / 2.0
    if low and high:
        sos = butter(4, [low / nyq, min(high, nyq * 0.99) / nyq], btype="bandpass", output="sos")
    elif high:
        sos = butter(4, min(high, nyq * 0.99) / nyq, btype="lowpass", output="sos")
    else:
        sos = butter(4, low / nyq, btype="highpass", output="sos")
    return sosfiltfilt(sos, x)


# コンプレッサの検出器に合わせた短時間窓。マスタリング処理の attack=5ms / release=80ms が
# 追随する時間スケールです。セクション全体の RMS は 36 秒を平均してしまうため、
# 圧縮がかかるかどうかの判断には使えません。
DETECTOR_WINDOW_SECONDS = 0.05
DETECTOR_HOP_SECONDS = 0.01


# 「圧縮域」の列は、生成側が assessment.json に残した閾値（mastering.treble_threshold_dbfs）に
# 音量合わせのゲインを足し、master.wav の上で引き直した線で数えます。
#
# 以前はマスタリング処理の固定値（acompressor=threshold=0.025、-32.0dBFS）で数えていました。
# 生成側は 2026-08-15 に閾値を曲ごとの相対へ変えており、それ以降の曲ではこの列が実際に
# かかった量を表していませんでした。曲ごとの閾値はログにしか残らなかったため、2026-10-29 に
# 生成側の記録へ載せる変更を入れ、ここはそれを読むだけにしています。記録の無い曲では列を
# 出しません。推測した閾値で数えると、読む人が実際の量と取り違えます。
#
# コンプレッサの出力が閾値を超えるのは入力が超えたときだけなので（静特性では単調）、
# 処理後の音で数えた時間は、圧縮が掛かった時間の近似になります。帯域の切り方は
# acrossover と違い、平均で 0.35dB ずれます（masterer.go の trebleThresholdOffsetDB）。
@dataclass
class BandLevels:
    section: Section
    low: float
    # presence は 4-6kHz です。2026-09-05 以降は圧縮対象ではありません。
    presence: float
    # treble が圧縮対象の 6-10kHz、air は 10kHz 以上です。
    treble: float
    air: float
    # treble_peak は 6-10kHz の短時間レベルの上位値、over_ratio はそれが閾値を超えた時間の割合です。
    # 閾値が分からない曲では over_ratio は None です。
    treble_peak: float
    over_ratio: float | None
    # low_punch は 40-120Hz の短時間レベルの上位値と中央値の差（dB）です。LOW_PUNCH_PERCENTILES を参照。
    low_punch: float


def analyse(audio_path: Path, recipe: Recipe, threshold_dbfs: float | None = None) -> list[BandLevels]:
    """セクションごとの帯域レベルを測る。

    threshold_dbfs は master.wav の上で見た圧縮の閾値です（MasteringSettings.master_threshold_dbfs）。
    省くと over_ratio は None になります。
    """
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    seconds = len(mono) / sr

    lo_cut, hi_cut = CROSSOVER_HZ
    bands = {
        "low": _band(mono, sr, None, PRESENCE_HZ),
        "presence": _band(mono, sr, PRESENCE_HZ, lo_cut),
        "treble": _band(mono, sr, lo_cut, hi_cut),
        "air": _band(mono, sr, hi_cut, None),
        "kick": _band(mono, sr, *LOW_PUNCH_HZ),
    }

    levels = []
    for section in recipe.sections:
        start, end = int(section.start * sr), int(min(section.end, seconds) * sr)
        if end - start <= 0:
            continue
        short_term = _short_term_dbfs(bands["treble"][start:end], sr)
        levels.append(
            BandLevels(
                section=section,
                low=dbfs(bands["low"][start:end]),
                presence=dbfs(bands["presence"][start:end]),
                treble=dbfs(bands["treble"][start:end]),
                air=dbfs(bands["air"][start:end]),
                treble_peak=float(np.percentile(short_term, 95)) if short_term.size else SILENCE_DBFS,
                over_ratio=None
                if threshold_dbfs is None
                else float(np.mean(short_term > threshold_dbfs)) if short_term.size else 0.0,
                low_punch=low_punch(bands["kick"][start:end], sr),
            )
        )
    return levels


def low_punch(kick_band: np.ndarray, sr: int) -> float:
    """40-120Hz に切った信号から、短時間レベルの上位値と中央値の差を返す。"""
    short_term = _short_term_dbfs(kick_band, sr)
    if not short_term.size:
        return 0.0
    high, mid = np.percentile(short_term, LOW_PUNCH_PERCENTILES)
    return float(high - mid)


def _short_term_dbfs(x: np.ndarray, sr: int) -> np.ndarray:
    """短時間窓ごとの RMS を dBFS で返す。コンプレッサの検出器に相当する量。"""
    window = int(DETECTOR_WINDOW_SECONDS * sr)
    hop = int(DETECTOR_HOP_SECONDS * sr)
    if x.size < window:
        return np.array([])

    starts = np.arange(0, x.size - window, hop)
    frames = np.stack([x[s : s + window] for s in starts])
    rms = np.sqrt(np.mean(np.square(frames), axis=1))
    return 20.0 * np.log10(np.maximum(rms, 1e-12))


def render(title: str, levels: list[BandLevels], threshold_dbfs: float | None = None) -> str:
    lines = [
        f"\n{title}",
        f"{'section':<11}{'<4k':>9}{'4-6k':>9}{'6-10k':>9}{'>10k':>9}"
        f"{'6-10k 短時間':>11}{'圧縮域':>5}{'低域起伏':>8}",
        "-" * 78,
    ]
    for lv in levels:
        lines.append(
            f"{lv.section.name:<11}{lv.low:>7.1f}dB{lv.presence:>7.1f}dB"
            f"{lv.treble:>7.1f}dB{lv.air:>7.1f}dB"
            f"{lv.treble_peak:>11.1f}dB"
            f"{'-' if lv.over_ratio is None else f'{lv.over_ratio:.0%}':>8}{lv.low_punch:>8.1f}dB"
        )

    sung = [lv for lv in levels if not lv.section.instrumental]
    chorus = [lv for lv in sung if "Chorus" in lv.section.name]
    verse = [lv for lv in sung if "Verse" in lv.section.name]
    lines.append("-" * 78)
    if chorus and verse:
        c = float(np.mean([lv.treble for lv in chorus]))
        v = float(np.mean([lv.treble for lv in verse]))
        cp = float(np.mean([lv.treble_peak for lv in chorus]))
        lines.append(
            f"サビ {c:.1f}dB / Verse {v:.1f}dB / 差 {c - v:.1f}dB"
            f"  (マスタリングの前提: {ASSUMED_CHORUS_DBFS:.0f} / {ASSUMED_VERSE_DBFS:.0f}、当時は 4-10k)"
        )
        if threshold_dbfs is None:
            lines.append(
                f"サビの短時間レベル {cp:.1f}dB / 閾値は不明"
                "（assessment.json に mastering が無い曲。圧縮域の列も出しません）"
            )
        else:
            lines.append(
                f"サビの短時間レベル {cp:.1f}dB / 閾値 {threshold_dbfs:.1f}dB (master.wav 上)"
                f"  → {'圧縮がかかる' if cp > threshold_dbfs else '閾値に届かない'}"
            )
        # 4-6kHz は 2026-09-05 に圧縮対象から外れた帯域です。声の芯が残っているかを
        # 曲を跨いで追うための行で、判定には使いません。
        cpr = float(np.mean([lv.presence for lv in chorus]))
        vpr = float(np.mean([lv.presence for lv in verse]))
        lines.append(
            f"4-6k(無処理) サビ {cpr:.1f}dB / Verse {vpr:.1f}dB / 差 {cpr - vpr:.1f}dB"
        )

    # Chorus / Verse 以外の歌唱区間（8 セクション構成の Bridge など）。要約の 2 行は
    # 「サビだけ抑える」という設計の確認なので、そこへ混ぜずに別に並べます。
    others = [lv for lv in sung if "Chorus" not in lv.section.name and "Verse" not in lv.section.name]
    if others:
        lines.append(
            "その他の歌唱区間: "
            + " / ".join(f"{lv.section.name} {lv.treble:.1f}dB" for lv in others)
        )
    return "\n".join(lines)


def recorded_threshold(audio: Path, assessments: Path) -> float | None:
    """assessments/<音源の名前>.json に記録された閾値を、master.wav の上の値で返す。"""
    path = assessments / f"{audio.stem}.json"
    if not path.exists():
        return None
    mastering = load_assessment(path).mastering
    return None if mastering is None else mastering.master_threshold_dbfs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, nargs="+", help="audio recipe を交互に並べる")
    parser.add_argument("--assessments", type=Path, default=Path("assessments"),
                        help="assessment.json の置き場所（圧縮の閾値をここから読みます）")
    args = parser.parse_args()

    if len(args.pairs) % 2:
        raise SystemExit("audio と recipe を対で渡してください")

    for audio, recipe_path in zip(args.pairs[::2], args.pairs[1::2]):
        rec = recipe_mod.load(recipe_path)
        threshold = recorded_threshold(audio, args.assessments)
        print(render(f"{rec.title}  ({audio.stem})", analyse(audio, rec, threshold), threshold))


if __name__ == "__main__":
    main()
