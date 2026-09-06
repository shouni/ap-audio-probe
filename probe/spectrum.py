"""セクション別の帯域バランスを測る。

生成側のマスタリング処理は高域だけを圧縮しています。その定数の根拠は「サビの当該
帯域が -27dBFS 前後、Verse は -35dBFS 前後」という実測でしたが、測定そのものは
残っていませんでした。ここで同じ量を曲を跨いで測り直します。

測れるのは処理後の音だけです。コンプレッサがどれだけ効いたかは処理前と比べないと
分かりません。「効いている」ことは言えても「必要である」ことの証明にはなりません。

この計測で 11曲を測った結果、閾値が固定値では曲ごとにばらつくと分かり、生成側は
2026-08-15 に曲ごとの相対へ変えました。COMPRESSOR_THRESHOLD_DBFS を参照。

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

# マスタリング処理の acompressor=threshold=0.025 を dBFS にした値です。サビがこれを超えると
# 圧縮がかかり、Verse は素通りする、という設計になっています。
#
# ただしこれは 2026-08-15 より前の固定値です。生成側はその日に、曲自身の高域平均から
# +2.5dB 上へ置く相対の閾値に変えました。probe 側は未対応なので、それ以降に生成された曲では
# 「圧縮域」の列は実際にかかった量ではありません。基準の帯域そのものも 2026-09-05 に
# 4-10kHz から 6-10kHz へ動いているので、その前後の曲を同じ列で比べることもできません。
COMPRESSOR_THRESHOLD_DBFS = 20.0 * np.log10(0.025)

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
    treble_peak: float
    over_ratio: float


def analyse(audio_path: Path, recipe: Recipe) -> list[BandLevels]:
    data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    seconds = len(mono) / sr

    lo_cut, hi_cut = CROSSOVER_HZ
    bands = {
        "low": _band(mono, sr, None, PRESENCE_HZ),
        "presence": _band(mono, sr, PRESENCE_HZ, lo_cut),
        "treble": _band(mono, sr, lo_cut, hi_cut),
        "air": _band(mono, sr, hi_cut, None),
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
                over_ratio=float(np.mean(short_term > COMPRESSOR_THRESHOLD_DBFS))
                if short_term.size
                else 0.0,
            )
        )
    return levels


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


def render(title: str, levels: list[BandLevels]) -> str:
    lines = [
        f"\n{title}",
        f"{'section':<11}{'<4k':>9}{'4-6k':>9}{'6-10k':>9}{'>10k':>9}"
        f"{'6-10k 短時間':>11}{'圧縮域':>5}",
        "-" * 68,
    ]
    for lv in levels:
        lines.append(
            f"{lv.section.name:<11}{lv.low:>7.1f}dB{lv.presence:>7.1f}dB"
            f"{lv.treble:>7.1f}dB{lv.air:>7.1f}dB"
            f"{lv.treble_peak:>11.1f}dB{lv.over_ratio:>8.0%}"
        )

    sung = [lv for lv in levels if not lv.section.instrumental]
    chorus = [lv for lv in sung if "Chorus" in lv.section.name]
    verse = [lv for lv in sung if "Verse" in lv.section.name]
    lines.append("-" * 68)
    if chorus and verse:
        c = float(np.mean([lv.treble for lv in chorus]))
        v = float(np.mean([lv.treble for lv in verse]))
        cp = float(np.mean([lv.treble_peak for lv in chorus]))
        lines.append(
            f"サビ {c:.1f}dB / Verse {v:.1f}dB / 差 {c - v:.1f}dB"
            f"  (マスタリングの前提: {ASSUMED_CHORUS_DBFS:.0f} / {ASSUMED_VERSE_DBFS:.0f}、当時は 4-10k)"
        )
        lines.append(
            f"サビの短時間レベル {cp:.1f}dB / 閾値 {COMPRESSOR_THRESHOLD_DBFS:.1f}dB"
            f"  → {'圧縮がかかる' if cp > COMPRESSOR_THRESHOLD_DBFS else '閾値に届かない'}"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pairs", type=Path, nargs="+", help="audio recipe を交互に並べる")
    args = parser.parse_args()

    if len(args.pairs) % 2:
        raise SystemExit("audio と recipe を対で渡してください")

    for audio, recipe_path in zip(args.pairs[::2], args.pairs[1::2]):
        rec = recipe_mod.load(recipe_path)
        print(render(f"{rec.title}  ({audio.stem})", analyse(audio, rec)))


if __name__ == "__main__":
    main()
