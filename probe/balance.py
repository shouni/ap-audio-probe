"""歌唱セクションごとに、声が伴奏よりどれだけ前に出ているかを測る。

    python -m probe.balance audio/foo.wav recipes/foo.json

講評モデルの「サビでシンセが声を覆う」という指摘を、数字で確かめるための計測です。
demucs で分けた 2 本の stem（vocals.wav と no_vocals.wav）を同じ時刻で比べ、
声が鳴っているフレームだけで「声 − 伴奏」の dB 差の中央値を出します。

2026-10-17 の 4 曲では、サビが -4〜-7dB（伴奏のほうが大きい）、Verse が -1dB 前後で、
講評が「埋もれる」と言った区間と一致しました。生成側の edm_pop のサビ指示を直した後の
曲が -3.2 / -3.6dB です。値は曲ごとの比較と、プロンプトを変えた前後の比較に使います。
何 dB なら合格という閾値は置いていません（判定は聴く人に残す、というこのツールの方針）。
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from . import recipe as recipe_mod
from .recipe import Recipe, Section
from .separate import separate_vocals
from .vocals import FRAME_SECONDS, frame_levels, sung_reference

# 声のフレームと見なす床。歌唱区間の RMS 中央値（sung_reference）からこの幅までを
# 「歌っている」とし、息継ぎや無音の行間を差の計算に入れません。行間で伴奏だけが
# 鳴っているフレームを含めると、声の大きさと無関係に差がマイナスへ引っ張られます。
VOICED_MARGIN_DB = 12.0

# 有声フレームがこれより少ない区間は、差を出しません（中央値が数フレームで決まるため）。
MIN_VOICED_FRAMES = 20


@dataclass
class SectionBalance:
    section: Section
    # 声が鳴っているフレームでの「声 − 伴奏」の中央値（dB）。出せなければ None。
    vocal_over_backing_db: float | None
    voiced_frames: int


def _mono(path: Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def analyse_balance(vocals_wav: Path, backing_wav: Path, recipe: Recipe) -> list[SectionBalance]:
    vocals, sr = _mono(vocals_wav)
    backing, backing_sr = _mono(backing_wav)
    if backing_sr != sr:
        raise ValueError(f"stem のサンプルレートが揃っていない: {sr} / {backing_sr}")
    length = min(len(vocals), len(backing))
    vocals, backing = vocals[:length], backing[:length]
    seconds = length / sr

    floor = sung_reference(vocals, sr, recipe) - VOICED_MARGIN_DB

    results: list[SectionBalance] = []
    for section in recipe.sections:
        if section.instrumental:
            continue
        lo, hi = int(section.start * sr), int(min(section.end, seconds) * sr)
        v_db = frame_levels(vocals[lo:hi], sr)
        a_db = frame_levels(backing[lo:hi], sr)
        n = min(len(v_db), len(a_db))
        voiced = v_db[:n] > floor
        count = int(voiced.sum())
        value = float(np.median(v_db[:n][voiced] - a_db[:n][voiced])) if count >= MIN_VOICED_FRAMES else None
        results.append(SectionBalance(section=section, vocal_over_backing_db=value, voiced_frames=count))
    return results


def _render(title: str, balances: list[SectionBalance]) -> str:
    lines = [
        title,
        "",
        f"{'section':<10} {'区間':<14} {'声 − 伴奏':>10} {'有声':>8}",
        "-" * 48,
    ]
    for b in balances:
        span = f"{b.section.start:.0f}-{b.section.end:.0f}s"
        value = "-" if b.vocal_over_backing_db is None else f"{b.vocal_over_backing_db:+.1f}dB"
        lines.append(f"{b.section.name:<10} {span:<14} {value:>10} {b.voiced_frames * FRAME_SECONDS:>7.1f}s")
    lines += [
        "-" * 48,
        "声が鳴っているフレームだけで測った中央値です。マイナスは伴奏のほうが大きいことを示します。",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="probe.balance", description=__doc__)
    parser.add_argument("audio", type=Path, help="音源 (mp3/wav)")
    parser.add_argument("recipe", type=Path, help="recipe.json")
    parser.add_argument("--out", type=Path, default=Path("out"), help="stem の出力先")
    parser.add_argument("--device", help="demucs の実行デバイス (省略時は mps / cuda / cpu から自動)")
    args = parser.parse_args()

    vocals = separate_vocals(args.audio, args.out, device=args.device)
    backing = vocals.with_name("no_vocals.wav")
    recipe = recipe_mod.load(args.recipe)
    print(_render(f"{recipe.title}  /  {recipe.tempo} BPM", analyse_balance(vocals, backing, recipe)))


if __name__ == "__main__":
    main()
