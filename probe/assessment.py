"""生成側が残した assessment.json と、手元の音源を突き合わせる。

生成側はジョブごとに `music/<job_id>/assessment.json` を置きます。中身は
生成時に測った実測値と、聴かせたモデルの講評です。ここではその記録と、
同じ量を手元で測り直した値を並べます。

    python -m probe.assessment audio/foo.wav assessments/foo.json

**測っている対象が違います。** 生成側が測るのは Web 配信用の mp3
（`Inspect(mastered.Web)`）で、こちらが測るのは配信用マスターの wav です。
差はその 2 つの差であって、どちらかが間違っているわけではありません。
説明が付く差（下の PADDING / peak_amplitude の注記）とそうでない差を分けるのが
この計測の目的です。

再実装した測り方は生成側の inspector.go に合わせてあります。同じ数字を
別の道具で出すためで、実装が離れると差が「mp3 と wav の差」なのか
「測り方の差」なのか分からなくなります。
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt

from .loudness import measure as measure_loudness

# 生成側 inspector.go の定数。名前も値もあちらに合わせています。
ANALYSIS_WINDOW_SECONDS = 0.05
SILENCE_THRESHOLD_RMS = 0.005
CLIPPING_THRESHOLD = 0.999
MIN_AUDIO_DBFS = -120.0

# Inspector の帯域。マスタリングの圧縮帯域（spectrum.py: CROSSOVER_HZ）とは別物で、
# 2026-09-05 に圧縮が 6kHz 始まりへ動いた後も、この記録は 4-10kHz のままです。
MID_BAND_HZ = (200.0, 4000.0)
HIGH_BAND_HZ = (4000.0, 10000.0)

# mp3 デコーダが足す padding。2,304 サンプル = MP3 フレーム 2 つぶんで、
# 生成側の尺はこのぶん長く出ます。44.1kHz で 0.052 秒。
MP3_PADDING_SAMPLES = 2304

# 説明の付かない差として拾う幅。mp3 と wav の符号化差は実測で 0.1dB 台なので、
# その一桁上に置いています。尺は padding を引いた残りで見ます。
LEVEL_TOLERANCE_DB = 0.5
DURATION_TOLERANCE_SECONDS = 0.2
RATIO_TOLERANCE = 0.02


@dataclass(frozen=True)
class Measurements:
    """inspector.go の AudioMeasurements と同じ項目。"""

    duration_seconds: float
    leading_silence_seconds: float
    trailing_silence_seconds: float
    longest_internal_silence_seconds: float
    silent_ratio: float
    peak_amplitude: float
    clipped_sample_ratio: float
    rms_dbfs: float
    mid_band_dbfs: float
    high_band_dbfs: float
    true_peak_dbtp: float
    integrated_lufs: float

    @property
    def treble_headroom_db(self) -> float:
        return self.mid_band_dbfs - self.high_band_dbfs


@dataclass(frozen=True)
class Record:
    """assessment.json（生成側の AudioAssessmentRecord）から使う部分。"""

    job_id: str
    created_at: str
    score: float
    summary: str
    passed: bool
    mastered: bool
    measurements: Measurements
    issues: list[tuple[str, str]]
    review: dict | None
    lyrics: dict | None


def _bandpass_sos(sample_rate: int, low_hz: float, high_hz: float) -> np.ndarray:
    """inspector.go の newBandFilter と同じ 2 次バンドパス（RBJ Cookbook）。

    spectrum.py の 4 次バターワースではなくこちらを使うのは、生成側が記録した
    mid/high band と同じ数字を出すためです。1 段のバンドパスは裾が緩く、主帯域には
    キックやベースがかなり混ざります（inspector.go のコメントに但し書きがあります）。
    帯域の絶対量としてではなく、記録との突き合わせにだけ使ってください。
    """
    center = np.sqrt(low_hz * high_hz)
    q = center / (high_hz - low_hz)
    w0 = 2.0 * np.pi * center / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    a0 = 1.0 + alpha
    return np.array([[alpha / a0, 0.0, -alpha / a0, 1.0, -2.0 * np.cos(w0) / a0, (1.0 - alpha) / a0]])


def _dbfs(amplitude: float) -> float:
    return max(MIN_AUDIO_DBFS, 20.0 * np.log10(amplitude)) if amplitude > 0 else MIN_AUDIO_DBFS


def _silence(windows: np.ndarray) -> tuple[float, float, float, float]:
    """無音窓の並びから、先頭・末尾・曲中最長・全体の割合を出す。

    数え方は inspector.go の summariseSilence に合わせています（曲中最長は先頭と末尾を
    除いた内側だけを見る、割合は先頭・末尾も含めた全窓に対する比）。
    """
    if windows.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    silent = windows.tolist()
    leading = 0
    for value in silent:
        if not value:
            break
        leading += 1
    if leading == len(silent):
        return float(leading), 0.0, 0.0, 1.0

    trailing = 0
    for value in reversed(silent):
        if not value:
            break
        trailing += 1

    run = longest = internal = 0
    for value in silent[leading : len(silent) - trailing]:
        if value:
            run += 1
            longest = max(longest, run)
            internal += 1
            continue
        run = 0

    return float(leading), float(trailing), float(longest), (internal + leading + trailing) / len(silent)


def measure(path: Path) -> Measurements:
    """生成側 Inspector と同じ方法で音源を測る。

    チャンネルを平均した 1 波形で測るのが要点です。生成側は decodeFrame → meanSample で
    全チャンネルを畳んでから振幅ピークを取るため、`peak_amplitude` はステレオ差のぶん
    必ず各チャンネルのピークより低く出ます（実測: 同じ mp3 で ch 別 0.903 / ch 平均 0.867）。
    別々のものを比べないよう、こちらも同じ畳み方をします。真正ピークとラウドネスだけは
    規格がチャンネルごとに定義されているため、loudness.py（ffmpeg）の値を使います。
    """
    data, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    mono = data.mean(axis=1)

    window = max(int(sample_rate * ANALYSIS_WINDOW_SECONDS), 1)
    full = mono[: mono.size // window * window].reshape(-1, window)
    window_rms = np.sqrt(np.mean(np.square(full), axis=1)) if full.size else np.empty(0)
    # 端数の窓も無音判定へ含める（末尾のフェードアウトが丸ごと落ちるのを防ぐ）。
    remainder = mono[full.size :]
    if remainder.size:
        window_rms = np.append(window_rms, np.sqrt(np.mean(np.square(remainder))))

    leading, trailing, longest, silent_ratio = _silence(window_rms < SILENCE_THRESHOLD_RMS)
    window_seconds = window / sample_rate

    mid = sosfilt(_bandpass_sos(sample_rate, *MID_BAND_HZ), mono)
    high = sosfilt(_bandpass_sos(sample_rate, *HIGH_BAND_HZ), mono)
    loudness = measure_loudness(path)

    return Measurements(
        duration_seconds=mono.size / sample_rate,
        leading_silence_seconds=leading * window_seconds,
        trailing_silence_seconds=trailing * window_seconds,
        longest_internal_silence_seconds=longest * window_seconds,
        silent_ratio=silent_ratio,
        peak_amplitude=float(np.abs(mono).max()) if mono.size else 0.0,
        clipped_sample_ratio=float(np.mean(np.abs(mono) >= CLIPPING_THRESHOLD)) if mono.size else 0.0,
        rms_dbfs=_dbfs(float(np.sqrt(np.mean(np.square(mono))))),
        mid_band_dbfs=_dbfs(float(np.sqrt(np.mean(np.square(mid))))),
        high_band_dbfs=_dbfs(float(np.sqrt(np.mean(np.square(high))))),
        true_peak_dbtp=loudness.true_peak_dbtp,
        integrated_lufs=loudness.integrated_lufs,
    )


def load(path: Path) -> Record:
    data = json.loads(path.read_text(encoding="utf-8"))
    assessment = data.get("assessment") or {}
    quality = assessment.get("quality") or {}
    m = quality.get("measurements") or {}

    return Record(
        job_id=data.get("job_id", ""),
        created_at=data.get("created_at", ""),
        score=float(data.get("score", 0.0)),
        summary=data.get("summary", ""),
        passed=bool(data.get("passed", False)),
        mastered=bool(assessment.get("mastered", False)),
        measurements=Measurements(
            duration_seconds=float(m.get("duration_seconds", 0.0)),
            leading_silence_seconds=float(m.get("leading_silence_seconds", 0.0)),
            trailing_silence_seconds=float(m.get("trailing_silence_seconds", 0.0)),
            longest_internal_silence_seconds=float(m.get("longest_internal_silence_seconds", 0.0)),
            silent_ratio=float(m.get("silent_ratio", 0.0)),
            peak_amplitude=float(m.get("peak_amplitude", 0.0)),
            clipped_sample_ratio=float(m.get("clipped_sample_ratio", 0.0)),
            rms_dbfs=float(m.get("rms_dbfs", 0.0)),
            mid_band_dbfs=float(m.get("mid_band_dbfs", 0.0)),
            high_band_dbfs=float(m.get("high_band_dbfs", 0.0)),
            true_peak_dbtp=float(m.get("true_peak_dbtp", 0.0)),
            integrated_lufs=float(m.get("integrated_lufs", 0.0)),
        ),
        issues=[(i.get("kind", ""), i.get("detail", "")) for i in (quality.get("issues") or [])],
        review=assessment.get("review"),
        lyrics=assessment.get("lyrics"),
    )


@dataclass(frozen=True)
class Comparison:
    label: str
    recorded: float
    measured: float
    unit: str
    # note は行に添える但し書きです。読む人向けで、判定は動かしません。
    note: str = ""
    tolerance: float = 0.0
    # explained は「この差はもう説明が付いている」ことを表します。note と分けているのは、
    # 但し書きが常に出る項目（peak_amplitude の測り方など）で、差の大きさまで見逃さない
    # ようにするためです。説明を付けた時点で警告を止めると、測り方の注記が差の免罪符に
    # なります。
    explained: bool = False

    @property
    def diff(self) -> float:
        return self.measured - self.recorded

    @property
    def unexplained(self) -> bool:
        return not self.explained and abs(self.diff) > self.tolerance


def compare(record: Record, measured: Measurements, sample_rate: int = 44100) -> list[Comparison]:
    """記録と再測を項目ごとに並べる。差の説明が付くものには但し書きを添える。"""
    r, m = record.measurements, measured
    padding = MP3_PADDING_SAMPLES / sample_rate

    padded = abs((r.duration_seconds - m.duration_seconds) - padding) < DURATION_TOLERANCE_SECONDS

    return [
        Comparison(
            "尺", r.duration_seconds, m.duration_seconds, "s",
            f"mp3 の padding {padding * 1000:.0f}ms で説明が付く" if padded else "",
            DURATION_TOLERANCE_SECONDS, explained=padded,
        ),
        # 許容幅は mp3 の符号化によるオーバーシュートぶんです（実測: ch 平均で wav 0.887 /
        # mp3 0.867 の 0.02）。倍を取って、それを超える差は測り方を疑う側へ倒します。
        Comparison(
            "peak_amplitude", r.peak_amplitude, m.peak_amplitude, "",
            "どちらもチャンネル平均の波形で測った値", 0.05,
        ),
        Comparison("true peak", r.true_peak_dbtp, m.true_peak_dbtp, "dBTP", "", LEVEL_TOLERANCE_DB),
        Comparison("integrated", r.integrated_lufs, m.integrated_lufs, "LUFS", "", LEVEL_TOLERANCE_DB),
        Comparison("RMS", r.rms_dbfs, m.rms_dbfs, "dBFS", "", LEVEL_TOLERANCE_DB),
        Comparison("200Hz-4kHz", r.mid_band_dbfs, m.mid_band_dbfs, "dBFS", "", LEVEL_TOLERANCE_DB),
        Comparison("4-10kHz", r.high_band_dbfs, m.high_band_dbfs, "dBFS", "", LEVEL_TOLERANCE_DB),
        Comparison(
            "treble headroom", r.treble_headroom_db, m.treble_headroom_db, "dB", "", LEVEL_TOLERANCE_DB
        ),
        Comparison("曲頭の無音", r.leading_silence_seconds, m.leading_silence_seconds, "s", "", DURATION_TOLERANCE_SECONDS),
        Comparison("曲尾の無音", r.trailing_silence_seconds, m.trailing_silence_seconds, "s", "", DURATION_TOLERANCE_SECONDS),
        Comparison("曲中最長の無音", r.longest_internal_silence_seconds, m.longest_internal_silence_seconds, "s", "", DURATION_TOLERANCE_SECONDS),
        Comparison("無音の割合", r.silent_ratio, m.silent_ratio, "", "", RATIO_TOLERANCE),
        Comparison("クリップ率", r.clipped_sample_ratio, m.clipped_sample_ratio, "", "", RATIO_TOLERANCE),
    ]


def _display_width(text: str) -> int:
    """端末での表示幅。全角は 2 桁を占めます。

    書式指定子の幅は文字数で数えるため、日本語の見出しを混ぜると列がずれます。
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _cell(text: str, width: int, *, right: bool = False) -> str:
    pad = " " * max(0, width - _display_width(text))
    return pad + text if right else text + pad


def render(record: Record, measured: Measurements, audio_name: str) -> str:
    lines = [
        f"\n{record.job_id or '(job_id なし)'}  記録 {record.created_at}",
        f"総合点 {record.score:.2f} / 実測の合否 {'合格' if record.passed else '不合格'}"
        f" / マスタリング {'適用' if record.mastered else '未適用'}",
        f"要約: {record.summary}",
        "",
        _cell("項目", 18) + _cell("生成側(mp3)", 14, right=True)
        + _cell(f"手元({audio_name})", 16, right=True) + _cell("差", 10, right=True),
        "-" * 76,
    ]

    unexplained = []
    for c in compare(record, measured):
        mark = " ⚠" if c.unexplained else ""
        lines.append(
            _cell(c.label, 18)
            + f"{c.recorded:>14.3f}{c.measured:>16.3f}{c.diff:>+10.3f}{mark}"
            + (f"   {c.note}" if c.note else "")
        )
        if c.unexplained:
            unexplained.append(c)

    lines.append("-" * 76)
    if record.issues:
        lines.append("生成側が挙げた指摘:")
        lines.extend(f"  [{kind}] {detail}" for kind, detail in record.issues)

    if record.review:
        r = record.review
        lines.append(
            f"講評: 歌詞 {r.get('lyrics_intelligibility', 0)}/5 / 構成 {r.get('structure_match', 0)}/5"
            f" / 進行 {r.get('pacing', 0)}/5 / ボーカル {r.get('vocal_balance', 0)}/5"
        )
        lines.extend(f"  - {f}" for f in (r.get("findings") or []))

    lyrics = record.lyrics
    if lyrics and (lyrics.get("missing_lines") or lyrics.get("extra_lines")):
        lines.append(
            f"譜面の行数: 不足 {lyrics.get('missing_lines', 0)} / 余剰 {lyrics.get('extra_lines', 0)}"
            "  ※ モデルが返した譜面の比較で、鳴った音は見ていません(check_lyrics.py の担当)"
        )

    lines.append("")
    if unexplained:
        lines.append(f"⚠ 説明の付かない差が {len(unexplained)} 件あります:")
        lines.extend(f"   {c.label} {c.diff:+.3f}{c.unit}" for c in unexplained)
        lines.append("  mp3 と wav の差は実測で 0.1dB 台です。それを超える差は測り方か対象を疑ってください。")
    else:
        lines.append("✓ 記録と手元の測定は、mp3 と wav の差で説明が付く範囲に収まっています。")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="測る音源(master.wav 推奨)")
    parser.add_argument("assessment", type=Path, help="生成側の assessment.json")
    args = parser.parse_args()

    record = load(args.assessment)
    print(render(record, measure(args.audio), args.audio.suffix.lstrip(".") or "手元"))


if __name__ == "__main__":
    main()
