"""歌詞の行が実際に歌われたかを ASR で確かめる。

Lyria は奇数行のセクションで1行落とすことがあり、今それを見ているのは
「内部の無音が長いか」という間接的な指標だけです。書き起こして突き合わせれば
どの行が消えたのかまで分かります。

期待値は低めに見てください。大音量の伴奏を伴う日本語歌唱の認識精度は出ないため、
これは校正ツールではなく「行が生き残ったか」の粗い検査です。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pykakasi

from .recipe import Recipe, Section

MODEL = "large-v3"

# whisper が内部で使うサンプリングレート。切り出しの単位もこれに揃えます。
SAMPLE_RATE = 16000

# 期待した行の読みが、書き起こしの中にこの割合以上ひと続きで現れれば「歌われた」と見なす。
# 認識が荒れるため厳しくすると全行が落ちたことになります。
COVERAGE_THRESHOLD = 0.45

_kks = pykakasi.kakasi()

# 日本語歌唱に混ざった英語は日本語 ASR が安定して聞き取れません。実測では
# "Split the role" が一貫して "Speed the road" になりました。行は歌われている
# のに不一致として上がるため、比較からは日本語部分だけを取り出します。
_LATIN = re.compile(r"[A-Za-z0-9'’\-]+")


def to_kana(text: str) -> str:
    """比較のために読みへ正規化する。漢字と仮名の表記差で不一致になるのを防ぐ。

    英数字は落とします。日本語 ASR が英語を正しく書き起こせないため、残すと
    歌われている行まで不一致になります。
    """
    japanese_only = _LATIN.sub(" ", text)
    kana = "".join(part["kana"] for part in _kks.convert(japanese_only))
    return "".join(kana.split())


@dataclass
class LineCheck:
    line: str
    coverage: float

    @property
    def verifiable(self) -> bool:
        """日本語を含まない行は、この方法では歌われたか確かめられない。"""
        return bool(to_kana(self.line))

    @property
    def sung(self) -> bool:
        return not self.verifiable or self.coverage >= COVERAGE_THRESHOLD


@dataclass
class SectionCheck:
    section: Section
    transcript: str
    lines: list[LineCheck]

    @property
    def missing(self) -> list[LineCheck]:
        return [l for l in self.lines if not l.sung]


def transcribe(
    vocals_wav: Path,
    recipe: Recipe,
    device: str = "cpu",
    model_name: str = MODEL,
) -> list[tuple[Section, str]]:
    """ボーカル stem をセクションごとに切って書き起こす。伴奏を除いてあるぶん認識率が上がる。

    期待した歌詞を initial_prompt に渡してはいけません。認識率は上がりますが、
    モデルが与えた歌詞へ引きずられ、落ちた行まで書き起こされてしまいます。
    検証したい対象を答えとして渡すことになるためです。

    stem 全体を一度に渡してはいけません。whisper は 30 秒窓で処理するため、
    冒頭のインストが窓に丸ごと入ると無音とみなして幻聴を返します。実測では
    「降ってこい」の 0-30s が「ご視聴ありがとうございました」1セグメントに潰れ、
    歌われている Verse の6行が落ちたと報告されました。区間ごとに切れば同じ音源で
    全行が出ます。ついでに書き起こしとセクションの対応が時刻の重なり判定でなく
    切り出しそのもので決まるため、境界付近の取りこぼしもなくなります。
    """
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(vocals_wav), sampling_rate=SAMPLE_RATE)
    model = WhisperModel(model_name, device=device, compute_type="int8")

    results = []
    for section in recipe.sections:
        if not recipe.lyrics.get(section.name):
            continue

        clip = audio[int(section.start * SAMPLE_RATE):int(section.end * SAMPLE_RATE)]
        if clip.size == 0:  # 音源がレシピの宣言尺より短い場合
            results.append((section, ""))
            continue

        # 長いセクションは内部で複数の窓に分かれます。前の窓の結果を引き継がせると、
        # 一度荒れた認識がそのまま後ろへ伝播するため切ります。
        segments, _ = model.transcribe(
            clip,
            language="ja",
            vad_filter=False,
            condition_on_previous_text=False,
        )
        results.append((section, " ".join(s.text.strip() for s in segments)))
    return results


def _coverage(line_kana: str, heard_kana: str) -> float:
    """期待した行の読みが、書き起こしにどれだけ現れているか。

    最長の連続一致ではなく一致ブロックの合計を使います。「境界をここに引け」が
    「妖怪よ ここに行け」と聞き取られた場合、両端の小さな誤りで連続一致が分断され、
    実際には歌われているのに 27% まで落ちてしまうためです。
    """
    if not line_kana:
        return 1.0
    matcher = SequenceMatcher(None, heard_kana, line_kana, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return min(matched / len(line_kana), 1.0)


def check(transcripts: list[tuple[Section, str]], recipe: Recipe) -> list[SectionCheck]:
    """セクションごとの書き起こしと、そこで歌われるはずの行を突き合わせる。"""
    results = []
    for section, heard in transcripts:
        expected = recipe.lyrics.get(section.name, [])
        if not expected:
            continue

        heard_kana = to_kana(heard)
        results.append(
            SectionCheck(
                section=section,
                transcript=heard,
                lines=[LineCheck(line, _coverage(to_kana(line), heard_kana)) for line in expected],
            )
        )
    return results


def render(checks: list[SectionCheck]) -> str:
    lines = []
    for c in checks:
        lines.append(f"\n[{c.section.name}]  {c.section.start:.0f}-{c.section.end:.0f}s")
        for lc in c.lines:
            if not lc.verifiable:
                lines.append(f"     ―――  {lc.line}  (英語のみ・判定対象外)")
                continue
            mark = "  " if lc.sung else "❓"
            lines.append(f"  {mark} {lc.coverage:>4.0%}  {lc.line}")
        lines.append(f"     書き起こし: {c.transcript[:110] or '(なし)'}")

    missing = [(c, lc) for c in checks for lc in c.missing]
    lines.append("\n" + "-" * 70)
    if missing:
        lines.append(f"❓ 一致しなかった行が {len(missing)} 件あります:")
        for c, lc in missing:
            lines.append(f"   [{c.section.name}] {lc.line}  ({lc.coverage:.0%})")
        lines.append(
            "\n※ 認識精度の問題と実際の行落ちは、この指標だけでは区別できません。"
            "\n  該当箇所を実際に聴いて確かめてください。"
        )
    else:
        lines.append("✓ すべての行が書き起こしの中に見つかりました。")
    return "\n".join(lines)
