"""Check with ASR whether each lyric line was actually sung.

Lyria drops lines when the amount of lyrics and the length do not fit. Transcribing and comparing with the
lyrics shows which lines vanished. How lines are dropped depends on the model: lyria-3.5 can also sing one
hook twice and drop another line (seen on Chorus 1 of a song generated on 09-04).

The generator also compares lyrics, but it compares **the number of lines in the score the model returned**,
not the audio (stated in the generator's `domain/lyric_fidelity.go`). Only transcribing the actual audio
catches lines that were not sung as scored.

Keep expectations low. Recognition of Japanese singing over loud backing is not accurate, so this is a rough
"did the line survive" check, not a calibration tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pykakasi

from .recipe import Recipe, Section
from .vocals import FRAME_FLOOR_DB, FRAME_SECONDS, frame_levels, sung_reference

MODEL = "large-v3"

# The sample rate whisper uses internally; clips are cut at this rate too.
SAMPLE_RATE = 16000

# Padding kept around the voice, enough not to cut consonant onsets or trailing endings.
CLIP_PAD_SECONDS = 0.2

# Silence longer than this is treated by whisper as a sign of hallucination (see transcribe).
HALLUCINATION_SILENCE_SECONDS = 2.0

# A line counts as sung when at least this share of its reading is found in the transcript (counted as in
# _coverage). Recognition is noisy, so a stricter value would report every line as dropped.
COVERAGE_THRESHOLD = 0.45

_kks = pykakasi.kakasi()

# Japanese ASR does not reliably hear English mixed into Japanese singing: "Split the role" came out as
# "Speed the road" every time. A sung line would be reported as a mismatch, so only the Japanese part is compared.
_LATIN = re.compile(r"[A-Za-z0-9'’\-]+")


def to_kana(text: str) -> str:
    """Normalise to the reading (kana) for comparison, so kanji vs kana spelling does not cause a mismatch.

    Latin letters and digits are dropped: Japanese ASR cannot transcribe English correctly, and keeping them
    would make sung lines mismatch.
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
        """A line without Japanese cannot be checked this way."""
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


def trim_to_voice(
    audio: np.ndarray,
    floor_dbfs: float,
    sample_rate: int = SAMPLE_RATE,
    start: int = 0,
    end: int | None = None,
) -> np.ndarray:
    """Trim audio[start:end] to where the voice is.

    The point is to return nothing for a section whose voice never crosses the floor. Nothing was sung there,
    yet whisper hallucinates when given it (an outro with only separation leakage came back as
    「ありがとうございました。」). A section that lost all its lines looks exactly like this, so filling it with
    hallucinations would defeat the check.

    Whether there is voice is judged inside the section only; only the padding reaches outside it. If the
    padding stopped at the section edge, a section whose voice starts right at its head would get zero padding
    and whisper would flatten it into a hallucination (a 120 BPM Verse at 16-40 s became the single sentence
    「ご視聴ありがとうございました。」 and all six lines were reported dropped; widening by 0.1 s brought every
    line back). The judgement itself does not reach outside, so a section that lost everything does not count
    as "voiced" by catching the neighbour's held ending.

    Trimming the edges of a voiced section does not, by itself, stop hallucinations (see transcribe).
    """
    end = len(audio) if end is None else end
    levels = frame_levels(audio[start:end], sample_rate)
    voiced = np.flatnonzero(levels > floor_dbfs)
    if voiced.size == 0:
        return audio[:0]

    frame = int(FRAME_SECONDS * sample_rate)
    pad = int(CLIP_PAD_SECONDS * sample_rate)
    lo = max(0, start + int(voiced[0]) * frame - pad)
    hi = min(len(audio), start + (int(voiced[-1]) + 1) * frame + pad)
    return audio[lo:hi]


def transcribe(
    vocals_wav: Path,
    recipe: Recipe,
    device: str = "cpu",
    model_name: str = MODEL,
) -> list[tuple[Section, str]]:
    """Transcribe the vocal stem section by section. With the backing removed, recognition improves.

    Never pass the expected lyrics as initial_prompt. Recognition improves, but the model is pulled towards the
    given lyrics and transcribes even the dropped lines: it would be handing over the answer to what is being checked.

    Never pass the whole stem at once. whisper works in 30-second windows, and an instrumental opening that
    fills a window is taken as silence and hallucinated over. On 「降ってこい」, 0-30 s collapsed into one
    segment 「ご視聴ありがとうございました」 and the six sung lines of the Verse were reported dropped. Cutting
    per section brings every line back from the same audio. It also maps transcripts to sections by the cut
    itself rather than by time overlap, so nothing is lost near boundaries.

    Hallucinations survive per-section cuts too: the last partial window is padded with silence internally,
    and the same thing happens there (a Chorus 2 whose singing ends at 161 s ended its transcript with
    「ご視聴ありがとうございました」 and its last line was reported dropped). Only
    `hallucination_silence_threshold` stopped it. Trimming trailing silence, `vad_filter=True` and a lower
    `no_speech_threshold` all failed. The parameter only works with `word_timestamps=True`, which is slower,
    but that beats judging dropped lines with hallucinations mixed in.
    """
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio

    audio = decode_audio(str(vocals_wav), sampling_rate=SAMPLE_RATE)
    model = WhisperModel(model_name, device=device, compute_type="int8")

    # Silence is judged with the same yardstick as vocals.py (relative to the sung sections).
    floor = sung_reference(audio, SAMPLE_RATE, recipe) - FRAME_FLOOR_DB

    results = []
    for section in recipe.sections:
        if not recipe.lyrics.get(section.name):
            continue

        clip = trim_to_voice(
            audio, floor, start=int(section.start * SAMPLE_RATE), end=int(section.end * SAMPLE_RATE)
        )
        if clip.size == 0:  # the audio is shorter than the recipe, or there is no voice
            results.append((section, ""))
            continue

        # Long sections are split into several windows internally. Carrying the previous window's text over
        # would propagate a bad recognition to everything after it, so it is turned off.
        segments, _ = model.transcribe(
            clip,
            language="ja",
            vad_filter=False,
            condition_on_previous_text=False,
            word_timestamps=True,
            hallucination_silence_threshold=HALLUCINATION_SILENCE_SECONDS,
        )
        results.append((section, " ".join(s.text.strip() for s in segments)))
    return results


def _coverage(line_kana: str, heard_kana: str) -> float:
    """How much of the expected line's reading appears in the transcript.

    It sums the matching blocks rather than taking the longest run: when 「境界をここに引け」 was heard as
    「妖怪よ ここに行け」, small errors at both ends broke the run and a sung line fell to 27 %.
    """
    if not line_kana:
        return 1.0
    matcher = SequenceMatcher(None, heard_kana, line_kana, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return min(matched / len(line_kana), 1.0)


def check(transcripts: list[tuple[Section, str]], recipe: Recipe) -> list[SectionCheck]:
    """Compare each section's transcript with the lines that should be sung there."""
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
                lines.append(f"     ―――  {lc.line}  (English only, not checked)")
                continue
            mark = "  " if lc.sung else "❓"
            lines.append(f"  {mark} {lc.coverage:>4.0%}  {lc.line}")
        lines.append(f"     transcript: {c.transcript[:110] or '(none)'}")

    missing = [(c, lc) for c in checks for lc in c.missing]
    lines.append("\n" + "-" * 70)
    if missing:
        lines.append(f"❓ {len(missing)} line(s) did not match:")
        for c, lc in missing:
            lines.append(f"   [{c.section.name}] {lc.line}  ({lc.coverage:.0%})")
        lines.append(
            "\n* This measure alone cannot tell recognition errors from lines that were really dropped."
            "\n  Listen to those passages to confirm."
        )
    else:
        lines.append("✓ every line was found in the transcript.")
    return "\n".join(lines)
