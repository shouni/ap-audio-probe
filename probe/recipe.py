"""ap-comp のレシピ JSON からセクション構成を読む。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ap-comp のインスト指定セクションは、プロンプト冒頭に必ずこの語が入る。
INSTRUMENTAL_MARKER = "Instruments only"


@dataclass(frozen=True)
class Section:
    name: str
    start: float
    end: float
    prompt: str

    @property
    def instrumental(self) -> bool:
        """レシピ自身が「楽器のみ」と指示しているセクションか。"""
        return INSTRUMENTAL_MARKER in self.prompt


@dataclass(frozen=True)
class Recipe:
    title: str
    tempo: int
    sections: list[Section]
    # lyrics はセクション名から、そこで歌われるはずの行を引くための対応表です。
    # インスト曲では空になります。
    lyrics: dict[str, list[str]] = field(default_factory=dict)

    @property
    def declared_duration(self) -> float:
        return max(s.end for s in self.sections) if self.sections else 0.0


def parse_lyrics(text: str) -> dict[str, list[str]]:
    """`[Verse]` のようなタグ行で区切られた歌詞を、セクション名ごとの行に割る。

    タグは sections[].name と完全一致する前提です(ap-comp 側の取り決め)。
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1]
            sections.setdefault(current, [])
        elif current is not None:
            sections[current].append(line)
    return sections


def load(path: Path) -> Recipe:
    data = json.loads(path.read_text(encoding="utf-8"))
    sections = [
        Section(
            name=s["name"],
            start=float(s["start_seconds"]),
            end=float(s["end_seconds"]),
            prompt=s["prompt"],
        )
        for s in data["sections"]
    ]
    sections.sort(key=lambda s: s.start)

    lyrics_block = (data.get("lyrics") or {}).get("lyrics", "")
    return Recipe(
        title=data["title"],
        tempo=int(data["tempo"]),
        sections=sections,
        lyrics=parse_lyrics(lyrics_block),
    )
