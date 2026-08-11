"""ap-comp のレシピ JSON からセクション構成を読む。"""

from __future__ import annotations

import json
from dataclasses import dataclass
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
    def duration(self) -> float:
        return self.end - self.start

    @property
    def instrumental(self) -> bool:
        """レシピ自身が「楽器のみ」と指示しているセクションか。"""
        return INSTRUMENTAL_MARKER in self.prompt


@dataclass(frozen=True)
class Recipe:
    title: str
    tempo: int
    sections: list[Section]

    @property
    def declared_duration(self) -> float:
        return max(s.end for s in self.sections) if self.sections else 0.0


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
    return Recipe(title=data["title"], tempo=int(data["tempo"]), sections=sections)
