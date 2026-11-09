"""Read the section layout from a recipe JSON."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# A section marked instrumental always starts its prompt with this phrase (the generator's prompt convention).
INSTRUMENTAL_MARKER = "Instruments only"


@dataclass(frozen=True)
class Section:
    name: str
    start: float
    end: float
    prompt: str

    @property
    def instrumental(self) -> bool:
        """Whether the recipe itself says this section is instruments only."""
        return INSTRUMENTAL_MARKER in self.prompt


@dataclass(frozen=True)
class Recipe:
    title: str
    tempo: int
    sections: list[Section]
    # lyrics maps a section name to the lines that should be sung there. Empty for instrumentals.
    lyrics: dict[str, list[str]] = field(default_factory=dict)
    # compose_mode is the generator's composing mode (edm_pop, sparkle_rock, ...). Cross-song comparisons
    # use it to line up songs of the same mode. Older recipes lack it, and then it is empty.
    compose_mode: str = ""

    @property
    def declared_duration(self) -> float:
        return max(s.end for s in self.sections) if self.sections else 0.0


def parse_lyrics(text: str) -> dict[str, list[str]]:
    """Split lyrics separated by tag lines such as `[Verse]` into lines per section name.

    Tags are assumed to match sections[].name exactly (the generator's convention).
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
        compose_mode=data.get("compose_mode", ""),
    )
