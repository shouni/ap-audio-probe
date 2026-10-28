"""端末に表を出すときの桁揃え。

書式指定子の幅は文字数で数えるため、日本語の見出しを混ぜると列がずれます。
"""

from __future__ import annotations

import unicodedata


def display_width(text: str) -> int:
    """端末での表示幅。全角は 2 桁を占めます。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def cell(text: str, width: int, *, right: bool = False) -> str:
    pad = " " * max(0, width - display_width(text))
    return pad + text if right else text + pad
