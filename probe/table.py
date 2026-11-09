"""Column alignment for tables printed to the terminal.

Format-spec widths count characters, so columns shift when a heading contains wide (e.g. Japanese) characters.
"""

from __future__ import annotations

import unicodedata


def display_width(text: str) -> int:
    """Display width in the terminal. Full-width characters take two columns."""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def cell(text: str, width: int, *, right: bool = False) -> str:
    pad = " " * max(0, width - display_width(text))
    return pad + text if right else text + pad
