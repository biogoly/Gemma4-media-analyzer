"""Shared token helpers for transcript validation, reconciliation, and rendering."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextToken:
    start: int
    end: int
    normalized: str


def text_tokens(text: Any) -> list[TextToken]:
    if not isinstance(text, str):
        return []
    tokens: list[TextToken] = []
    for match in re.finditer(r"\S+", text):
        normalized = re.sub(r"[^\w']+", "", match.group().casefold()).strip("_'")
        if normalized:
            tokens.append(TextToken(match.start(), match.end(), normalized))
    return tokens


def normalized_words(text: Any) -> list[str]:
    return [token.normalized for token in text_tokens(text)]


def text_equivalent(left: Any, right: Any) -> bool:
    return normalized_words(left) == normalized_words(right)
