"""Deterministic, bounded context carried between visual chunks."""

from __future__ import annotations

from typing import Any

VISUAL_CONTEXT_VERSION = "grounded-continuity-v1"
MAX_CONTINUITY_CHARS = 1600


def _compact(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    shortened = text[: limit - 1].rsplit(" ", 1)[0].rstrip(" ,;:-")
    return f"{shortened or text[: limit - 1]}…"


def _audio_text(chunk: dict[str, Any]) -> str:
    audio = chunk.get("audio") or {}
    transcript = _compact(audio.get("transcript"), 1200)
    translation = _compact(audio.get("translation"), 1200)
    if translation and translation.casefold() != transcript.casefold():
        return f"{transcript} Translation: {translation}" if transcript else translation
    return transcript


def _deduplicated_items(chunks: list[dict[str, Any]], name: str, limit: int) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        values = (chunk.get("visual") or {}).get(name) or []
        if not isinstance(values, list):
            continue
        for value in values:
            text = _compact(value, 60)
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            items.append(text)
            if len(items) == limit:
                return items
    return items


def build_visual_continuity(previous_chunks: list[dict[str, Any]]) -> str:
    """Build a small evidence record without another model invocation.

    Spoken context is carried as evidence. Visual conclusions remain explicitly
    provisional so one mistaken classification does not become an established fact.
    """

    if not previous_chunks:
        return ""

    sections: list[str] = []
    opening_audio = ""
    for chunk in previous_chunks:
        opening_audio = _audio_text(chunk)
        if opening_audio:
            break
    if opening_audio:
        sections.append(
            "Opening spoken context (background only; it may no longer apply): "
            + _compact(opening_audio, 240)
        )

    recent_audio = [_audio_text(chunk) for chunk in previous_chunks[-2:]]
    recent_audio = [text for text in recent_audio if text and text != opening_audio]
    if recent_audio:
        sections.append("Recent spoken context: " + _compact(" ".join(recent_audio), 360))

    previous_visual = previous_chunks[-1].get("visual") or {}
    summary = _compact(previous_visual.get("summary"), 180)
    if summary:
        sections.append("Previous visual summary (provisional): " + summary)

    shots = previous_visual.get("shots") or []
    if isinstance(shots, list):
        final_shot = next((shot for shot in reversed(shots) if isinstance(shot, dict)), None)
        if final_shot:
            description = _compact(final_shot.get("description"), 220)
            if description:
                sections.append("Last observed action (provisional): " + description)

    objects = _deduplicated_items(previous_chunks[-3:], "objects", 5)
    if objects:
        sections.append("Recently observed objects (provisional): " + "; ".join(objects))

    setting = _compact(previous_visual.get("setting"), 100)
    if setting:
        sections.append("Previous setting (provisional): " + setting)

    return _compact("\n".join(sections), MAX_CONTINUITY_CHARS)
