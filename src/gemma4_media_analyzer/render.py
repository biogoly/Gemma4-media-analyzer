"""Deterministic final report rendering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .text_utils import text_equivalent


def format_timestamp(seconds: float) -> str:
    milliseconds = max(0, round(float(seconds) * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


def _text(value: Any, fallback: str = "") -> str:
    return value.strip() if isinstance(value, str) and value.strip() else fallback


def _join(values: Any) -> str:
    if not isinstance(values, list):
        return ""
    return ", ".join(str(value).strip() for value in values if str(value).strip())


def render_markdown(document: dict[str, Any]) -> str:
    """Concatenate all chunk results into one chronological, readable script."""

    media_type = _text(document.get("media_type"), "media").title()
    lines = [
        f"# Gemma 4 {media_type} Analysis",
        "",
        f"- Source: `{document['source']}`",
        f"- Media type: {media_type}",
        f"- Duration: {format_timestamp(document['duration_seconds'])}",
        f"- Backend: `{document['backend']}`",
        f"- Model: `{document['model']}`",
        "",
        "## Chronological script",
        "",
    ]

    for chunk in document.get("chunks", []):
        start = float(chunk["start_seconds"])
        end = start + float(chunk["duration_seconds"])
        audio = chunk.get("audio") or {}
        visual = chunk.get("visual") or {}
        lines.extend([f"### {format_timestamp(start)}–{format_timestamp(end)}", ""])

        summary = _text(visual.get("summary"))
        setting = _text(visual.get("setting"))
        if summary:
            lines.append(f"**Scene:** {summary}")
        if setting:
            lines.append(f"**Setting:** {setting}")
        people = _join(visual.get("people"))
        objects = _join(visual.get("objects"))
        if people:
            lines.append(f"**People:** {people}")
        if objects:
            lines.append(f"**Objects:** {objects}")
        if summary or setting or people or objects:
            lines.append("")

        events: list[tuple[float, int, str]] = []
        for shot in visual.get("shots") or []:
            shot_start = float(shot.get("start", start))
            description = _text(shot.get("description"), "Visual event")
            visible_text = _join(shot.get("visible_text"))
            suffix = f" Visible text: {visible_text}" if visible_text else ""
            events.append((shot_start, 0, f"**Visual:** {description}{suffix}"))
        segments = [
            segment
            for segment in audio.get("segments") or []
            if isinstance(segment, dict) and _text(segment.get("text"))
        ]
        canonical_transcript = _text(audio.get("transcript"))
        segment_transcript = " ".join(_text(segment.get("text")) for segment in segments)
        segment_coverage_complete = not canonical_transcript or text_equivalent(
            canonical_transcript,
            segment_transcript,
        )
        if segment_coverage_complete:
            for segment in segments:
                segment_start = float(segment.get("start", start))
                speaker = _text(segment.get("speaker"), "Speaker")
                transcript = _text(segment.get("text"))
                translation = _text(segment.get("translation"))
                body = f"**{speaker}:** {transcript}"
                if translation:
                    body += f"  \n*Translation:* {translation}"
                events.append((segment_start, 1, body))
        elif segments and canonical_transcript:
            # Completeness is more important than fine timing when Gemma's segment array
            # disagrees with its own canonical transcript.
            first = segments[0]
            segment_start = float(first.get("start", start))
            speaker = _text(first.get("speaker"), "Speaker")
            body = f"**{speaker}:** {canonical_transcript}"
            translation = _text(audio.get("translation"))
            if translation:
                body += f"  \n*Translation:* {translation}"
            events.append((segment_start, 1, body))

        for event_start, _order, body in sorted(events, key=lambda event: (event[0], event[1])):
            lines.extend([f"`{format_timestamp(event_start)}` {body}", ""])

        if not segments:
            transcript = canonical_transcript
            translation = _text(audio.get("translation"))
            if transcript:
                lines.extend([f"**Transcript:** {transcript}", ""])
            if translation:
                lines.extend([f"**Translation:** {translation}", ""])
        sounds = _join(audio.get("non_speech_audio"))
        if sounds:
            lines.extend([f"**Non-speech audio:** {sounds}", ""])

    return "\n".join(lines).rstrip() + "\n"


def render_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(f"{path.suffix}.partial")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
