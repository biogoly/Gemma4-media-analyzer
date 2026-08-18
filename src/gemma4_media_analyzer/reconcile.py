"""Deterministic reconciliation for overlapping media-analysis windows."""

from __future__ import annotations

import copy
from difflib import SequenceMatcher
from typing import Any

from .text_utils import normalized_words, text_equivalent, text_tokens

_tokens = text_tokens


def find_boundary_overlap(
    previous: str,
    current: str,
    *,
    max_tokens: int = 80,
) -> tuple[int, int, float]:
    """Return matching suffix/prefix token counts and a deterministic similarity score."""

    previous_tokens = _tokens(previous)[-max_tokens:]
    current_tokens = _tokens(current)[:max_tokens]
    best = (0, 0, 0.0)
    best_key = (0.0, 0)

    for previous_count in range(1, len(previous_tokens) + 1):
        suffix = previous_tokens[-previous_count:]
        previous_text = " ".join(token.normalized for token in suffix)
        lower = max(1, previous_count - 2)
        upper = min(len(current_tokens), previous_count + 2)
        for current_count in range(lower, upper + 1):
            prefix = current_tokens[:current_count]
            current_text = " ".join(token.normalized for token in prefix)
            ratio = SequenceMatcher(None, previous_text, current_text, autojunk=False).ratio()
            minimum_count = min(previous_count, current_count)
            if minimum_count == 1:
                long_single_token = max(len(previous_text), len(current_text)) >= 5
                accepted = long_single_token and ratio >= 0.88
            else:
                accepted = ratio >= 0.82
            key = (round(ratio, 6), minimum_count)
            if accepted and key > best_key:
                best = (previous_count, current_count, ratio)
                best_key = key
    return best


def _trim_suffix(text: Any, token_count: int) -> str | None:
    if text is None:
        return None
    value = str(text)
    tokens = _tokens(value)
    if token_count <= 0:
        return value
    if token_count >= len(tokens):
        return ""
    return value[: tokens[-token_count].start].rstrip()


def _trim_prefix(text: Any, token_count: int) -> str | None:
    if text is None:
        return None
    value = str(text)
    tokens = _tokens(value)
    if token_count <= 0:
        return value
    if token_count >= len(tokens):
        return ""
    return value[tokens[token_count].start :].lstrip()


def _segments_cover_transcript(audio: dict[str, Any]) -> bool:
    segments = audio.get("segments")
    if not isinstance(segments, list):
        return False
    joined = " ".join(
        str(segment.get("text") or "").strip()
        for segment in segments
        if isinstance(segment, dict) and str(segment.get("text") or "").strip()
    )
    return text_equivalent(audio.get("transcript"), joined)


def _mark_reconciled_coverage(audio: dict[str, Any], complete: bool) -> None:
    coverage = dict(audio.get("segment_coverage") or {})
    coverage.update(
        {
            "status": "reconciled" if complete else "canonical-fallback",
            "complete": complete,
        }
    )
    audio["segment_coverage"] = coverage


def _trim_audio_suffix(audio: dict[str, Any], token_count: int) -> None:
    if token_count <= 0:
        return
    complete = _segments_cover_transcript(audio)
    audio["transcript"] = _trim_suffix(audio.get("transcript"), token_count) or ""
    segments = audio.get("segments")
    if not isinstance(segments, list):
        return
    if not complete:
        _mark_reconciled_coverage(audio, False)
        return

    keep = len(_tokens(audio["transcript"]))
    retained: list[dict[str, Any]] = []
    for segment in segments:
        if keep <= 0 or not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "")
        segment_tokens = _tokens(text)
        if not segment_tokens:
            continue
        if keep >= len(segment_tokens):
            retained.append(segment)
            keep -= len(segment_tokens)
            continue

        segment["text"] = _trim_suffix(text, len(segment_tokens) - keep) or ""
        start = float(segment.get("start", 0.0))
        end = max(start, float(segment.get("end", start)))
        segment["end"] = start + ((end - start) * keep / len(segment_tokens))
        # A partial source segment cannot be mapped safely onto its chunk-level translation.
        segment["translation"] = None
        retained.append(segment)
        keep = 0

    audio["segments"] = retained
    _mark_reconciled_coverage(audio, True)


def _trim_audio_prefix(audio: dict[str, Any], token_count: int) -> None:
    if token_count <= 0:
        return
    complete = _segments_cover_transcript(audio)
    audio["transcript"] = _trim_prefix(audio.get("transcript"), token_count) or ""
    segments = audio.get("segments")
    if not isinstance(segments, list):
        return
    if not complete:
        _mark_reconciled_coverage(audio, False)
        return

    remaining = token_count
    retained: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = str(segment.get("text") or "")
        segment_tokens = _tokens(text)
        if not segment_tokens:
            continue
        if remaining >= len(segment_tokens):
            remaining -= len(segment_tokens)
            continue
        if remaining:
            kept_count = len(segment_tokens) - remaining
            segment["text"] = _trim_prefix(text, remaining) or ""
            start = float(segment.get("start", 0.0))
            end = max(start, float(segment.get("end", start)))
            segment["start"] = end - ((end - start) * kept_count / len(segment_tokens))
            segment["translation"] = None
            remaining = 0
        retained.append(segment)

    audio["segments"] = retained
    _mark_reconciled_coverage(audio, True)


def _boundary_text(text: str, token_count: int, *, prefix: bool) -> str:
    tokens = _tokens(text)
    if not tokens or token_count <= 0:
        return ""
    if prefix:
        end = tokens[min(token_count, len(tokens)) - 1].end
        return text[:end].strip()
    start = tokens[-min(token_count, len(tokens))].start
    return text[start:].strip()


def _ends_sentence(text: str) -> bool:
    return text.rstrip().rstrip('"\')]}').endswith((".", "!", "?"))


def _overlap_source(
    previous: str,
    current: str,
    previous_count: int,
    current_count: int,
) -> str:
    comparison_count = max(previous_count, current_count)
    previous_match = _boundary_text(previous, comparison_count, prefix=False)
    current_match = _boundary_text(current, comparison_count, prefix=True)
    previous_words = normalized_words(previous_match)
    current_words = normalized_words(current_match)
    if previous_words == current_words:
        return "current"
    if previous_words and current_words:
        previous_last = previous_words[-1]
        current_last = current_words[-1]
        if current_last.startswith(previous_last) and len(current_last) > len(previous_last):
            return "current"
    if _ends_sentence(current_match) and not _ends_sentence(previous_match):
        return "current"
    return "previous"


def _join_segment_field(segments: list[dict[str, Any]], field: str) -> str:
    return " ".join(
        str(segment.get(field) or "").strip()
        for segment in segments
        if str(segment.get(field) or "").strip()
    )


def _partition_visual_by_time(
    previous: dict[str, Any],
    current: dict[str, Any],
    seam_seconds: float,
) -> None:
    previous_shots = previous.get("shots")
    current_shots = current.get("shots")
    if not isinstance(previous_shots, list) or not isinstance(current_shots, list):
        return

    def midpoint(shot: dict[str, Any]) -> float:
        start = float(shot.get("start", seam_seconds))
        end = float(shot.get("end", start))
        return (start + end) / 2

    previous["shots"] = [
        shot for shot in previous_shots if isinstance(shot, dict) and midpoint(shot) < seam_seconds
    ]
    current["shots"] = [
        shot for shot in current_shots if isinstance(shot, dict) and midpoint(shot) >= seam_seconds
    ]


def _retime_audio_after(audio: dict[str, Any], lower_bound: float, chunk_end: float) -> bool:
    """Compress a chunk timeline forward when its retained prefix crosses the prior chunk."""

    segments = [
        segment for segment in audio.get("segments") or [] if isinstance(segment, dict)
    ]
    if not segments:
        return False
    first_start = float(segments[0].get("start", lower_bound))
    if first_start >= lower_bound:
        return False

    source_span = max(0.0, chunk_end - first_start)
    target_span = max(0.0, chunk_end - lower_bound)
    scale = target_span / source_span if source_span > 0 else 0.0
    previous_start = lower_bound
    for segment in segments:
        start = float(segment.get("start", first_start))
        end = max(start, float(segment.get("end", start)))
        adjusted_start = lower_bound + (max(0.0, start - first_start) * scale)
        adjusted_end = lower_bound + (max(0.0, end - first_start) * scale)
        adjusted_start = min(chunk_end, max(previous_start, adjusted_start))
        adjusted_end = min(chunk_end, max(adjusted_start, adjusted_end))
        segment["start"] = adjusted_start
        segment["end"] = adjusted_end
        previous_start = adjusted_start
    return True


def _enforce_audio_chronology(chunks: list[dict[str, Any]]) -> None:
    last_start: float | None = None
    for chunk in chunks:
        audio = chunk.get("audio") or {}
        segments = [
            segment for segment in audio.get("segments") or [] if isinstance(segment, dict)
        ]
        if not segments:
            continue
        if last_start is not None:
            chunk_end = float(chunk["start_seconds"]) + float(chunk["duration_seconds"])
            lower_bound = min(chunk_end, last_start + 0.001)
            if _retime_audio_after(audio, lower_bound, chunk_end):
                audio["timestamp_reconciliation"] = {
                    "strategy": "monotonic-overlap",
                    "minimum_start": lower_bound,
                }
        last_start = float(segments[-1].get("start", last_start or 0.0))


def reconcile_overlapping_chunks(
    chunks: list[dict[str, Any]],
    overlap_seconds: float,
) -> list[dict[str, Any]]:
    """Remove duplicated boundary content while preserving cached raw model results."""

    reconciled = copy.deepcopy(chunks)
    if overlap_seconds <= 0:
        return reconciled

    for index in range(len(reconciled) - 1):
        previous_chunk = reconciled[index]
        current_chunk = reconciled[index + 1]
        previous_audio = previous_chunk.get("audio") or {}
        current_audio = current_chunk.get("audio") or {}
        previous_text = str(previous_audio.get("transcript") or "")
        current_text = str(current_audio.get("transcript") or "")
        current_start = float(current_chunk["start_seconds"])
        previous_end = float(previous_chunk["start_seconds"]) + float(previous_chunk["duration_seconds"])
        actual_overlap = max(0.0, previous_end - current_start)
        if actual_overlap <= 0:
            continue
        seam = current_start + (actual_overlap / 2)
        previous_count, current_count, confidence = find_boundary_overlap(previous_text, current_text)

        if previous_count:
            previous_translation = str(previous_audio.get("translation") or "")
            current_translation = str(current_audio.get("translation") or "")
            translation_previous, translation_current, _ = find_boundary_overlap(
                previous_translation,
                current_translation,
            )
            retained = _overlap_source(
                previous_text,
                current_text,
                previous_count,
                current_count,
            )
            if retained == "current":
                _trim_audio_suffix(previous_audio, previous_count)
                if previous_audio.get("translation") is not None:
                    if translation_previous:
                        previous_audio["translation"] = _trim_suffix(
                            previous_audio.get("translation"),
                            translation_previous,
                        ) or None
                    else:
                        segment_translation = _join_segment_field(
                            previous_audio.get("segments") or [],
                            "translation",
                        )
                        if segment_translation:
                            previous_audio["translation"] = segment_translation
                removal = {"removed_suffix_tokens": previous_count}
            else:
                _trim_audio_prefix(current_audio, current_count)
                if current_audio.get("translation") is not None:
                    if translation_current:
                        current_audio["translation"] = _trim_prefix(
                            current_audio.get("translation"),
                            translation_current,
                        ) or None
                    else:
                        segment_translation = _join_segment_field(
                            current_audio.get("segments") or [],
                            "translation",
                        )
                        if segment_translation:
                            current_audio["translation"] = segment_translation
                removal = {"removed_prefix_tokens": current_count}
            previous_audio["boundary_reconciliation"] = {
                "strategy": "text-alignment",
                "retained_boundary": retained,
                **removal,
                "matched_prefix_tokens": current_count,
                "confidence": round(confidence, 4),
            }
        elif previous_text and current_text:
            # Gemma timestamps and speech-rate estimates are not reliable enough to justify
            # deleting unmatched dialogue. A possible duplicate is safer than an omission.
            previous_audio["boundary_reconciliation"] = {
                "strategy": "preserve-unmatched",
                "overlap_seconds": actual_overlap,
            }

        previous_visual = previous_chunk.get("visual") or {}
        current_visual = current_chunk.get("visual") or {}
        _partition_visual_by_time(previous_visual, current_visual, seam)

    _enforce_audio_chronology(reconciled)
    return reconciled


def concatenate_audio_field(chunks: list[dict[str, Any]], field: str) -> str | None:
    values = [
        str((chunk.get("audio") or {}).get(field) or "").strip()
        for chunk in chunks
        if str((chunk.get("audio") or {}).get(field) or "").strip()
    ]
    if not values:
        return None if field == "translation" else ""
    return " ".join(values)
