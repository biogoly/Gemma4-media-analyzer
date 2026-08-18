"""Native Hugging Face Transformers backend and output validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, ModelOutputError
from .prompts import SYSTEM_PROMPT, audio_prompt, visual_prompt
from .text_utils import normalized_words, text_tokens

AUDIO_TIMESTAMP_VERSION = "duration-bounded-affine-v1"
AUDIO_SEGMENT_VERSION = "canonical-transcript-v1"


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the first balanced JSON object, tolerating model wrapper tokens/fences."""

    start = text.find("{")
    if start < 0:
        raise ModelOutputError(f"Model response contained no JSON object: {text[:300]!r}")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                try:
                    value = json.loads(text[start : index + 1])
                except json.JSONDecodeError as exc:
                    raise ModelOutputError(f"Model returned malformed JSON: {exc}") from exc
                if not isinstance(value, dict):
                    raise ModelOutputError("Model JSON response was not an object")
                return value
    raise ModelOutputError("Model response contained an unterminated JSON object")


def _string(value: Any, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    return str(value).strip() if value is not None else ""


def _bounded_string(value: Any, max_length: int) -> str:
    return str(_string(value) or "")[:max_length].rstrip()


def _time(value: Any, fallback: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = fallback
    if not math.isfinite(number):
        number = fallback
    return max(0.0, number)


def _normalize_intervals(
    intervals: list[tuple[float, float]],
    duration_seconds: float | None,
) -> list[tuple[float, float]]:
    """Bound model-estimated relative intervals while preserving their supplied order."""

    ordered: list[tuple[float, float]] = []
    previous_start = 0.0
    for raw_start, raw_end in intervals:
        raw_duration = max(0.0, raw_end - raw_start)
        start = max(previous_start, raw_start)
        ordered.append((start, start + raw_duration))
        previous_start = start

    if duration_seconds is None:
        return ordered
    duration = _time(duration_seconds, 0.0)
    reported_end = max((end for start, end in ordered), default=0.0)
    scale = duration / reported_end if reported_end > duration and reported_end > 0 else 1.0
    return [
        (
            min(duration, start * scale),
            min(duration, max(start * scale, end * scale)),
        )
        for start, end in ordered
    ]


def _join_segment_text(segments: list[dict[str, Any]]) -> str:
    return " ".join(str(segment.get("text") or "").strip() for segment in segments).strip()


def _canonical_segment_parts(transcript: str, counts: list[int]) -> list[str]:
    tokens = text_tokens(transcript)
    parts: list[str] = []
    cursor = 0
    for count in counts:
        if count <= 0:
            parts.append("")
            continue
        start = tokens[cursor].start
        cursor += count
        end = tokens[cursor].start if cursor < len(tokens) else len(transcript)
        parts.append(transcript[start:end].strip())
    return parts


def _repair_audio_segment_coverage(
    audio: dict[str, Any],
    *,
    start_seconds: float,
    duration_seconds: float | None,
) -> None:
    """Make timestamped segments cover the canonical transcript or mark a safe fallback."""

    transcript = str(audio.get("transcript") or "").strip()
    segments = [
        segment
        for segment in audio.get("segments") or []
        if isinstance(segment, dict) and str(segment.get("text") or "").strip()
    ]
    audio["segments"] = segments
    joined = _join_segment_text(segments)

    if not transcript and joined:
        audio["transcript"] = joined
        transcript = joined
        audio["segment_coverage"] = {
            "version": AUDIO_SEGMENT_VERSION,
            "status": "transcript-from-segments",
            "complete": True,
        }
        return
    if not transcript:
        audio["segment_coverage"] = {
            "version": AUDIO_SEGMENT_VERSION,
            "status": "empty",
            "complete": True,
        }
        return

    chunk_end = (
        start_seconds + max(0.0, float(duration_seconds))
        if duration_seconds is not None
        else max((float(segment.get("end", start_seconds)) for segment in segments), default=start_seconds)
    )
    if not segments:
        audio["segments"] = [
            {
                "start": start_seconds,
                "end": chunk_end,
                "speaker": "Speaker",
                "text": transcript,
                "translation": None,
                "inferred_timestamp": True,
            }
        ]
        audio["segment_coverage"] = {
            "version": AUDIO_SEGMENT_VERSION,
            "status": "synthesized",
            "complete": True,
        }
        return

    transcript_words = normalized_words(transcript)
    segment_words = normalized_words(joined)
    if segment_words == transcript_words:
        counts = [len(normalized_words(segment.get("text"))) for segment in segments]
        canonical_parts = _canonical_segment_parts(transcript, counts)
        for segment, text in zip(segments, canonical_parts, strict=True):
            segment["text"] = text
        audio["segment_coverage"] = {
            "version": AUDIO_SEGMENT_VERSION,
            "status": "complete",
            "complete": True,
        }
        return

    if segment_words and segment_words == transcript_words[: len(segment_words)]:
        missing_token = text_tokens(transcript)[len(segment_words)]
        missing = transcript[missing_token.start :].strip()
        last = segments[-1]
        inferred_start = min(chunk_end, float(last.get("end", chunk_end)))
        segments.append(
            {
                "start": inferred_start,
                "end": chunk_end,
                "speaker": str(last.get("speaker") or "Speaker"),
                "text": missing,
                "translation": None,
                "inferred_timestamp": True,
            }
        )
        audio["segment_coverage"] = {
            "version": AUDIO_SEGMENT_VERSION,
            "status": "repaired-tail",
            "complete": True,
        }
        return

    if segment_words and segment_words == transcript_words[-len(segment_words) :]:
        missing_count = len(transcript_words) - len(segment_words)
        first_segment_token = text_tokens(transcript)[missing_count]
        missing = transcript[: first_segment_token.start].strip()
        first = segments[0]
        inferred_end = max(start_seconds, float(first.get("start", start_seconds)))
        segments.insert(
            0,
            {
                "start": start_seconds,
                "end": inferred_end,
                "speaker": str(first.get("speaker") or "Speaker"),
                "text": missing,
                "translation": None,
                "inferred_timestamp": True,
            },
        )
        audio["segment_coverage"] = {
            "version": AUDIO_SEGMENT_VERSION,
            "status": "repaired-head",
            "complete": True,
        }
        return

    audio["segment_coverage"] = {
        "version": AUDIO_SEGMENT_VERSION,
        "status": "canonical-fallback",
        "complete": False,
    }


def validate_audio_result(
    result: dict[str, Any],
    *,
    start_seconds: float,
    duration_seconds: float | None = None,
    timestamps_are_absolute: bool = False,
) -> dict[str, Any]:
    if not isinstance(result.get("segments", []), list):
        raise ModelOutputError("Audio response 'segments' must be an array")
    origin = start_seconds if timestamps_are_absolute else 0.0
    raw_segments: list[tuple[dict[str, Any], float, float]] = []
    intervals: list[tuple[float, float]] = []
    for raw in result.get("segments", []):
        if not isinstance(raw, dict):
            continue
        relative_start = max(0.0, _time(raw.get("start"), origin) - origin)
        relative_end = max(relative_start, _time(raw.get("end"), origin + relative_start) - origin)
        raw_segments.append((raw, relative_start, relative_end))
        intervals.append((relative_start, relative_end))

    segments: list[dict[str, Any]] = []
    normalized = _normalize_intervals(intervals, duration_seconds)
    for (raw, _relative_start, _relative_end), (relative_start, relative_end) in zip(
        raw_segments,
        normalized,
        strict=True,
    ):
        segments.append(
            {
                "start": start_seconds + relative_start,
                "end": start_seconds + relative_end,
                "speaker": _string(raw.get("speaker")) or "Speaker",
                "text": _string(raw.get("text")) or "",
                "translation": _string(raw.get("translation"), nullable=True),
            }
        )
    sounds = result.get("non_speech_audio", [])
    if not isinstance(sounds, list):
        sounds = []
    audio = {
        "language": _string(result.get("language")) or "unknown",
        "transcript": _string(result.get("transcript")) or "",
        "translation": _string(result.get("translation"), nullable=True),
        "segments": segments,
        "non_speech_audio": [str(item).strip() for item in sounds if str(item).strip()],
    }
    _repair_audio_segment_coverage(
        audio,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
    )
    return audio


def validate_visual_result(
    result: dict[str, Any],
    *,
    start_seconds: float,
    duration_seconds: float | None = None,
    timestamps_are_absolute: bool = False,
) -> dict[str, Any]:
    if not isinstance(result.get("shots", []), list):
        raise ModelOutputError("Visual response 'shots' must be an array")
    origin = start_seconds if timestamps_are_absolute else 0.0
    raw_shots: list[tuple[dict[str, Any], float, float]] = []
    intervals: list[tuple[float, float]] = []
    for raw in result.get("shots", [])[:10]:
        if not isinstance(raw, dict):
            continue
        relative_start = max(0.0, _time(raw.get("start"), origin) - origin)
        relative_end = max(relative_start, _time(raw.get("end"), origin + relative_start) - origin)
        raw_shots.append((raw, relative_start, relative_end))
        intervals.append((relative_start, relative_end))

    shots: list[dict[str, Any]] = []
    seen_visible_text: set[str] = set()
    normalized = _normalize_intervals(intervals, duration_seconds)
    for (raw, _relative_start, _relative_end), (relative_start, relative_end) in zip(
        raw_shots,
        normalized,
        strict=True,
    ):
        visible = raw.get("visible_text", [])
        if not isinstance(visible, list):
            visible = [visible] if visible else []
        bounded_visible: list[str] = []
        for item in visible:
            text = _bounded_string(item, 160)
            key = text.casefold()
            if not text or key in seen_visible_text:
                continue
            seen_visible_text.add(key)
            bounded_visible.append(text)
            if len(bounded_visible) == 6:
                break
        shots.append(
            {
                "start": start_seconds + relative_start,
                "end": start_seconds + relative_end,
                "description": _bounded_string(raw.get("description"), 360),
                "visible_text": bounded_visible,
            }
        )

    def string_list(name: str, max_items: int) -> list[str]:
        values = result.get(name, [])
        if not isinstance(values, list):
            return []
        bounded: list[str] = []
        for item in values:
            text = _bounded_string(item, 160)
            if text:
                bounded.append(text)
            if len(bounded) == max_items:
                break
        return bounded

    return {
        "summary": _bounded_string(result.get("summary"), 500),
        "shots": shots,
        "people": string_list("people", 12),
        "objects": string_list("objects", 24),
        "setting": _bounded_string(result.get("setting"), 360),
    }


class Gemma4Analyzer:
    """Load Gemma 4 locally through Transformers only when this backend is selected."""

    def __init__(
        self,
        *,
        model_id: str,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 4096,
        seed: int = 7,
    ) -> None:
        self.model_id = model_id
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.processor: Any = None
        self.model: Any = None
        self.torch: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise ConfigurationError(
                "The Transformers backend is not installed. Run: pip install -e '.[transformers]'"
            ) from exc
        dtype: Any = self.dtype
        if self.dtype != "auto":
            dtype = getattr(torch, self.dtype, None)
            if dtype is None:
                raise ConfigurationError(f"Unknown torch dtype: {self.dtype}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            dtype=dtype,
            device_map=self.device_map,
        )
        self.model.eval()
        self.torch = torch

    def _generate(self, messages: list[dict[str, Any]]) -> str:
        self.load()
        self.torch.manual_seed(self.seed)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[-1]
        with self.torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
            )
        return self.processor.decode(output[0][input_length:], skip_special_tokens=False)

    def transcribe(
        self,
        audio_path: Path,
        *,
        start_seconds: float,
        duration_seconds: float,
        source_language: str | None,
        translate_to: str | None,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": audio_prompt(source_language, translate_to, duration_seconds),
                    },
                    {"type": "audio", "audio": str(audio_path.resolve())},
                ],
            },
        ]
        return validate_audio_result(
            extract_json_object(self._generate(messages)),
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )

    def describe(
        self,
        video_path: Path,
        *,
        start_seconds: float,
        duration_seconds: float,
        frame_paths: tuple[Path, ...] = (),
        transcript: str | None = None,
        continuity: str | None = None,
        user_context: str | None = None,
    ) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video_path.resolve())},
                    {
                        "type": "text",
                        "text": visual_prompt(
                            duration_seconds,
                            transcript=transcript,
                            continuity=continuity,
                            user_context=user_context,
                        ),
                    },
                ],
            },
        ]
        return validate_visual_result(
            extract_json_object(self._generate(messages)),
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
