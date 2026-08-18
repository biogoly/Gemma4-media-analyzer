"""Streaming HTTP inference backends for llama.cpp and Ollama."""

from __future__ import annotations

import base64
import json
import os
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from .errors import ConfigurationError, MediaAnalyzerError, ModelOutputError
from .gemma import extract_json_object, validate_audio_result, validate_visual_result
from .prompts import SYSTEM_PROMPT, audio_prompt, visual_prompt


class BackendError(MediaAnalyzerError):
    """Raised when an inference server cannot be reached or returns an error."""


AUDIO_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "language": {"type": "string"},
        "transcript": {"type": "string"},
        "translation": {"type": ["string", "null"]},
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "minimum": 0},
                    "end": {"type": "number", "minimum": 0},
                    "speaker": {"type": "string"},
                    "text": {"type": "string"},
                    "translation": {"type": ["string", "null"]},
                },
                "required": ["start", "end", "speaker", "text", "translation"],
                "additionalProperties": False,
            },
        },
        "non_speech_audio": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["language", "transcript", "translation", "segments", "non_speech_audio"],
    "additionalProperties": False,
}

VISUAL_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "maxLength": 500},
        "shots": {
            "type": "array",
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number", "minimum": 0},
                    "end": {"type": "number", "minimum": 0},
                    "description": {"type": "string", "maxLength": 360},
                    "visible_text": {
                        "type": "array",
                        "maxItems": 6,
                        "items": {"type": "string", "maxLength": 160},
                    },
                },
                "required": ["start", "end", "description", "visible_text"],
                "additionalProperties": False,
            },
        },
        "people": {
            "type": "array",
            "maxItems": 12,
            "items": {"type": "string", "maxLength": 160},
        },
        "objects": {
            "type": "array",
            "maxItems": 24,
            "items": {"type": "string", "maxLength": 160},
        },
        "setting": {"type": "string", "maxLength": 360},
    },
    "required": ["summary", "shots", "people", "objects", "setting"],
    "additionalProperties": False,
}

UrlOpen = Callable[..., Any]


def _decode_line(raw: bytes | str) -> str:
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def parse_llama_sse(lines: Iterable[bytes | str]) -> str:
    """Accumulate text from an OpenAI-compatible server-sent event stream."""

    parts: list[str] = []
    for raw in lines:
        line = _decode_line(raw).strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BackendError(f"llama.cpp returned invalid SSE JSON: {payload[:160]}") from exc
        if event.get("error"):
            raise BackendError(f"llama.cpp error: {event['error']}")
        choices = event.get("choices") or []
        if choices:
            choice = choices[0]
            content = (choice.get("delta") or {}).get("content")
            if content is None:
                content = (choice.get("message") or {}).get("content")
            parts.append(_content_text(content))
    return "".join(parts)


def parse_ollama_ndjson(lines: Iterable[bytes | str]) -> str:
    """Accumulate text from Ollama's newline-delimited response stream."""

    parts: list[str] = []
    for raw in lines:
        line = _decode_line(raw).strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BackendError(f"Ollama returned invalid streaming JSON: {line[:160]}") from exc
        if event.get("error"):
            raise BackendError(f"Ollama error: {event['error']}")
        parts.append(_content_text((event.get("message") or {}).get("content")))
    return "".join(parts)


def _base64_file(path: Path) -> str:
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise BackendError(f"Unable to read media chunk {path}: {exc}") from exc


def _llama_media(path: Path, mode: str, media_root: Path | None) -> dict[str, str]:
    if mode == "base64":
        return {"data": _base64_file(path)}
    if mode != "file-url":
        raise ConfigurationError(f"Unsupported server media mode: {mode}")
    if media_root is None:
        raise ConfigurationError("--media-root is required with --media-mode file-url")
    try:
        relative = path.resolve().relative_to(media_root.resolve())
    except ValueError as exc:
        raise ConfigurationError(f"Media chunk {path} is outside --media-root {media_root}") from exc
    return {"url": f"file://{quote(relative.as_posix())}"}


def _llama_image(path: Path, mode: str, media_root: Path | None) -> dict[str, str]:
    if mode == "base64":
        return {"url": f"data:image/jpeg;base64,{_base64_file(path)}"}
    return _llama_media(path, mode, media_root)


class _JsonHttpClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: float,
        opener: UrlOpen = urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._opener = opener
        self._headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def _open(
        self,
        request: Request,
        parser: Callable[[Any], Any],
        *,
        retry_transport_errors: bool = True,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self._opener(request, timeout=self.timeout_seconds) as response:
                    return parser(response)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code not in {408, 409, 429, 500, 502, 503, 504} or attempt == 2:
                    raise BackendError(f"HTTP {exc.code} from {request.full_url}: {detail}") from exc
                last_error = exc
            except (URLError, TimeoutError, OSError) as exc:
                if not retry_transport_errors or attempt == 2:
                    raise BackendError(f"Cannot reach {request.full_url}: {exc}") from exc
                last_error = exc
            time.sleep(0.5 * (2**attempt))
        raise BackendError(f"Request failed: {last_error}")

    def check(self, path: str) -> None:
        request = Request(f"{self.base_url}{path}", headers=self._headers, method="GET")
        self._open(request, lambda response: response.read())

    def post(self, path: str, payload: dict[str, Any], parser: Callable[[Any], str]) -> str:
        request = Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers=self._headers,
            method="POST",
        )
        # Once an inference request has been dispatched, a timeout is ambiguous: the server
        # may still be working. Replaying it can duplicate a long-running multimodal task.
        return self._open(request, parser, retry_transport_errors=False)


class LlamaCppAnalyzer:
    """Gemma 4 analysis through llama.cpp's OpenAI-compatible server."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str = "http://127.0.0.1:8080",
        api_key_env: str | None = None,
        timeout_seconds: float = 3600,
        stream: bool = True,
        media_mode: str = "base64",
        media_root: Path | None = None,
        context_size: int = 16384,
        max_new_tokens: int = 4096,
        seed: int = 7,
        opener: UrlOpen = urlopen,
    ) -> None:
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise ConfigurationError(f"API-key environment variable is not set: {api_key_env}")
        normalized = base_url.rstrip("/")
        if normalized.endswith("/v1"):
            normalized = normalized[:-3]
        self.client = _JsonHttpClient(
            base_url=normalized,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        self.model_id = model_id
        self.stream = stream
        self.media_mode = media_mode
        self.media_root = media_root
        self.context_size = context_size
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self.client.check("/health")
            self._loaded = True

    def _generate(
        self,
        messages: list[dict[str, Any]],
        schema: dict[str, Any],
        *,
        max_tokens: int | None = None,
    ) -> str:
        self.load()
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_new_tokens if max_tokens is None else max_tokens,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
            "seed": self.seed,
            "stream": self.stream,
            "response_format": {"type": "json_schema", "schema": schema},
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        parser = parse_llama_sse if self.stream else self._nonstream
        text = self.client.post("/v1/chat/completions", payload, parser)
        if not text.strip():
            raise ModelOutputError("llama.cpp returned an empty completion")
        return text

    @staticmethod
    def _nonstream(response: Any) -> str:
        try:
            payload = json.loads(response.read())
            return _content_text(payload["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise BackendError("llama.cpp returned an invalid response") from exc

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
                    {
                        "type": "input_audio",
                        "input_audio": _llama_media(audio_path, self.media_mode, self.media_root),
                    },
                ],
            },
        ]
        result = extract_json_object(self._generate(messages, AUDIO_RESPONSE_SCHEMA))
        return validate_audio_result(
            result,
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
        if frame_paths:
            media_content = [
                {
                    "type": "image_url",
                    "image_url": _llama_image(path, self.media_mode, self.media_root),
                }
                for path in frame_paths
            ]
        else:
            media_content = [
                {
                    "type": "input_video",
                    "input_video": _llama_media(video_path, self.media_mode, self.media_root),
                }
            ]
        prompt = visual_prompt(
            duration_seconds,
            frame_count=len(frame_paths) or None,
            transcript=transcript,
            continuity=continuity,
            user_context=user_context,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": media_content
                + [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
        ]
        try:
            result = extract_json_object(self._generate(messages, VISUAL_RESPONSE_SCHEMA))
            return validate_visual_result(
                result,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
            )
        except ModelOutputError:
            retry_prompt = prompt + """

The previous attempt did not finish a valid JSON object. Retry from the beginning and prioritize
finishing the JSON over extra detail. Use at most 6 shots, at most 3 unique visible_text strings per
shot, at most 6 people, and at most 10 objects. Keep every description to two short sentences."""
            retry_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": media_content + [{"type": "text", "text": retry_prompt}],
                },
            ]
            retry_max_tokens = min(
                max(self.max_new_tokens, 6144),
                max(self.max_new_tokens, self.context_size // 2),
            )
            result = extract_json_object(
                self._generate(
                    retry_messages,
                    VISUAL_RESPONSE_SCHEMA,
                    max_tokens=retry_max_tokens,
                )
            )
            return validate_visual_result(
                result,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
            )


class OllamaAnalyzer:
    """Experimental Gemma 4 backend using Ollama's multimodal chat API."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str = "http://127.0.0.1:11434",
        api_key_env: str | None = None,
        timeout_seconds: float = 3600,
        stream: bool = True,
        context_size: int = 16384,
        keep_alive: str = "30m",
        max_new_tokens: int = 4096,
        seed: int = 7,
        opener: UrlOpen = urlopen,
    ) -> None:
        api_key = os.environ.get(api_key_env) if api_key_env else None
        if api_key_env and not api_key:
            raise ConfigurationError(f"API-key environment variable is not set: {api_key_env}")
        normalized = base_url.rstrip("/")
        if normalized.endswith("/api"):
            normalized = normalized[:-4]
        self.client = _JsonHttpClient(
            base_url=normalized,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            opener=opener,
        )
        self.model_id = model_id
        self.stream = stream
        self.context_size = context_size
        self.keep_alive = keep_alive
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self._loaded = False

    def load(self) -> None:
        if not self._loaded:
            self.client.check("/api/tags")
            self._loaded = True

    def _generate(self, prompt: str, media_path: Path, schema: dict[str, Any]) -> str:
        self.load()
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": [_base64_file(media_path)]},
            ],
            "format": schema,
            "think": False,
            "stream": self.stream,
            "keep_alive": self.keep_alive,
            "options": {
                "num_ctx": self.context_size,
                "num_predict": self.max_new_tokens,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
                "seed": self.seed,
            },
        }
        parser = parse_ollama_ndjson if self.stream else self._nonstream
        text = self.client.post("/api/chat", payload, parser)
        if not text.strip():
            raise ModelOutputError("Ollama returned an empty completion")
        return text

    @staticmethod
    def _nonstream(response: Any) -> str:
        try:
            payload = json.loads(response.read())
            return _content_text(payload["message"]["content"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise BackendError("Ollama returned an invalid response") from exc

    def transcribe(
        self,
        audio_path: Path,
        *,
        start_seconds: float,
        duration_seconds: float,
        source_language: str | None,
        translate_to: str | None,
    ) -> dict[str, Any]:
        text = self._generate(
            audio_prompt(source_language, translate_to, duration_seconds),
            audio_path,
            AUDIO_RESPONSE_SCHEMA,
        )
        return validate_audio_result(
            extract_json_object(text),
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
        text = self._generate(
            visual_prompt(
                duration_seconds,
                transcript=transcript,
                continuity=continuity,
                user_context=user_context,
            ),
            video_path,
            VISUAL_RESPONSE_SCHEMA,
        )
        return validate_visual_result(
            extract_json_object(text),
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
