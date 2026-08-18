"""Common interface implemented by local and server inference backends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol


class AnalyzerBackend(Protocol):
    """Minimal contract used by the media-analysis pipeline."""

    def load(self) -> None: ...

    def transcribe(
        self,
        audio_path: Path,
        *,
        start_seconds: float,
        duration_seconds: float,
        source_language: str | None,
        translate_to: str | None,
    ) -> dict[str, Any]: ...

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
    ) -> dict[str, Any]: ...
