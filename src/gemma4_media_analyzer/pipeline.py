"""Resumable, backend-neutral audio and video analysis orchestration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .backend import AnalyzerBackend
from .continuity import VISUAL_CONTEXT_VERSION, build_visual_continuity
from .errors import ConfigurationError
from .gemma import (
    AUDIO_SEGMENT_VERSION,
    AUDIO_TIMESTAMP_VERSION,
    Gemma4Analyzer,
    validate_audio_result,
    validate_visual_result,
)
from .media import MediaChunk, MediaInfo, prepare_chunks
from .reconcile import concatenate_audio_field, reconcile_overlapping_chunks
from .render import render_json, render_markdown
from .server import LlamaCppAnalyzer, OllamaAnalyzer

Progress = Callable[[str], None]


@dataclass(frozen=True)
class AnalysisConfig:
    output_dir: Path
    backend: str = "transformers"
    model_id: str = "google/gemma-4-12B-it"
    chunk_seconds: float = 30.0
    chunk_overlap_seconds: float = 2.0
    fps: float = 1.0
    max_edge: int = 720
    source_language: str | None = None
    translate_to: str | None = None
    user_context: str | None = None
    device_map: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 4096
    seed: int = 7
    server_url: str | None = None
    api_key_env: str | None = None
    server_timeout_seconds: float = 3600.0
    server_stream: bool = True
    media_mode: str = "base64"
    media_root: Path | None = None
    server_context_size: int = 16384
    keep_alive: str = "30m"

    def validate(self) -> None:
        if self.backend not in {"transformers", "llama-cpp", "ollama"}:
            raise ConfigurationError(f"Unsupported inference backend: {self.backend}")
        if self.chunk_seconds <= 0 or self.chunk_seconds > 30:
            raise ConfigurationError("Chunk duration must be greater than 0 and at most 30 seconds")
        if self.chunk_overlap_seconds < 0 or self.chunk_overlap_seconds >= self.chunk_seconds:
            raise ConfigurationError("Chunk overlap must be non-negative and shorter than the chunk duration")
        if self.fps <= 0:
            raise ConfigurationError("FPS must be positive")
        if self.max_edge < 224:
            raise ConfigurationError("Maximum video edge must be at least 224 pixels")
        if self.max_new_tokens < 128:
            raise ConfigurationError("max_new_tokens must be at least 128")
        if self.server_context_size < 2048:
            raise ConfigurationError("Server context size must be at least 2048")
        if self.server_timeout_seconds <= 0:
            raise ConfigurationError("Server timeout must be positive")
        if self.media_mode not in {"base64", "file-url"}:
            raise ConfigurationError("Media mode must be base64 or file-url")
        if self.backend == "ollama" and self.media_mode != "base64":
            raise ConfigurationError("Ollama currently supports only base64 media mode")


def _json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.partial")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _json_read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def _source_identity(source: Path) -> dict[str, Any]:
    resolved = source.resolve()
    try:
        stat = resolved.stat()
    except OSError as exc:
        raise ConfigurationError(f"Cannot read input media {resolved}: {exc}") from exc
    return {"path": str(resolved), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _empty_audio(language: str | None) -> dict[str, Any]:
    return {
        "language": language or "none",
        "transcript": "",
        "translation": None,
        "segments": [],
        "non_speech_audio": [],
    }


def _empty_visual() -> dict[str, Any]:
    return {"summary": "", "shots": [], "people": [], "objects": [], "setting": ""}


def _visual_audio_context(audio: dict[str, Any]) -> str:
    transcript = str(audio.get("transcript") or "").strip()
    translation = str(audio.get("translation") or "").strip()
    if translation and translation.casefold() != transcript.casefold():
        return f"{transcript}\nTranslation: {translation}" if transcript else translation
    return transcript


class MediaAnalysisPipeline:
    def __init__(
        self,
        config: AnalysisConfig,
        *,
        analyzer: AnalyzerBackend | None = None,
        progress: Progress | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.progress = progress or (lambda _message: None)
        self.analyzer = analyzer

    def _create_analyzer(self) -> AnalyzerBackend:
        config = self.config
        if config.backend == "transformers":
            return Gemma4Analyzer(
                model_id=config.model_id,
                device_map=config.device_map,
                dtype=config.dtype,
                max_new_tokens=config.max_new_tokens,
                seed=config.seed,
            )
        if config.backend == "llama-cpp":
            return LlamaCppAnalyzer(
                model_id=config.model_id,
                base_url=config.server_url or "http://127.0.0.1:8080",
                api_key_env=config.api_key_env,
                timeout_seconds=config.server_timeout_seconds,
                stream=config.server_stream,
                media_mode=config.media_mode,
                media_root=config.media_root or config.output_dir,
                context_size=config.server_context_size,
                max_new_tokens=config.max_new_tokens,
                seed=config.seed,
            )
        return OllamaAnalyzer(
            model_id=config.model_id,
            base_url=config.server_url or "http://127.0.0.1:11434",
            api_key_env=config.api_key_env,
            timeout_seconds=config.server_timeout_seconds,
            stream=config.server_stream,
            context_size=config.server_context_size,
            keep_alive=config.keep_alive,
            max_new_tokens=config.max_new_tokens,
            seed=config.seed,
        )

    def _workspace(self, source: Path) -> tuple[Path, dict[str, Any]]:
        identity = _source_identity(source)
        preparation = {
            "source": identity,
            "chunk_seconds": self.config.chunk_seconds,
            "chunk_overlap_seconds": self.config.chunk_overlap_seconds,
            "fps": self.config.fps,
            "max_edge": self.config.max_edge,
        }
        safe_stem = "".join(character if character.isalnum() or character in "-_" else "_" for character in source.stem)
        workspace = self.config.output_dir / ".gemma4-cache" / f"{safe_stem}-{_stable_hash(preparation)}"
        return workspace, preparation

    def prepare(self, source: Path) -> tuple[MediaInfo, list[MediaChunk], Path]:
        source = source.resolve()
        workspace, preparation = self._workspace(source)
        self.progress("Probing and preprocessing media with FFmpeg...")
        info, chunks = prepare_chunks(
            source,
            workspace / "chunks",
            chunk_seconds=self.config.chunk_seconds,
            overlap_seconds=self.config.chunk_overlap_seconds,
            fps=self.config.fps,
            max_edge=self.config.max_edge,
        )
        _json_write(
            workspace / "manifest.json",
            {
                "preparation": preparation,
                "media": asdict(info),
                "chunks": [chunk.to_manifest() for chunk in chunks],
            },
        )
        return info, chunks, workspace

    def analyze(self, source: Path) -> tuple[Path, Path]:
        source = source.resolve()
        info, chunks, workspace = self.prepare(source)
        user_context = " ".join((self.config.user_context or "").split()) or None
        cache_settings = {
            "backend": self.config.backend,
            "model_id": self.config.model_id,
            "source_language": self.config.source_language,
            "translate_to": self.config.translate_to,
            "chunk_overlap_seconds": self.config.chunk_overlap_seconds,
            "max_new_tokens": self.config.max_new_tokens,
            "seed": self.config.seed,
            "device_map": self.config.device_map if self.config.backend == "transformers" else None,
            "dtype": self.config.dtype if self.config.backend == "transformers" else None,
            "server_url": self.config.server_url,
            "server_context_size": self.config.server_context_size,
            "media_mode": self.config.media_mode,
            "visual_transport": "sampled-frames-v1" if self.config.backend == "llama-cpp" else "video",
        }
        analysis_settings = {
            **cache_settings,
            "audio_segment_version": AUDIO_SEGMENT_VERSION,
            "audio_timestamp_version": AUDIO_TIMESTAMP_VERSION,
            "visual_context_version": VISUAL_CONTEXT_VERSION,
            "user_context": user_context,
        }
        result_dir = workspace / "analysis" / _stable_hash(cache_settings)
        analyzer = self.analyzer
        results: list[dict[str, Any]] = []

        for number, chunk in enumerate(chunks, start=1):
            self.progress(f"Analyzing chunk {number}/{len(chunks)} ({chunk.start_seconds:.1f}s)...")
            audio_cache = result_dir / (
                f"chunk-{chunk.index:06d}-audio-{AUDIO_TIMESTAMP_VERSION}.json"
            )

            if chunk.audio_path is None:
                audio = _empty_audio(self.config.source_language)
            elif audio_cache.is_file():
                cached_audio = _json_read(audio_cache)
                audio = validate_audio_result(
                    cached_audio,
                    start_seconds=chunk.start_seconds,
                    duration_seconds=chunk.duration_seconds,
                    timestamps_are_absolute=True,
                )
                if audio != cached_audio:
                    _json_write(audio_cache, audio)
            else:
                if analyzer is None:
                    analyzer = self._create_analyzer()
                self.progress(f"Transcribing audio for chunk {number}/{len(chunks)}...")
                audio = analyzer.transcribe(
                    chunk.audio_path,
                    start_seconds=chunk.start_seconds,
                    duration_seconds=chunk.duration_seconds,
                    source_language=self.config.source_language,
                    translate_to=self.config.translate_to,
                )
                audio = validate_audio_result(
                    audio,
                    start_seconds=chunk.start_seconds,
                    duration_seconds=chunk.duration_seconds,
                    timestamps_are_absolute=True,
                )
                _json_write(audio_cache, audio)

            if chunk.video_path is None:
                visual = _empty_visual()
            else:
                transcript = _visual_audio_context(audio)
                continuity = build_visual_continuity(results)
                visual_context_key = _stable_hash(
                    {
                        "version": VISUAL_CONTEXT_VERSION,
                        "user_context": user_context,
                        "transcript": transcript,
                        "continuity": continuity,
                    }
                )
                visual_cache = result_dir / (
                    f"chunk-{chunk.index:06d}-visual-context-{visual_context_key}.json"
                )
                if visual_cache.is_file():
                    cached_visual = _json_read(visual_cache)
                    visual = validate_visual_result(
                        cached_visual,
                        start_seconds=chunk.start_seconds,
                        duration_seconds=chunk.duration_seconds,
                        timestamps_are_absolute=True,
                    )
                    if visual != cached_visual:
                        _json_write(visual_cache, visual)
                else:
                    if analyzer is None:
                        analyzer = self._create_analyzer()
                    frame_detail = (
                        f" using {len(chunk.frame_paths)} sampled frames" if chunk.frame_paths else ""
                    )
                    self.progress(
                        f"Describing video for chunk {number}/{len(chunks)}{frame_detail}..."
                    )
                    visual = analyzer.describe(
                        chunk.video_path,
                        start_seconds=chunk.start_seconds,
                        duration_seconds=chunk.duration_seconds,
                        frame_paths=chunk.frame_paths,
                        transcript=transcript,
                        continuity=continuity,
                        user_context=user_context,
                    )
                    visual = validate_visual_result(
                        visual,
                        start_seconds=chunk.start_seconds,
                        duration_seconds=chunk.duration_seconds,
                        timestamps_are_absolute=True,
                    )
                    _json_write(visual_cache, visual)

            results.append(
                {
                    "index": chunk.index,
                    "start_seconds": chunk.start_seconds,
                    "duration_seconds": chunk.duration_seconds,
                    "audio": audio,
                    "visual": visual,
                }
            )

        self.progress("Reconciling overlapping chunk boundaries...")
        reconciled_results = reconcile_overlapping_chunks(results, self.config.chunk_overlap_seconds)
        document = {
            "source": str(source),
            "media_type": info.media_type,
            "duration_seconds": info.duration_seconds,
            "backend": self.config.backend,
            "model": self.config.model_id,
            "settings": analysis_settings,
            "transcript": concatenate_audio_field(reconciled_results, "transcript"),
            "translation": concatenate_audio_field(reconciled_results, "translation"),
            "chunks": reconciled_results,
        }
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = self.config.output_dir / f"{source.stem}.analysis.md"
        json_path = self.config.output_dir / f"{source.stem}.analysis.json"
        markdown_path.write_text(render_markdown(document), encoding="utf-8")
        render_json(json_path, document)
        self.progress(f"Analysis complete: {markdown_path}")
        return markdown_path, json_path


# Backward-compatible import for applications built against version 0.2.
VideoAnalysisPipeline = MediaAnalysisPipeline
