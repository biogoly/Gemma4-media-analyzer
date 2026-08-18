"""FFmpeg/ffprobe preprocessing for bounded multimodal chunks."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, ExternalToolError


@dataclass(frozen=True)
class MediaInfo:
    duration_seconds: float
    width: int
    height: int
    has_audio: bool
    has_video: bool

    @property
    def media_type(self) -> str:
        return "video" if self.has_video else "audio"


@dataclass(frozen=True)
class MediaChunk:
    index: int
    start_seconds: float
    duration_seconds: float
    video_path: Path | None
    audio_path: Path | None
    frame_paths: tuple[Path, ...] = ()

    def to_manifest(self) -> dict[str, Any]:
        data = asdict(self)
        data["video_path"] = str(self.video_path) if self.video_path else None
        data["audio_path"] = str(self.audio_path) if self.audio_path else None
        data["frame_paths"] = [str(path) for path in self.frame_paths]
        return data


def _require_tool(name: str) -> str:
    executable = shutil.which(name)
    if not executable:
        raise ConfigurationError(f"Required executable was not found on PATH: {name}")
    return executable


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise ExternalToolError(f"Command failed ({command[0]}): {detail[-2000:]}") from exc


def probe_media(source: Path, *, ffprobe_bin: str | None = None) -> MediaInfo:
    """Read duration, dimensions, and audio presence without decoding the file."""

    if not source.is_file():
        raise ConfigurationError(f"Input media does not exist: {source}")
    ffprobe = ffprobe_bin or _require_tool("ffprobe")
    result = _run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,width,height:stream_disposition=attached_pic",
            "-of",
            "json",
            str(source),
        ]
    )
    try:
        payload = json.loads(result.stdout)
        duration = float(payload["format"]["duration"])
        streams = payload.get("streams", [])
        video = next(
            (
                stream
                for stream in streams
                if stream.get("codec_type") == "video"
                and not (stream.get("disposition") or {}).get("attached_pic")
            ),
            None,
        )
        has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExternalToolError(f"ffprobe could not read usable media from {source}") from exc
    if video is None and not has_audio:
        raise ExternalToolError(f"ffprobe found no audio or video stream in {source}")
    if not math.isfinite(duration) or duration <= 0:
        raise ExternalToolError(f"Invalid media duration reported for {source}: {duration}")
    return MediaInfo(
        duration_seconds=duration,
        width=int(video.get("width") or 0) if video else 0,
        height=int(video.get("height") or 0) if video else 0,
        has_audio=has_audio,
        has_video=video is not None,
    )


def chunk_starts(
    duration_seconds: float,
    chunk_seconds: float,
    overlap_seconds: float = 0.0,
) -> list[float]:
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ConfigurationError("Media duration must be positive")
    if not math.isfinite(chunk_seconds) or chunk_seconds <= 0 or chunk_seconds > 30:
        raise ConfigurationError("Chunk duration must be greater than 0 and no more than 30 seconds")
    if not math.isfinite(overlap_seconds) or overlap_seconds < 0 or overlap_seconds >= chunk_seconds:
        raise ConfigurationError("Chunk overlap must be non-negative and shorter than the chunk duration")
    if duration_seconds <= chunk_seconds:
        return [0.0]
    stride = chunk_seconds - overlap_seconds
    count = math.ceil((duration_seconds - chunk_seconds) / stride) + 1
    return [index * stride for index in range(count)]


def chunk_count(duration_seconds: float, chunk_seconds: float, overlap_seconds: float = 0.0) -> int:
    return len(chunk_starts(duration_seconds, chunk_seconds, overlap_seconds))


def _complete(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _atomic_target(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def _prepare_video_frames(video_path: Path, ffmpeg: str) -> tuple[Path, ...]:
    """Decode an already sampled video chunk without changing its frame rate."""

    frame_dir = video_path.with_name(f"{video_path.stem}-frames")
    marker = frame_dir / ".complete"
    existing_frames = tuple(sorted(frame_dir.glob("frame-*.jpg")))
    reusable = marker.is_file() and existing_frames and all(_complete(path) for path in existing_frames)
    if not reusable:
        frame_dir.mkdir(parents=True, exist_ok=True)
        _run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video_path),
                "-map",
                "0:v:0",
                "-fps_mode",
                "passthrough",
                "-q:v",
                "2",
                str(frame_dir / "frame-%06d.jpg"),
            ]
        )
        marker.write_text("complete\n", encoding="ascii")
    frames = tuple(sorted(frame_dir.glob("frame-*.jpg")))
    if not frames or any(not _complete(path) for path in frames):
        raise ExternalToolError(f"FFmpeg produced no usable frames for {video_path}")
    return frames


def prepare_chunks(
    source: Path,
    chunk_dir: Path,
    *,
    chunk_seconds: float = 30.0,
    overlap_seconds: float = 0.0,
    fps: float = 1.0,
    max_edge: int = 720,
    ffmpeg_bin: str | None = None,
    ffprobe_bin: str | None = None,
) -> tuple[MediaInfo, list[MediaChunk]]:
    """Create analysis-ready MP4 and WAV chunks, resuming completed files."""

    info = probe_media(source, ffprobe_bin=ffprobe_bin)
    if info.has_video and fps <= 0:
        raise ConfigurationError("Frame sampling rate must be positive")
    if info.has_video and max_edge < 224:
        raise ConfigurationError("Maximum video edge must be at least 224 pixels")
    ffmpeg = ffmpeg_bin or _require_tool("ffmpeg")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunks: list[MediaChunk] = []
    normalized_edge = max_edge - (max_edge % 2)
    scale = (
        f"fps={fps},"
        f"scale=w='if(gte(iw,ih),min(iw,{normalized_edge}),-2)':"
        f"h='if(lt(iw,ih),min(ih,{normalized_edge}),-2)'"
    )

    for index, start in enumerate(chunk_starts(info.duration_seconds, chunk_seconds, overlap_seconds)):
        duration = min(chunk_seconds, info.duration_seconds - start)
        video_path = chunk_dir / f"chunk-{index:06d}.mp4" if info.has_video else None
        audio_path = chunk_dir / f"chunk-{index:06d}.wav" if info.has_audio else None

        if video_path is not None and not _complete(video_path):
            partial = _atomic_target(video_path)
            partial.unlink(missing_ok=True)
            _run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start:.6f}",
                    "-t",
                    f"{duration:.6f}",
                    "-i",
                    str(source),
                    "-map",
                    "0:v:0",
                    "-an",
                    "-vf",
                    scale,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(partial),
                ]
            )
            partial.replace(video_path)

        if audio_path is not None and not _complete(audio_path):
            partial = _atomic_target(audio_path)
            partial.unlink(missing_ok=True)
            _run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{start:.6f}",
                    "-t",
                    f"{duration:.6f}",
                    "-i",
                    str(source),
                    "-map",
                    "0:a:0",
                    "-vn",
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_f32le",
                    str(partial),
                ]
            )
            partial.replace(audio_path)

        frame_paths = _prepare_video_frames(video_path, ffmpeg) if video_path is not None else ()
        chunks.append(MediaChunk(index, start, duration, video_path, audio_path, frame_paths))
    return info, chunks
