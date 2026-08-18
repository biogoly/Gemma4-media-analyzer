import json
import subprocess

import pytest

from gemma4_media_analyzer.errors import ConfigurationError
from gemma4_media_analyzer.media import (
    MediaInfo,
    chunk_count,
    chunk_starts,
    prepare_chunks,
    probe_media,
)


def test_chunk_count_covers_tail_without_extra_chunk() -> None:
    assert chunk_count(60.0, 30.0) == 2
    assert chunk_count(60.01, 30.0) == 3
    assert chunk_count(0.25, 30.0) == 1


def test_overlapping_chunk_windows_never_exceed_limit() -> None:
    assert chunk_starts(60.0, 30.0, 2.0) == [0.0, 28.0, 56.0]
    assert chunk_count(60.0, 30.0, 2.0) == 3
    assert all(min(30.0, 60.0 - start) <= 30.0 for start in chunk_starts(60.0, 30.0, 2.0))


@pytest.mark.parametrize("value", [0, -1, 30.01])
def test_chunk_duration_is_bounded(value: float) -> None:
    with pytest.raises(ConfigurationError):
        chunk_count(60.0, value)


@pytest.mark.parametrize("overlap", [-1, 30, 31])
def test_chunk_overlap_is_bounded(overlap: float) -> None:
    with pytest.raises(ConfigurationError):
        chunk_count(60.0, 30.0, overlap)


def test_probe_ignores_embedded_cover_art(tmp_path, monkeypatch) -> None:
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"audio")
    payload = {
        "streams": [
            {"index": 0, "codec_type": "audio", "disposition": {"attached_pic": 0}},
            {
                "index": 1,
                "codec_type": "video",
                "width": 360,
                "height": 360,
                "disposition": {"attached_pic": 1},
            },
        ],
        "format": {"duration": "185.956583"},
    }
    result = subprocess.CompletedProcess([], 0, stdout=json.dumps(payload), stderr="")
    monkeypatch.setattr("gemma4_media_analyzer.media._run", lambda _command: result)

    info = probe_media(source, ffprobe_bin="ffprobe")

    assert info.media_type == "audio"
    assert info.has_audio is True
    assert info.has_video is False
    assert info.width == 0
    assert info.height == 0


def test_audio_only_input_creates_wav_chunks_without_video(tmp_path, monkeypatch) -> None:
    source = tmp_path / "recording.flac"
    source.write_bytes(b"audio")
    commands = []

    monkeypatch.setattr(
        "gemma4_media_analyzer.media.probe_media",
        lambda *args, **kwargs: MediaInfo(10, 0, 0, True, False),
    )

    def fake_run(command):
        commands.append(command)
        (tmp_path / "chunks" / "chunk-000000.partial.wav").write_bytes(b"wav")

    monkeypatch.setattr("gemma4_media_analyzer.media._run", fake_run)
    info, chunks = prepare_chunks(source, tmp_path / "chunks", ffmpeg_bin="ffmpeg")

    assert info.media_type == "audio"
    assert chunks[0].video_path is None
    assert chunks[0].audio_path is not None
    assert len(commands) == 1
    assert "0:a:0" in commands[0]


def test_video_chunks_include_exact_sampled_frames(tmp_path, monkeypatch) -> None:
    source = tmp_path / "recording.webm"
    source.write_bytes(b"video")
    commands = []
    chunk_dir = tmp_path / "chunks"

    monkeypatch.setattr(
        "gemma4_media_analyzer.media.probe_media",
        lambda *args, **kwargs: MediaInfo(10, 1280, 720, False, True),
    )

    def fake_run(command):
        commands.append(command)
        if str(command[-1]).endswith("partial.mp4"):
            (chunk_dir / "chunk-000000.partial.mp4").write_bytes(b"mp4")
        else:
            frame_dir = chunk_dir / "chunk-000000-frames"
            frame_dir.mkdir(parents=True, exist_ok=True)
            (frame_dir / "frame-000001.jpg").write_bytes(b"jpeg")

    monkeypatch.setattr("gemma4_media_analyzer.media._run", fake_run)
    _, chunks = prepare_chunks(source, chunk_dir, ffmpeg_bin="ffmpeg")

    assert [path.name for path in chunks[0].frame_paths] == ["frame-000001.jpg"]
    assert len(commands) == 2
    assert "passthrough" in commands[1]
