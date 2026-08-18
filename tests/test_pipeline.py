import json

from gemma4_media_analyzer.media import MediaChunk, MediaInfo
from gemma4_media_analyzer.pipeline import AnalysisConfig, MediaAnalysisPipeline


class FakeAnalyzer:
    def __init__(self) -> None:
        self.audio_calls = 0
        self.audio_kwargs = []
        self.video_calls = 0
        self.video_kwargs = []

    def load(self) -> None:
        pass

    def transcribe(self, audio_path, **kwargs):
        self.audio_calls += 1
        self.audio_kwargs.append(kwargs)
        return {
            "language": "English",
            "transcript": "Hello",
            "translation": None,
            "segments": [],
            "non_speech_audio": [],
        }

    def describe(self, video_path, **kwargs):
        self.video_calls += 1
        self.video_kwargs.append(kwargs)
        return {
            "summary": "A scene",
            "shots": [],
            "people": [],
            "objects": [],
            "setting": "Room",
        }


def test_pipeline_resumes_independent_cached_passes(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    video = tmp_path / "chunk.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"audio")
    fake = FakeAnalyzer()

    def prepare(*args, **kwargs):
        return MediaInfo(10, 720, 404, True, True), [MediaChunk(0, 0, 10, video, audio)]

    monkeypatch.setattr("gemma4_media_analyzer.pipeline.prepare_chunks", prepare)
    pipeline = MediaAnalysisPipeline(AnalysisConfig(output_dir=tmp_path / "out"), analyzer=fake)
    markdown, structured = pipeline.analyze(source)
    pipeline.analyze(source)

    assert markdown.is_file()
    assert structured.is_file()
    assert fake.audio_calls == 1
    assert fake.video_calls == 1


def test_pipeline_normalizes_legacy_cached_visual_results(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    video = tmp_path / "chunk.mp4"
    video.write_bytes(b"video")
    fake = FakeAnalyzer()
    fake.describe = lambda *args, **kwargs: {
        "summary": "A scene",
        "shots": [
            {
                "start": index,
                "end": index + 1,
                "description": "Scene",
                "visible_text": [f"Text {item}" for item in range(8)],
            }
            for index in range(12)
        ],
        "people": [],
        "objects": [],
        "setting": "Room",
    }

    def prepare(*args, **kwargs):
        return MediaInfo(10, 720, 404, False, True), [MediaChunk(0, 0, 10, video, None)]

    monkeypatch.setattr("gemma4_media_analyzer.pipeline.prepare_chunks", prepare)
    pipeline = MediaAnalysisPipeline(AnalysisConfig(output_dir=tmp_path / "out"), analyzer=fake)
    _, structured = pipeline.analyze(source)
    _, structured = pipeline.analyze(source)
    document = json.loads(structured.read_text(encoding="utf-8"))

    assert len(document["chunks"][0]["visual"]["shots"]) == 10
    assert len(document["chunks"][0]["visual"]["shots"][0]["visible_text"]) == 6


def test_audio_pipeline_skips_visual_inference(tmp_path, monkeypatch) -> None:
    source = tmp_path / "recording.mp3"
    source.write_bytes(b"source")
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"audio")
    fake = FakeAnalyzer()

    def prepare(*args, **kwargs):
        return MediaInfo(10, 0, 0, True, False), [MediaChunk(0, 0, 10, None, audio)]

    monkeypatch.setattr("gemma4_media_analyzer.pipeline.prepare_chunks", prepare)
    pipeline = MediaAnalysisPipeline(AnalysisConfig(output_dir=tmp_path / "out"), analyzer=fake)
    markdown, structured = pipeline.analyze(source)
    document = json.loads(structured.read_text(encoding="utf-8"))

    assert fake.audio_calls == 1
    assert fake.video_calls == 0
    assert document["media_type"] == "audio"
    assert document["transcript"] == "Hello"
    assert document["settings"]["chunk_overlap_seconds"] == 2.0
    assert document["settings"]["audio_segment_version"] == "canonical-transcript-v1"
    assert document["settings"]["audio_timestamp_version"] == "duration-bounded-affine-v1"
    assert fake.audio_kwargs[0]["duration_seconds"] == 10
    assert "# Gemma 4 Audio Analysis" in markdown.read_text(encoding="utf-8")


def test_pipeline_grounds_visual_chunks_with_audio_and_continuity(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    videos = [tmp_path / f"chunk-{index}.mp4" for index in range(2)]
    audios = [tmp_path / f"chunk-{index}.wav" for index in range(2)]
    for path in videos + audios:
        path.write_bytes(b"media")
    fake = FakeAnalyzer()
    transcripts = iter(
        [
            "Opening a 1958 British Royal Navy sea ration.",
            "The bale wire protects the components from seawater.",
        ]
    )

    def transcribe(*args, **kwargs):
        fake.audio_calls += 1
        return {
            "language": "English",
            "transcript": next(transcripts),
            "translation": None,
            "segments": [],
            "non_speech_audio": [],
        }

    def prepare(*args, **kwargs):
        return MediaInfo(18, 720, 404, True, True), [
            MediaChunk(0, 0, 10, videos[0], audios[0]),
            MediaChunk(1, 8, 10, videos[1], audios[1]),
        ]

    fake.transcribe = transcribe
    monkeypatch.setattr("gemma4_media_analyzer.pipeline.prepare_chunks", prepare)
    pipeline = MediaAnalysisPipeline(
        AnalysisConfig(output_dir=tmp_path / "out", user_context="A naval ration demonstration"),
        analyzer=fake,
    )
    _, structured = pipeline.analyze(source)
    document = json.loads(structured.read_text(encoding="utf-8"))

    assert fake.video_kwargs[0]["transcript"].startswith("Opening a 1958")
    assert fake.video_kwargs[0]["continuity"] == ""
    assert fake.video_kwargs[1]["transcript"].startswith("The bale wire")
    assert "1958 British Royal Navy sea ration" in fake.video_kwargs[1]["continuity"]
    assert "Previous visual summary (provisional): A scene" in fake.video_kwargs[1]["continuity"]
    assert fake.video_kwargs[1]["user_context"] == "A naval ration demonstration"
    assert document["settings"]["user_context"] == "A naval ration demonstration"
    assert document["settings"]["visual_context_version"] == "grounded-continuity-v1"


def test_user_context_invalidates_only_visual_cache(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    video = tmp_path / "chunk.mp4"
    video.write_bytes(b"video")
    audio = tmp_path / "chunk.wav"
    audio.write_bytes(b"audio")

    def prepare(*args, **kwargs):
        return MediaInfo(10, 720, 404, True, True), [MediaChunk(0, 0, 10, video, audio)]

    monkeypatch.setattr("gemma4_media_analyzer.pipeline.prepare_chunks", prepare)
    first = FakeAnalyzer()
    MediaAnalysisPipeline(
        AnalysisConfig(output_dir=tmp_path / "out", user_context="First context"),
        analyzer=first,
    ).analyze(source)
    second = FakeAnalyzer()
    MediaAnalysisPipeline(
        AnalysisConfig(output_dir=tmp_path / "out", user_context="Second context"),
        analyzer=second,
    ).analyze(source)

    assert first.audio_calls == 1
    assert first.video_calls == 1
    assert second.audio_calls == 0
    assert second.video_calls == 1
    assert len(list((tmp_path / "out").rglob("*-visual-context-*.json"))) == 2
