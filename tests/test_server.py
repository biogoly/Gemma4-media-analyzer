import json

import pytest

from gemma4_media_analyzer.server import (
    AUDIO_RESPONSE_SCHEMA,
    VISUAL_RESPONSE_SCHEMA,
    BackendError,
    LlamaCppAnalyzer,
    _JsonHttpClient,
    parse_llama_sse,
    parse_ollama_ndjson,
)


def test_llama_sse_accumulates_streamed_content() -> None:
    lines = [
        b'data: {"choices":[{"delta":{"content":"{\\"a\\":"}}]}\n',
        b'data: {"choices":[{"delta":{"content":"1}"}}]}\n',
        b"data: [DONE]\n",
    ]
    assert parse_llama_sse(lines) == '{"a":1}'


def test_ollama_ndjson_accumulates_streamed_content() -> None:
    lines = [
        b'{"message":{"content":"hello "},"done":false}\n',
        b'{"message":{"content":"world"},"done":true}\n',
    ]
    assert parse_ollama_ndjson(lines) == "hello world"


def test_llama_audio_places_prompt_before_audio(tmp_path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"RIFF-test")
    analyzer = LlamaCppAnalyzer(model_id="gemma4-media")
    captured = {}

    def generate(messages, schema):
        captured["messages"] = messages
        captured["schema"] = schema
        return json.dumps(
            {
                "language": "English",
                "transcript": "",
                "translation": None,
                "segments": [],
                "non_speech_audio": [],
            }
        )

    analyzer._generate = generate
    analyzer.transcribe(
        audio,
        start_seconds=0,
        duration_seconds=12.5,
        source_language=None,
        translate_to=None,
    )
    content = captured["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "input_audio"]
    assert "12.500 seconds" in content[0]["text"]
    assert captured["schema"] == AUDIO_RESPONSE_SCHEMA


def test_llama_video_uses_ordered_sampled_images(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    frames = (tmp_path / "frame-1.jpg", tmp_path / "frame-2.jpg")
    for frame in frames:
        frame.write_bytes(b"jpeg")
    analyzer = LlamaCppAnalyzer(model_id="gemma4-media")
    captured = {}

    def generate(messages, schema):
        captured["messages"] = messages
        captured["schema"] = schema
        return json.dumps(
            {"summary": "", "shots": [], "people": [], "objects": [], "setting": ""}
        )

    analyzer._generate = generate
    analyzer.describe(
        video,
        start_seconds=0,
        duration_seconds=2,
        frame_paths=frames,
        transcript="The speaker calls this a naval ration container.",
        continuity="A rectangular metal container was opened in the previous segment.",
        user_context="A 1958 Royal Navy ration demonstration",
    )

    content = captured["messages"][1]["content"]
    assert [item["type"] for item in content] == ["image_url", "image_url", "text"]
    assert all(item["image_url"]["url"].startswith("data:image/jpeg;base64,") for item in content[:2])
    assert "2 supplied images" in content[-1]["text"]
    assert "Authoritative user context" in content[-1]["text"]
    assert "Current segment audio transcript" in content[-1]["text"]
    assert "Earlier-chunk continuity record" in content[-1]["text"]
    assert "omit uncertain characters" in content[-1]["text"]
    assert captured["schema"] == VISUAL_RESPONSE_SCHEMA


def test_visual_schema_bounds_repetitive_shots_and_ocr() -> None:
    shots = VISUAL_RESPONSE_SCHEMA["properties"]["shots"]
    shot_properties = shots["items"]["properties"]

    assert shots["maxItems"] == 10
    assert shot_properties["description"]["maxLength"] == 360
    assert shot_properties["visible_text"]["maxItems"] == 6
    assert shot_properties["visible_text"]["items"]["maxLength"] == 160


def test_llama_video_retries_unterminated_json_with_compact_prompt(tmp_path) -> None:
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"mp4")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"jpeg")
    analyzer = LlamaCppAnalyzer(model_id="gemma4-media")
    calls = []

    def generate(messages, schema, *, max_tokens=None):
        calls.append((messages, schema, max_tokens))
        if len(calls) == 1:
            return '{"summary": "unfinished"'
        return json.dumps(
            {"summary": "done", "shots": [], "people": [], "objects": [], "setting": ""}
        )

    analyzer._generate = generate
    result = analyzer.describe(
        video,
        start_seconds=0,
        duration_seconds=2,
        frame_paths=(frame,),
    )

    assert result["summary"] == "done"
    assert len(calls) == 2
    assert calls[0][2] is None
    assert calls[1][2] == 6144
    assert "previous attempt did not finish" in calls[1][0][1]["content"][-1]["text"]


def test_inference_timeout_is_not_retried() -> None:
    calls = 0

    def opener(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise TimeoutError("timed out")

    client = _JsonHttpClient(base_url="http://127.0.0.1:8080", api_key=None, timeout_seconds=1, opener=opener)

    with pytest.raises(BackendError, match="timed out"):
        client.post("/v1/chat/completions", {}, lambda response: response.read())

    assert calls == 1
