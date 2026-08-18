import pytest

from gemma4_media_analyzer.errors import ModelOutputError
from gemma4_media_analyzer.gemma import (
    extract_json_object,
    validate_audio_result,
    validate_visual_result,
)


def test_extract_json_object_tolerates_wrapper_and_braces_in_strings() -> None:
    text = '<|channel>final<|message>```json\n{"text":"a } brace", "items": []}\n```'
    assert extract_json_object(text) == {"text": "a } brace", "items": []}


def test_extract_json_object_rejects_incomplete_output() -> None:
    with pytest.raises(ModelOutputError):
        extract_json_object('{"items": [1]')


def test_audio_validation_converts_relative_to_absolute_timestamps() -> None:
    result = validate_audio_result(
        {
            "language": "English",
            "transcript": "Hello",
            "translation": None,
            "segments": [
                {"start": 1, "end": 2.5, "speaker": "Speaker 1", "text": "Hello", "translation": None}
            ],
            "non_speech_audio": ["door closes"],
        },
        start_seconds=30,
    )
    assert result["segments"][0]["start"] == 31
    assert result["segments"][0]["end"] == 32.5


def test_audio_validation_scales_overflow_and_preserves_statement_order() -> None:
    result = validate_audio_result(
        {
            "language": "English",
            "transcript": "First. Second. Third.",
            "translation": None,
            "segments": [
                {"start": 0, "end": 5, "text": "First."},
                {"start": 25, "end": 45, "text": "Second."},
                {"start": 10, "end": 12, "text": "Third."},
            ],
            "non_speech_audio": [],
        },
        start_seconds=28,
        duration_seconds=30,
    )

    starts = [segment["start"] for segment in result["segments"]]
    assert starts == sorted(starts)
    assert starts[0] == 28
    assert result["segments"][1]["end"] == 58
    assert all(28 <= segment["start"] <= segment["end"] <= 58 for segment in result["segments"])


def test_audio_validation_normalizes_cached_absolute_timestamps() -> None:
    result = validate_audio_result(
        {
            "segments": [
                {"start": 28, "end": 32, "speaker": "Speaker 1", "text": "First"},
                {"start": 68, "end": 72, "speaker": "Speaker 1", "text": "Second"},
            ]
        },
        start_seconds=28,
        duration_seconds=30,
        timestamps_are_absolute=True,
    )

    assert result["segments"][0]["start"] == 28
    assert result["segments"][-1]["end"] == 58


def test_audio_validation_restores_unsegmented_transcript_tail() -> None:
    result = validate_audio_result(
        {
            "language": "English",
            "transcript": (
                "They command violence against unbelievers. "
                "Jihad isn't some metaphor for personal struggle."
            ),
            "segments": [
                {
                    "start": 26,
                    "end": 30,
                    "speaker": "Speaker 1",
                    "text": "They command violence against unbelievers.",
                }
            ],
        },
        start_seconds=0,
        duration_seconds=30,
    )

    assert [segment["text"] for segment in result["segments"]] == [
        "They command violence against unbelievers.",
        "Jihad isn't some metaphor for personal struggle.",
    ]
    assert result["segments"][-1]["inferred_timestamp"] is True
    assert result["segment_coverage"]["status"] == "repaired-tail"
    assert result["segment_coverage"]["complete"] is True


def test_audio_validation_synthesizes_segment_for_complete_transcript() -> None:
    result = validate_audio_result(
        {"transcript": "Speech without timestamped segments.", "segments": []},
        start_seconds=28,
        duration_seconds=30,
    )

    assert result["segments"] == [
        {
            "start": 28,
            "end": 58,
            "speaker": "Speaker",
            "text": "Speech without timestamped segments.",
            "translation": None,
            "inferred_timestamp": True,
        }
    ]
    assert result["segment_coverage"]["status"] == "synthesized"


def test_audio_validation_marks_complex_segment_disagreement_for_fallback() -> None:
    result = validate_audio_result(
        {
            "transcript": "One omitted phrase then three.",
            "segments": [{"start": 0, "end": 5, "text": "One then three."}],
        },
        start_seconds=0,
        duration_seconds=10,
    )

    assert result["segment_coverage"] == {
        "version": "canonical-transcript-v1",
        "status": "canonical-fallback",
        "complete": False,
    }


def test_visual_validation_normalizes_visible_text() -> None:
    result = validate_visual_result(
        {
            "summary": "A title card",
            "shots": [{"start": 0, "end": 1, "description": "Card", "visible_text": "HELLO"}],
            "people": [],
            "objects": [],
            "setting": "Studio",
        },
        start_seconds=10,
    )
    assert result["shots"][0]["visible_text"] == ["HELLO"]
    assert result["shots"][0]["start"] == 10


def test_visual_validation_bounds_overflow_to_chunk_duration() -> None:
    result = validate_visual_result(
        {
            "summary": "A scene",
            "shots": [
                {"start": 0, "end": 20, "description": "First", "visible_text": []},
                {"start": 30, "end": 45, "description": "Second", "visible_text": []},
            ],
            "people": [],
            "objects": [],
            "setting": "Room",
        },
        start_seconds=56,
        duration_seconds=30,
    )

    assert result["shots"][-1]["end"] == 86
    assert all(56 <= shot["start"] <= shot["end"] <= 86 for shot in result["shots"])


def test_visual_validation_enforces_schema_bounds_and_deduplicates_ocr() -> None:
    raw = {
        "summary": "s" * 600,
        "shots": [
            {
                "start": index,
                "end": index + 1,
                "description": "d" * 400,
                "visible_text": ["Repeat", "repeat"] + [f"Text {item}" for item in range(8)],
            }
            for index in range(12)
        ],
        "people": [f"Person {index}" for index in range(15)],
        "objects": [f"Object {index}" for index in range(30)],
        "setting": "x" * 400,
    }

    result = validate_visual_result(raw, start_seconds=0)

    assert len(result["summary"]) == 500
    assert len(result["shots"]) == 10
    assert len(result["shots"][0]["description"]) == 360
    assert result["shots"][0]["visible_text"] == [
        "Repeat",
        "Text 0",
        "Text 1",
        "Text 2",
        "Text 3",
        "Text 4",
    ]
    assert result["shots"][1]["visible_text"] == ["Text 5", "Text 6", "Text 7"]
    assert len(result["people"]) == 12
    assert len(result["objects"]) == 24
    assert len(result["setting"]) == 360
