from gemma4_media_analyzer.render import format_timestamp, render_markdown


def test_timestamp_format_includes_hours_and_milliseconds() -> None:
    assert format_timestamp(3661.234) == "01:01:01.234"


def test_markdown_interleaves_visuals_and_speech() -> None:
    report = render_markdown(
        {
            "source": "clip.mp4",
            "media_type": "video",
            "duration_seconds": 5,
            "backend": "llama-cpp",
            "model": "gemma4-media",
            "chunks": [
                {
                    "start_seconds": 0,
                    "duration_seconds": 5,
                    "visual": {
                        "summary": "A person enters.",
                        "setting": "Room",
                        "people": ["one adult"],
                        "objects": ["door"],
                        "shots": [
                            {"start": 0, "end": 3, "description": "A door opens.", "visible_text": []}
                        ],
                    },
                    "audio": {
                        "segments": [
                            {
                                "start": 1,
                                "end": 2,
                                "speaker": "Speaker 1",
                                "text": "Hello.",
                                "translation": None,
                            }
                        ],
                        "non_speech_audio": [],
                    },
                }
            ],
        }
    )
    assert report.index("A door opens") < report.index("Hello")
    assert report.startswith("# Gemma 4 Video Analysis")
    assert "Backend: `llama-cpp`" in report


def test_markdown_falls_back_to_canonical_transcript_when_segments_are_incomplete() -> None:
    report = render_markdown(
        {
            "source": "clip.mp3",
            "media_type": "audio",
            "duration_seconds": 30,
            "backend": "llama-cpp",
            "model": "gemma4-media",
            "chunks": [
                {
                    "start_seconds": 0,
                    "duration_seconds": 30,
                    "visual": {},
                    "audio": {
                        "transcript": "First sentence. Previously unsegmented sentence.",
                        "segments": [
                            {
                                "start": 0,
                                "end": 20,
                                "speaker": "Speaker 1",
                                "text": "First sentence.",
                            }
                        ],
                        "non_speech_audio": [],
                    },
                }
            ],
        }
    )

    assert "First sentence. Previously unsegmented sentence." in report
    assert report.count("First sentence.") == 1
