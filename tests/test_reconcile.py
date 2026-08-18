from gemma4_media_analyzer.reconcile import (
    concatenate_audio_field,
    find_boundary_overlap,
    reconcile_overlapping_chunks,
)


def _audio(transcript, segments, translation=None):
    return {
        "language": "English",
        "transcript": transcript,
        "translation": translation,
        "segments": segments,
        "non_speech_audio": [],
    }


def _visual(shots=None):
    return {
        "summary": "",
        "shots": shots or [],
        "people": [],
        "objects": [],
        "setting": "",
    }


def test_fuzzy_alignment_prefers_complete_word_from_later_chunk() -> None:
    previous_count, current_count, confidence = find_boundary_overlap(
        "We need to solve this prob",
        "solve this problem today",
    )
    assert (previous_count, current_count) == (3, 3)
    assert confidence >= 0.82


def test_reconciliation_removes_previous_suffix_and_keeps_current_boundary() -> None:
    chunks = [
        {
            "start_seconds": 0,
            "duration_seconds": 30,
            "audio": _audio(
                "We need to solve this prob",
                [{"start": 0, "end": 30, "text": "We need to solve this prob", "translation": None}],
            ),
            "visual": _visual(),
        },
        {
            "start_seconds": 28,
            "duration_seconds": 30,
            "audio": _audio(
                "solve this problem today",
                [{"start": 28, "end": 35, "text": "solve this problem today", "translation": None}],
            ),
            "visual": _visual(),
        },
    ]

    result = reconcile_overlapping_chunks(chunks, 2.0)

    assert result[0]["audio"]["transcript"] == "We need to"
    assert result[1]["audio"]["transcript"] == "solve this problem today"
    assert concatenate_audio_field(result, "transcript") == "We need to solve this problem today"
    assert result[0]["audio"]["boundary_reconciliation"]["strategy"] == "text-alignment"
    assert chunks[0]["audio"]["transcript"] == "We need to solve this prob"


def test_fuzzy_conflict_keeps_prior_wording_and_trims_current_prefix() -> None:
    chunks = [
        {
            "start_seconds": 0,
            "duration_seconds": 30,
            "audio": _audio(
                "Jihad isn't a personal struggle, as apologists love to claim. No,",
                [
                    {
                        "start": 25,
                        "end": 30,
                        "text": "Jihad isn't a personal struggle, as apologists love to claim. No,",
                    }
                ],
            ),
            "visual": _visual(),
        },
        {
            "start_seconds": 28,
            "duration_seconds": 30,
            "audio": _audio(
                "polytheists love to claim. No, it's holy war.",
                [
                    {
                        "start": 28,
                        "end": 34,
                        "text": "polytheists love to claim. No, it's holy war.",
                    }
                ],
            ),
            "visual": _visual(),
        },
    ]

    result = reconcile_overlapping_chunks(chunks, 2.0)

    assert concatenate_audio_field(result, "transcript") == (
        "Jihad isn't a personal struggle, as apologists love to claim. No, it's holy war."
    )
    assert result[1]["audio"]["transcript"] == "it's holy war."
    boundary = result[0]["audio"]["boundary_reconciliation"]
    assert boundary["retained_boundary"] == "previous"
    assert boundary["removed_prefix_tokens"] == 5


def test_fuzzy_conflict_uses_later_complete_sentence_when_previous_is_partial() -> None:
    chunks = [
        {
            "start_seconds": 28,
            "duration_seconds": 30,
            "audio": _audio(
                "Why? Because multiculturalism ignores the",
                [{"start": 54, "end": 58, "text": "Why? Because multiculturalism ignores the"}],
            ),
            "visual": _visual(),
        },
        {
            "start_seconds": 56,
            "duration_seconds": 30,
            "audio": _audio(
                "because multiculturalism ignores doctrine. People object.",
                [
                    {
                        "start": 56,
                        "end": 62,
                        "text": "because multiculturalism ignores doctrine. People object.",
                    }
                ],
            ),
            "visual": _visual(),
        },
    ]

    result = reconcile_overlapping_chunks(chunks, 2.0)

    assert concatenate_audio_field(result, "transcript") == (
        "Why? because multiculturalism ignores doctrine. People object."
    )
    assert result[0]["audio"]["boundary_reconciliation"]["retained_boundary"] == "current"


def test_unmatched_audio_is_preserved_even_when_timestamps_cross_the_seam() -> None:
    chunks = [
        {
            "start_seconds": 0,
            "duration_seconds": 30,
            "audio": _audio(
                "early phrase alpha beta",
                [
                    {"start": 0, "end": 27, "text": "early phrase", "translation": None},
                    {"start": 28, "end": 30, "text": "alpha beta", "translation": None},
                ],
            ),
            "visual": _visual(
                [
                    {"start": 0, "end": 27, "description": "early", "visible_text": []},
                    {"start": 29, "end": 30, "description": "late overlap", "visible_text": []},
                ]
            ),
        },
        {
            "start_seconds": 28,
            "duration_seconds": 30,
            "audio": _audio(
                "entirely different later phrase",
                [
                    {"start": 28, "end": 29, "text": "entirely different", "translation": None},
                    {"start": 29, "end": 35, "text": "later phrase", "translation": None},
                ],
            ),
            "visual": _visual(
                [
                    {"start": 28, "end": 29, "description": "early overlap", "visible_text": []},
                    {"start": 29, "end": 35, "description": "later", "visible_text": []},
                ]
            ),
        },
    ]

    result = reconcile_overlapping_chunks(chunks, 2.0)

    assert result[0]["audio"]["transcript"] == "early phrase alpha beta"
    assert result[1]["audio"]["transcript"] == "entirely different later phrase"
    assert result[0]["audio"]["boundary_reconciliation"]["strategy"] == "preserve-unmatched"
    assert [shot["description"] for shot in result[0]["visual"]["shots"]] == ["early"]
    assert [shot["description"] for shot in result[1]["visual"]["shots"]] == ["later"]


def test_reconciliation_makes_retained_audio_monotonic_across_chunks() -> None:
    chunks = [
        {
            "start_seconds": 0,
            "duration_seconds": 30,
            "audio": _audio(
                "earlier statement",
                [{"start": 29.5, "end": 30, "text": "earlier statement"}],
            ),
            "visual": _visual(),
        },
        {
            "start_seconds": 28,
            "duration_seconds": 30,
            "audio": _audio(
                "different later statements",
                [
                    {"start": 28, "end": 32, "text": "different later"},
                    {"start": 40, "end": 44, "text": "statements"},
                ],
            ),
            "visual": _visual(),
        },
    ]

    result = reconcile_overlapping_chunks(chunks, 2.0)
    previous_start = result[0]["audio"]["segments"][-1]["start"]
    current_segments = result[1]["audio"]["segments"]

    assert current_segments[0]["start"] > previous_start
    assert [segment["start"] for segment in current_segments] == sorted(
        segment["start"] for segment in current_segments
    )
    assert current_segments[-1]["end"] <= 58
    assert result[1]["audio"]["timestamp_reconciliation"]["strategy"] == "monotonic-overlap"


def test_unmatched_transcripts_without_segments_are_preserved() -> None:
    chunks = [
        {
            "start_seconds": 0,
            "duration_seconds": 30,
            "audio": _audio("one two three four five six", []),
            "visual": _visual(),
        },
        {
            "start_seconds": 28,
            "duration_seconds": 30,
            "audio": _audio("completely unrelated opening", []),
            "visual": _visual(),
        },
    ]

    result = reconcile_overlapping_chunks(chunks, 2.0)

    assert result[0]["audio"]["transcript"] == "one two three four five six"
    assert result[1]["audio"]["transcript"] == "completely unrelated opening"
    assert result[0]["audio"]["boundary_reconciliation"]["strategy"] == "preserve-unmatched"


def test_unmatched_tail_regression_does_not_drop_complete_sentences() -> None:
    chunks = [
        {
            "start_seconds": 28,
            "duration_seconds": 30,
            "audio": _audio(
                "Love your enemies. Islam has no such pivot. Muhammad was a warrior prophet. "
                "His example is emulated today.",
                [
                    {"start": 52.1, "end": 56.2, "text": "Love your enemies.", "translation": None},
                    {
                        "start": 57.2,
                        "end": 59.3,
                        "text": "Islam has no such pivot.",
                        "translation": None,
                    },
                    {
                        "start": 60.4,
                        "end": 67.3,
                        "text": "Muhammad was a warrior prophet.",
                        "translation": None,
                    },
                    {
                        "start": 68.3,
                        "end": 71.8,
                        "text": "His example is emulated today.",
                        "translation": None,
                    },
                ],
            ),
            "visual": _visual(),
        },
        {
            "start_seconds": 56,
            "duration_seconds": 30,
            "audio": _audio(
                "The problem is emulated today. Look at ISIS.",
                [
                    {
                        "start": 56,
                        "end": 59,
                        "text": "The problem is emulated today.",
                        "translation": None,
                    },
                    {"start": 60, "end": 64, "text": "Look at ISIS.", "translation": None},
                ],
            ),
            "visual": _visual(),
        },
    ]

    result = reconcile_overlapping_chunks(chunks, 2.0)

    assert "Islam has no such pivot." in result[0]["audio"]["transcript"]
    assert "Muhammad was a warrior prophet." in result[0]["audio"]["transcript"]
