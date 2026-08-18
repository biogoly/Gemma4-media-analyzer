from gemma4_media_analyzer.continuity import MAX_CONTINUITY_CHARS, build_visual_continuity


def test_visual_continuity_is_bounded_and_does_not_carry_ocr() -> None:
    chunks = [
        {
            "audio": {"transcript": "Opening a 1958 British Royal Navy sea ration."},
            "visual": {
                "summary": "Hands open a rectangular metal ration container.",
                "shots": [
                    {
                        "description": "The metal container remains open beside a loop of wire.",
                        "visible_text": ["LUCID", "2019"],
                    }
                ],
                "objects": ["rectangular metal ration container", "loop of wire"],
                "setting": "A tabletop demonstration",
            },
        }
    ]

    continuity = build_visual_continuity(chunks)

    assert "1958 British Royal Navy sea ration" in continuity
    assert "rectangular metal ration container" in continuity
    assert "LUCID" not in continuity
    assert "provisional" in continuity
    assert len(continuity) <= MAX_CONTINUITY_CHARS

def test_visual_continuity_prefers_opening_and_recent_spoken_context() -> None:
    chunks = [
        {
            "audio": {"transcript": "Opening context " + "old " * 200},
            "visual": {"summary": "First", "shots": [], "objects": [], "setting": ""},
        },
        {
            "audio": {"transcript": "Recent evidence about bale wire."},
            "visual": {"summary": "Second", "shots": [], "objects": [], "setting": ""},
        },
    ]

    continuity = build_visual_continuity(chunks)

    assert "Opening spoken context" in continuity
    assert "Recent evidence about bale wire" in continuity
    assert len(continuity) <= MAX_CONTINUITY_CHARS
