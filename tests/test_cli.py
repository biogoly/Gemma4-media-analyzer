from gemma4_media_analyzer.cli import _config, build_parser


def test_analyze_context_is_added_to_configuration(tmp_path) -> None:
    args = build_parser().parse_args(
        [
            "analyze",
            str(tmp_path / "clip.mp4"),
            "--output-dir",
            str(tmp_path / "out"),
            "--context",
            "A naval ration demonstration",
        ]
    )

    assert _config(args).user_context == "A naval ration demonstration"
