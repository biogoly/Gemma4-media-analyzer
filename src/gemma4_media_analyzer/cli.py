"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import MediaAnalyzerError
from .pipeline import AnalysisConfig, MediaAnalysisPipeline
from .server import BackendError

DEFAULT_MODELS = {
    "transformers": "google/gemma-4-12B-it",
    "llama-cpp": "gemma4-media",
    "ollama": "gemma4:12b-it-qat",
}


def _media_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("media", type=Path, help="Input audio or video file")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("analysis-output"))
    parser.add_argument("--chunk-seconds", type=float, default=30.0, help="Chunk length (maximum 30)")
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=2.0,
        help="Seconds shared by adjacent chunks for boundary-safe transcription (default: 2)",
    )
    parser.add_argument("--fps", type=float, default=1.0, help="Video sampling rate sent to Gemma")
    parser.add_argument("--max-edge", type=int, default=720, help="Longest video edge in pixels")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gemma4-media",
        description="Analyze audio or video of arbitrary length with Gemma 4.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    analyze = commands.add_parser("analyze", help="Prepare, analyze, and concatenate audio or video")
    _media_arguments(analyze)
    analyze.add_argument(
        "--backend",
        choices=sorted(DEFAULT_MODELS),
        default="transformers",
        help="Inference runtime (llama-cpp is recommended for GGUF)",
    )
    analyze.add_argument("--model", help="Model/repository name; defaults depend on backend")
    analyze.add_argument("--source-language", help="Expected spoken language, or omit for detection")
    analyze.add_argument("--translate-to", help="Translate the complete transcript to this language")
    analyze.add_argument(
        "--context",
        dest="user_context",
        help="Authoritative background for grounding visual identity across chunks",
    )
    analyze.add_argument("--device-map", default="auto", help="Transformers backend device mapping")
    analyze.add_argument("--dtype", default="auto", help="Transformers backend tensor dtype")
    analyze.add_argument("--max-new-tokens", type=int, default=4096)
    analyze.add_argument("--seed", type=int, default=7)
    analyze.add_argument("--server-url", help="llama.cpp or Ollama base URL")
    analyze.add_argument("--api-key-env", help="Environment variable containing a server bearer token")
    analyze.add_argument("--server-timeout", type=float, default=3600.0, help="HTTP timeout in seconds")
    analyze.add_argument("--no-stream", action="store_true", help="Request one non-streaming response")
    analyze.add_argument(
        "--media-mode",
        choices=("base64", "file-url"),
        default="base64",
        help="llama.cpp media transport; file-url avoids base64 copies",
    )
    analyze.add_argument(
        "--media-root",
        type=Path,
        help="Root exposed by llama-server --media-path (defaults to output-dir)",
    )
    analyze.add_argument("--server-context-size", type=int, default=16384)
    analyze.add_argument("--keep-alive", default="30m", help="Ollama model keep-alive duration")

    prepare = commands.add_parser("prepare", help="Only create the normalized media chunks")
    _media_arguments(prepare)
    return parser


def _config(args: argparse.Namespace) -> AnalysisConfig:
    backend = getattr(args, "backend", "transformers")
    return AnalysisConfig(
        output_dir=args.output_dir.resolve(),
        backend=backend,
        model_id=getattr(args, "model", None) or DEFAULT_MODELS[backend],
        chunk_seconds=args.chunk_seconds,
        chunk_overlap_seconds=args.chunk_overlap,
        fps=args.fps,
        max_edge=args.max_edge,
        source_language=getattr(args, "source_language", None),
        translate_to=getattr(args, "translate_to", None),
        user_context=getattr(args, "user_context", None),
        device_map=getattr(args, "device_map", "auto"),
        dtype=getattr(args, "dtype", "auto"),
        max_new_tokens=getattr(args, "max_new_tokens", 4096),
        seed=getattr(args, "seed", 7),
        server_url=getattr(args, "server_url", None),
        api_key_env=getattr(args, "api_key_env", None),
        server_timeout_seconds=getattr(args, "server_timeout", 3600.0),
        server_stream=not getattr(args, "no_stream", False),
        media_mode=getattr(args, "media_mode", "base64"),
        media_root=getattr(args, "media_root", None),
        server_context_size=getattr(args, "server_context_size", 16384),
        keep_alive=getattr(args, "keep_alive", "30m"),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        pipeline = MediaAnalysisPipeline(_config(args), progress=lambda message: print(message, flush=True))
        if args.command == "prepare":
            info, chunks, workspace = pipeline.prepare(args.media)
            print(
                f"Prepared {len(chunks)} chunks for {info.duration_seconds:.2f}s of {info.media_type} in {workspace}",
                flush=True,
            )
            return 0
        if args.backend == "ollama":
            print(
                "Warning: Ollama audio/video transport is experimental; use llama-cpp for the supported GGUF path.",
                file=sys.stderr,
            )
        markdown_path, json_path = pipeline.analyze(args.media)
        print(f"Markdown: {markdown_path}")
        print(f"JSON: {json_path}")
        return 0
    except (MediaAnalyzerError, BackendError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
