"""Compatibility namespace for the former Gemma4-Video-Analyzer package."""

from __future__ import annotations

import sys
from importlib import import_module

from gemma4_media_analyzer import (
    AnalysisConfig,
    MediaAnalysisPipeline,
    VideoAnalysisPipeline,
    __version__,
)

__all__ = ["AnalysisConfig", "MediaAnalysisPipeline", "VideoAnalysisPipeline", "__version__"]

for _name in (
    "backend",
    "cli",
    "errors",
    "gemma",
    "media",
    "pipeline",
    "prompts",
    "reconcile",
    "render",
    "server",
):
    _module = import_module(f"gemma4_media_analyzer.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _module
    globals()[_name] = _module
