"""Project-specific exception hierarchy."""


class MediaAnalyzerError(RuntimeError):
    """Base class for expected, user-facing failures."""


VideoAnalyzerError = MediaAnalyzerError


class ConfigurationError(MediaAnalyzerError):
    """Raised for invalid configuration or missing prerequisites."""


class ExternalToolError(MediaAnalyzerError):
    """Raised when FFmpeg or ffprobe fails."""


class ModelOutputError(MediaAnalyzerError):
    """Raised when a model response cannot be parsed or validated."""
