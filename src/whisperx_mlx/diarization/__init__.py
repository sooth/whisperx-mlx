"""
Diarization backend factory for WhisperX-MLX.

This module provides platform detection and automatic backend selection
for optimal diarization performance on different hardware configurations.

Backends:
- senko: CoreML-accelerated diarization (macOS Apple Silicon only, ~7.7s for 1hr audio)
- pyannote: PyTorch-based diarization (cross-platform, slower on CPU)
"""

import logging
import platform
import sys
from typing import Optional, TYPE_CHECKING

# IMPORTANT: Import torch before senko to avoid OpenMP runtime conflicts
# Both use OpenMP but initialize it differently; torch must go first
import torch  # noqa: F401

if TYPE_CHECKING:
    from whisperx_mlx.diarization.base import DiarizationBackend

logger = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac."""
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_senko_available() -> bool:
    """Check if Senko (CoreML diarization) is available."""
    if not is_apple_silicon():
        return False

    try:
        import senko
        return True
    except ImportError:
        return False


def is_pyannote_available() -> bool:
    """Check if pyannote-audio is available."""
    try:
        from pyannote.audio import Pipeline
        return True
    except ImportError:
        return False


def get_available_backends() -> list[str]:
    """Get list of available diarization backends."""
    backends = []
    if is_senko_available():
        backends.append("senko")
    if is_pyannote_available():
        backends.append("pyannote")
    return backends


def get_default_backend() -> str:
    """Determine the optimal diarization backend for the current platform.

    Returns:
        Backend name: 'senko' on Apple Silicon with senko installed,
        'pyannote' otherwise
    """
    if is_senko_available():
        logger.info("Apple Silicon detected with Senko support - using CoreML diarization")
        return "senko"

    if is_pyannote_available():
        logger.info("Using pyannote diarization backend")
        return "pyannote"

    raise ImportError(
        "No diarization backend available. Install either:\n"
        "  - senko (for Apple Silicon): pip install senko\n"
        "  - pyannote-audio (cross-platform): pip install pyannote-audio"
    )


def load_diarization_pipeline(
    backend: str = "auto",
    use_auth_token: Optional[str] = None,
    device: str = "auto",
) -> "DiarizationBackend":
    """Load a diarization pipeline with the specified backend.

    This is the main factory function for loading diarization backends.
    It automatically selects the optimal backend based on the platform
    unless explicitly specified.

    Args:
        backend: Backend to use: 'auto', 'senko', or 'pyannote'
        use_auth_token: HuggingFace token (required for pyannote)
        device: Device to use ('auto', 'cpu', 'cuda', 'mps')

    Returns:
        Configured diarization backend

    Raises:
        ValueError: If specified backend is not available
        ImportError: If no backend is available
    """
    # Determine backend
    if backend == "auto":
        backend = get_default_backend()

    if backend == "senko":
        if not is_senko_available():
            if is_apple_silicon():
                raise ImportError(
                    "Senko backend requires senko package. "
                    "Install with: pip install senko"
                )
            else:
                raise ValueError(
                    "Senko backend is only available on Apple Silicon Macs. "
                    "Use 'pyannote' backend on this platform."
                )

        from whisperx_mlx.diarization.senko_backend import SenkoDiarizationPipeline
        logger.info("Loading Senko (CoreML) diarization backend")
        return SenkoDiarizationPipeline(device=device)

    elif backend == "pyannote":
        if not is_pyannote_available():
            raise ImportError(
                "Pyannote backend requires pyannote-audio. "
                "Install with: pip install pyannote-audio"
            )

        from whisperx_mlx.diarization.pyannote_backend import PyannoteDiarizationPipeline
        logger.info("Loading pyannote diarization backend")
        return PyannoteDiarizationPipeline(
            use_auth_token=use_auth_token,
            device=device,
        )

    else:
        available = get_available_backends()
        raise ValueError(
            f"Unknown diarization backend: {backend}. "
            f"Available: {available or ['none - install senko or pyannote-audio']}"
        )


__all__ = [
    "load_diarization_pipeline",
    "is_apple_silicon",
    "is_senko_available",
    "is_pyannote_available",
    "get_available_backends",
    "get_default_backend",
]
