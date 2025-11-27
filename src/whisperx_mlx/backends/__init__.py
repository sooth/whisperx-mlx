"""
Backend factory for WhisperX-MLX.

This module provides platform detection and automatic backend selection
for optimal performance on different hardware configurations.
"""

import platform
import sys
import logging
from typing import Optional, Any, Dict

logger = logging.getLogger(__name__)


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon Mac.

    Returns:
        True if running on macOS with ARM64 (Apple Silicon)
    """
    return sys.platform == "darwin" and platform.machine() == "arm64"


def is_mlx_available() -> bool:
    """Check if MLX and mlx-whisper are available.

    Returns:
        True if mlx-whisper can be imported
    """
    if not is_apple_silicon():
        return False

    try:
        import mlx_whisper
        return True
    except ImportError:
        return False


def is_faster_whisper_available() -> bool:
    """Check if faster-whisper is available.

    Returns:
        True if faster-whisper can be imported
    """
    try:
        import faster_whisper
        return True
    except ImportError:
        return False


def get_default_backend() -> str:
    """Determine the optimal backend for the current platform.

    Returns:
        Backend name: 'mlx' on Apple Silicon with mlx-whisper,
        'faster-whisper' otherwise
    """
    if is_mlx_available():
        logger.info("Apple Silicon detected with MLX support - using MLX backend")
        return "mlx"

    if is_faster_whisper_available():
        logger.info("Using faster-whisper backend")
        return "faster-whisper"

    raise ImportError(
        "No ASR backend available. Install either:\n"
        "  - mlx-whisper (for Apple Silicon): pip install mlx-whisper\n"
        "  - faster-whisper (for CUDA/CPU): pip install faster-whisper"
    )


def get_default_device() -> str:
    """Get the default device for the current platform.

    Returns:
        Device string: 'mlx' on Apple Silicon, 'cuda' if available, else 'cpu'
    """
    if is_apple_silicon():
        return "mlx"

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass

    return "cpu"


def load_model(
    model_name: str = "large-v3",
    backend: str = "auto",
    device: Optional[str] = None,
    device_index: int = 0,
    compute_type: str = "float16",
    language: Optional[str] = None,
    vad_model: Optional[Any] = None,
    vad_method: str = "silero",
    vad_options: Optional[Dict] = None,
    asr_options: Optional[Dict] = None,
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,
):
    """Load an ASR model with the specified backend.

    This is the main factory function for loading WhisperX-MLX models.
    It automatically selects the optimal backend based on the platform
    unless explicitly specified.

    Args:
        model_name: Model name (e.g., 'large-v3', 'turbo', 'small.en')
        backend: Backend to use: 'auto', 'mlx', or 'faster-whisper'
        device: Device to use (auto-detected if None)
        device_index: GPU index for CUDA (ignored for MLX)
        compute_type: Compute precision ('float16', 'float32', 'int8')
        language: Preset language code (None for auto-detection)
        vad_model: Pre-initialized VAD model (optional)
        vad_method: VAD method if vad_model not provided: 'pyannote' or 'silero'
        vad_options: VAD configuration options
        asr_options: Additional ASR options (initial_prompt, temperatures, etc.)
        download_root: Directory to download models to
        local_files_only: Only use locally cached models
        threads: Number of CPU threads (for faster-whisper)

    Returns:
        Configured ASR backend (MLXWhisperPipeline or FasterWhisperPipeline)

    Raises:
        ValueError: If specified backend is not available
        ImportError: If no backend is available
    """
    # Determine backend
    if backend == "auto":
        backend = get_default_backend()

    # Determine device
    if device is None:
        device = get_default_device()

    # Set up default VAD options
    default_vad_options = {
        "chunk_size": 30,
        "vad_onset": 0.5,
        "vad_offset": 0.363,
    }
    if vad_options:
        default_vad_options.update(vad_options)

    # Load VAD model if not provided
    if vad_model is None:
        vad_model = _load_vad_model(
            method=vad_method,
            device=device if backend != "mlx" else "cpu",  # VAD runs on CPU for MLX
            vad_options=default_vad_options,
        )

    # Load the appropriate backend
    if backend == "mlx":
        return _load_mlx_backend(
            model_name=model_name,
            language=language,
            vad_model=vad_model,
            vad_params=default_vad_options,
            asr_options=asr_options,
        )

    elif backend == "faster-whisper":
        return _load_faster_whisper_backend(
            model_name=model_name,
            device=device,
            device_index=device_index,
            compute_type=compute_type,
            language=language,
            vad_model=vad_model,
            vad_params=default_vad_options,
            asr_options=asr_options,
            download_root=download_root,
            local_files_only=local_files_only,
            threads=threads,
        )

    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'auto', 'mlx', or 'faster-whisper'"
        )


def _load_mlx_backend(
    model_name: str,
    language: Optional[str],
    vad_model: Optional[Any],
    vad_params: Dict,
    asr_options: Optional[Dict],
):
    """Load the MLX-Whisper backend."""
    if not is_mlx_available():
        raise ImportError(
            "MLX backend requires mlx-whisper. Install with: pip install mlx-whisper"
        )

    from whisperx_mlx.backends.mlx_whisper import MLXWhisperPipeline

    return MLXWhisperPipeline(
        model_name=model_name,
        vad_model=vad_model,
        vad_params=vad_params,
        language=language,
        asr_options=asr_options,
    )


def _load_faster_whisper_backend(
    model_name: str,
    device: str,
    device_index: int,
    compute_type: str,
    language: Optional[str],
    vad_model: Optional[Any],
    vad_params: Dict,
    asr_options: Optional[Dict],
    download_root: Optional[str],
    local_files_only: bool,
    threads: int,
):
    """Load the faster-whisper backend."""
    if not is_faster_whisper_available():
        raise ImportError(
            "faster-whisper backend requires faster-whisper. "
            "Install with: pip install faster-whisper"
        )

    from whisperx_mlx.backends.faster_whisper import FasterWhisperPipeline

    return FasterWhisperPipeline(
        model_name=model_name,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        vad_model=vad_model,
        vad_params=vad_params,
        language=language,
        asr_options=asr_options,
        download_root=download_root,
        local_files_only=local_files_only,
        threads=threads,
    )


def _load_vad_model(
    method: str,
    device: str,
    vad_options: Dict,
) -> Optional[Any]:
    """Load a VAD model.

    Args:
        method: 'pyannote' or 'silero'
        device: Device for the VAD model
        vad_options: VAD configuration

    Returns:
        VAD model instance or None if loading fails
    """
    try:
        if method == "pyannote":
            from whisperx_mlx.vads.pyannote import Pyannote
            import torch

            # Determine device
            if device == "mlx":
                vad_device = "cpu"  # VAD runs on CPU when using MLX
            elif device.startswith("cuda"):
                vad_device = device
            else:
                vad_device = "cpu"

            return Pyannote(
                device=torch.device(vad_device),
                vad_onset=vad_options.get("vad_onset", 0.5),
                vad_offset=vad_options.get("vad_offset", 0.363),
            )

        elif method == "silero":
            from whisperx_mlx.vads.silero import Silero
            return Silero(**vad_options)

        else:
            logger.warning(f"Unknown VAD method: {method}, proceeding without VAD")
            return None

    except Exception as e:
        logger.warning(f"Failed to load VAD model ({method}): {e}")
        return None


# Export key functions and classes
__all__ = [
    "load_model",
    "is_apple_silicon",
    "is_mlx_available",
    "is_faster_whisper_available",
    "get_default_backend",
    "get_default_device",
]
