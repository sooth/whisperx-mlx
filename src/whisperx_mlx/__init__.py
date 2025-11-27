"""
WhisperX-MLX: Speech recognition with MLX acceleration for Apple Silicon.

This package provides WhisperX functionality with native Apple Silicon GPU
acceleration through MLX, while maintaining compatibility with faster-whisper
for CUDA/CPU platforms.
"""

__version__ = "0.1.0"

from whisperx_mlx.transcribe import transcribe
from whisperx_mlx.alignment import load_align_model, align
from whisperx_mlx.diarize import DiarizationPipeline, assign_word_speakers

__all__ = [
    "transcribe",
    "load_align_model",
    "align",
    "DiarizationPipeline",
    "assign_word_speakers",
]
