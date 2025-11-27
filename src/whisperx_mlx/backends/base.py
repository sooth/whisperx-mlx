"""
Abstract base class for ASR backends.

This module defines the interface that all ASR backends must implement,
enabling WhisperX-MLX to support multiple backends (MLX, faster-whisper, etc.)
with a unified API.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any
import numpy as np

from whisperx_mlx.schema import TranscriptionResult


class ASRBackend(ABC):
    """Abstract base class for ASR (Automatic Speech Recognition) backends.

    All backend implementations must inherit from this class and implement
    the required methods to ensure compatibility with the WhisperX-MLX pipeline.
    """

    @abstractmethod
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_size: int = 30,
        print_progress: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio to text with timestamps.

        Args:
            audio: Path to audio file or numpy array of audio samples (16kHz mono float32)
            batch_size: Number of segments to process in parallel (backend-specific)
            language: Language code (e.g., 'en', 'fr'). None for auto-detection.
            task: Either 'transcribe' or 'translate' (to English)
            chunk_size: Maximum duration (seconds) of audio chunks for VAD segmentation
            print_progress: Whether to print progress during transcription
            verbose: Whether to print detailed output

        Returns:
            TranscriptionResult containing segments with text and timestamps
        """
        pass

    @abstractmethod
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the language of the audio.

        Args:
            audio: Numpy array of audio samples (16kHz mono float32)

        Returns:
            ISO 639-1 language code (e.g., 'en', 'fr', 'de')
        """
        pass

    @property
    @abstractmethod
    def device(self) -> str:
        """Return the device this backend is using (e.g., 'cpu', 'cuda', 'mps', 'mlx')."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the name/identifier of the loaded model."""
        pass


class VADInterface(ABC):
    """Abstract interface for Voice Activity Detection models."""

    @abstractmethod
    def __call__(self, audio_data: Dict[str, Any]) -> List[Dict[str, float]]:
        """Run VAD on audio data.

        Args:
            audio_data: Dictionary containing 'waveform' and 'sample_rate'

        Returns:
            List of segments, each with 'start' and 'end' timestamps in seconds
        """
        pass

    @staticmethod
    @abstractmethod
    def merge_chunks(
        segments: List[Dict[str, float]],
        chunk_size: float,
        onset: float = 0.5,
        offset: float = 0.363,
    ) -> List[Dict[str, float]]:
        """Merge VAD segments into larger chunks.

        Args:
            segments: List of VAD segments with 'start' and 'end' keys
            chunk_size: Maximum duration of merged chunks in seconds
            onset: VAD onset threshold
            offset: VAD offset threshold

        Returns:
            List of merged segments
        """
        pass
