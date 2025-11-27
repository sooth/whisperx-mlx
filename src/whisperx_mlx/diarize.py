"""
Speaker diarization module for WhisperX-MLX.

This module provides speaker diarization with automatic backend selection:
- Senko (CoreML): Fast diarization on Apple Silicon (~7.7s for 1hr audio)
- Pyannote: Cross-platform fallback (slower but widely compatible)

The unified DiarizationPipeline class automatically selects the optimal
backend based on the platform and installed packages.
"""

import logging
from typing import Optional, Union, List, Dict, Any, Tuple

import numpy as np

from whisperx_mlx.diarization import load_diarization_pipeline
from whisperx_mlx.diarization.base import DiarizationSegment

logger = logging.getLogger(__name__)

# Re-export Segment as alias for backwards compatibility
Segment = DiarizationSegment


class DiarizationPipeline:
    """Unified diarization interface with automatic backend selection.

    This class wraps the backend-specific implementations and provides
    a consistent interface for speaker diarization.

    On Apple Silicon Macs with Senko installed, it uses CoreML-accelerated
    diarization for maximum performance. Otherwise, it falls back to
    pyannote-audio.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: Optional[str] = None,
        device: str = "auto",
        backend: str = "auto",
    ):
        """Initialize the diarization pipeline.

        Args:
            model_name: HuggingFace model name (used by pyannote backend)
            use_auth_token: HuggingFace authentication token (required for pyannote)
            device: Device for computation ('auto', 'cpu', 'cuda', 'mps', 'mlx')
            backend: Diarization backend: 'auto', 'senko', or 'pyannote'
        """
        self._backend = load_diarization_pipeline(
            backend=backend,
            use_auth_token=use_auth_token,
            device=device,
        )
        self._model_name = model_name
        logger.info(f"Using diarization backend: {self._backend.backend_name}")

    @property
    def backend_name(self) -> str:
        """Return the name of the active backend."""
        return self._backend.backend_name

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[List[Segment], Tuple[List[Segment], Optional[Dict[str, np.ndarray]]]]:
        """Run speaker diarization on audio.

        Args:
            audio: Path to audio file or numpy array (16kHz mono)
            min_speakers: Minimum number of speakers (optional hint)
            max_speakers: Maximum number of speakers (optional hint)
            return_embeddings: Whether to return speaker embeddings

        Returns:
            List of Segment objects with speaker labels.
            If return_embeddings is True, also returns speaker embeddings dict.
        """
        return self._backend(
            audio,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            return_embeddings=return_embeddings,
        )


def assign_word_speakers(
    diarize_segments: List[Segment],
    transcript_result: Dict[str, Any],
    speaker_embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Assign speaker labels to transcript segments based on diarization.

    This function aligns diarization results with transcription to determine
    which speaker said each segment/word.

    Args:
        diarize_segments: Segments from diarization with speaker labels
        transcript_result: Transcription result with segments
        speaker_embeddings: Optional speaker embeddings

    Returns:
        Transcript result with speaker labels added to segments
    """
    if not diarize_segments:
        logger.warning("No diarization segments, skipping speaker assignment")
        return transcript_result

    def get_speaker_at_time(time: float) -> Optional[str]:
        """Get speaker at a specific timestamp."""
        for seg in diarize_segments:
            if seg.start <= time <= seg.end:
                return seg.speaker
        return None

    def get_majority_speaker(start: float, end: float) -> Optional[str]:
        """Get the speaker who spoke most during a time range."""
        speaker_times: Dict[str, float] = {}
        for seg in diarize_segments:
            # Calculate overlap
            overlap_start = max(start, seg.start)
            overlap_end = min(end, seg.end)
            if overlap_start < overlap_end:
                duration = overlap_end - overlap_start
                speaker_times[seg.speaker] = speaker_times.get(seg.speaker, 0) + duration

        if not speaker_times:
            return None
        return max(speaker_times, key=speaker_times.get)

    # Process segments
    result = dict(transcript_result)
    segments = result.get("segments", [])

    for segment in segments:
        # Assign speaker to segment
        speaker = get_majority_speaker(segment.get("start", 0), segment.get("end", 0))
        segment["speaker"] = speaker

        # Assign speaker to words if present
        words = segment.get("words", [])
        for word in words:
            word_speaker = get_speaker_at_time((word.get("start", 0) + word.get("end", 0)) / 2)
            word["speaker"] = word_speaker or speaker

    # Add speaker embeddings if available
    if speaker_embeddings:
        result["speaker_embeddings"] = speaker_embeddings

    return result
