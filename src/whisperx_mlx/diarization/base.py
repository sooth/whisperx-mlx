"""
Base class for diarization backends.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Tuple

import numpy as np


@dataclass
class DiarizationSegment:
    """A segment with speaker information."""
    start: float
    end: float
    speaker: str


class DiarizationBackend(ABC):
    """Abstract base class for diarization backends."""

    @abstractmethod
    def __call__(
        self,
        audio: Union[str, np.ndarray],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[List[DiarizationSegment], Tuple[List[DiarizationSegment], Optional[Dict[str, np.ndarray]]]]:
        """Run speaker diarization on audio.

        Args:
            audio: Path to audio file or numpy array (16kHz mono)
            min_speakers: Minimum number of speakers (optional hint)
            max_speakers: Maximum number of speakers (optional hint)
            return_embeddings: Whether to return speaker embeddings

        Returns:
            List of DiarizationSegment objects with speaker labels.
            If return_embeddings is True, also returns speaker embeddings dict.
        """
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend."""
        pass


def normalize_speaker_ids(segments: List[DiarizationSegment]) -> List[DiarizationSegment]:
    """Normalize speaker IDs to SPEAKER_00, SPEAKER_01, etc. format.

    Args:
        segments: List of diarization segments

    Returns:
        Segments with normalized speaker IDs
    """
    if not segments:
        return segments

    # Get unique speakers in order of appearance
    seen_speakers = {}
    for seg in segments:
        if seg.speaker not in seen_speakers:
            seen_speakers[seg.speaker] = f"SPEAKER_{len(seen_speakers):02d}"

    # Create new segments with normalized IDs
    return [
        DiarizationSegment(
            start=seg.start,
            end=seg.end,
            speaker=seen_speakers[seg.speaker],
        )
        for seg in segments
    ]
