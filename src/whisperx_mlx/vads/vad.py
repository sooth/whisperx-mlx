"""
Base VAD class for WhisperX-MLX.

Defines the interface and common merge_chunks functionality for VAD models.
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass


@dataclass
class Segment:
    """A speech segment with start/end times and optional speaker."""
    start: float
    end: float
    speaker: Optional[str] = None


class Vad:
    """Base class for Voice Activity Detection models.

    All VAD implementations should inherit from this class and implement
    the __call__ method for speech detection.
    """

    def __init__(self, vad_onset: float = 0.5, **kwargs):
        """Initialize the VAD model.

        Args:
            vad_onset: Onset threshold (decimal between 0 and 1)
        """
        if not (0 < vad_onset < 1):
            raise ValueError("vad_onset must be a decimal value between 0 and 1.")
        self.vad_onset = vad_onset

    def __call__(self, audio_data: Dict[str, Any]) -> Any:
        """Run VAD on audio data.

        Args:
            audio_data: Dictionary with 'waveform' and 'sample_rate'

        Returns:
            VAD output (format depends on implementation)
        """
        raise NotImplementedError

    @staticmethod
    def preprocess_audio(audio):
        """Preprocess audio for VAD. Override in subclasses."""
        return audio

    @staticmethod
    def merge_chunks(
        segments: List[Segment],
        chunk_size: float,
        onset: float = 0.5,
        offset: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Merge VAD segments into larger chunks for transcription.

        This implements the merge operation described in the WhisperX paper,
        combining adjacent speech segments while respecting the maximum chunk size.

        Args:
            segments: List of Segment objects from VAD
            chunk_size: Maximum chunk duration in seconds
            onset: VAD onset threshold
            offset: VAD offset threshold (unused in base implementation)

        Returns:
            List of merged segment dictionaries with 'start', 'end', and 'segments' keys
        """
        if not segments:
            return []

        curr_end = 0
        merged_segments = []
        seg_idxs: List[tuple] = []
        speaker_idxs: List[Optional[str]] = []

        curr_start = segments[0].start

        for seg in segments:
            if seg.end - curr_start > chunk_size and curr_end - curr_start > 0:
                merged_segments.append({
                    "start": curr_start,
                    "end": curr_end,
                    "segments": seg_idxs,
                })
                curr_start = seg.start
                seg_idxs = []
                speaker_idxs = []

            curr_end = seg.end
            seg_idxs.append((seg.start, seg.end))
            speaker_idxs.append(seg.speaker)

        # Add final segment
        merged_segments.append({
            "start": curr_start,
            "end": curr_end,
            "segments": seg_idxs,
        })

        return merged_segments
