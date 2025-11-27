"""
Silero-based Voice Activity Detection for WhisperX-MLX.

A lightweight VAD option using Silero's pre-trained models from torch.hub.
"""

import logging
from io import IOBase
from pathlib import Path
from typing import Mapping, Optional, Union, List, Dict, Any

import torch
import numpy as np

from whisperx_mlx.vads.vad import Vad, Segment as SegmentX

logger = logging.getLogger(__name__)

AudioFile = Union[str, Path, IOBase, Mapping]


class Silero(Vad):
    """Silero-based Voice Activity Detection.

    Uses the Silero VAD model from torch.hub for lightweight
    and efficient speech detection.
    """

    def __init__(
        self,
        chunk_size: float = 30,
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        **kwargs,
    ):
        """Initialize Silero VAD.

        Args:
            chunk_size: Maximum chunk duration in seconds
            vad_onset: Onset threshold for speech detection
            vad_offset: Offset threshold (not used by Silero)
        """
        super().__init__(vad_onset=vad_onset, **kwargs)
        logger.info("Initializing Silero voice activity detection...")

        self.chunk_size = chunk_size
        self.vad_offset = vad_offset

        # Load Silero VAD model from torch.hub
        self.vad_pipeline, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = vad_utils

    def __call__(self, audio: AudioFile, **kwargs) -> List[SegmentX]:
        """Run Silero VAD on audio.

        Args:
            audio: Audio data dictionary with 'waveform' and 'sample_rate'

        Returns:
            List of speech segments

        Raises:
            ValueError: If sample rate is not 16000 Hz
        """
        sample_rate = audio["sample_rate"]
        if sample_rate != 16000:
            raise ValueError("Silero VAD only supports 16000Hz sample rate")

        waveform = audio["waveform"]

        # Ensure waveform is a 1D tensor
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)
        if waveform.dim() > 1:
            waveform = waveform.squeeze()

        timestamps = self.get_speech_timestamps(
            waveform,
            model=self.vad_pipeline,
            sampling_rate=sample_rate,
            max_speech_duration_s=self.chunk_size,
            threshold=self.vad_onset,
        )

        return [
            SegmentX(ts['start'] / sample_rate, ts['end'] / sample_rate, "UNKNOWN")
            for ts in timestamps
        ]

    @staticmethod
    def preprocess_audio(audio: np.ndarray) -> np.ndarray:
        """Preprocess audio for Silero VAD.

        Silero expects raw audio samples, no preprocessing needed.

        Args:
            audio: NumPy array of audio samples

        Returns:
            Same audio array (no transformation)
        """
        return audio

    @staticmethod
    def merge_chunks(
        segments_list: List[SegmentX],
        chunk_size: float,
        onset: float = 0.5,
        offset: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Merge Silero VAD segments into chunks for transcription.

        Args:
            segments_list: List of speech segments from Silero
            chunk_size: Maximum chunk duration
            onset: Onset threshold (for compatibility)
            offset: Offset threshold (for compatibility)

        Returns:
            List of merged segment dictionaries
        """
        assert chunk_size > 0

        if not segments_list:
            logger.warning("No active speech found in audio")
            return []

        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)
