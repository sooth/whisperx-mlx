"""
Senko (CoreML) diarization backend for WhisperX-MLX.

Senko provides CoreML-accelerated speaker diarization on Apple Silicon,
achieving ~7.7 seconds for 1 hour of audio on M3.

See: https://github.com/narcotic-sh/senko
"""

import logging
import tempfile
from typing import Optional, Union, List, Dict, Tuple

import numpy as np

from whisperx_mlx.diarization.base import (
    DiarizationBackend,
    DiarizationSegment,
    normalize_speaker_ids,
)

logger = logging.getLogger(__name__)


class SenkoDiarizationPipeline(DiarizationBackend):
    """CoreML-accelerated diarization using Senko.

    Senko uses Apple's CoreML and Neural Engine for fast speaker diarization
    on macOS with Apple Silicon.
    """

    def __init__(self, device: str = "auto"):
        """Initialize the Senko diarization pipeline.

        Args:
            device: Device for computation. Senko supports 'auto', 'cpu', 'mps'.
                    'auto' uses CoreML with ANE when available.
        """
        try:
            import senko
        except ImportError:
            raise ImportError(
                "Senko is required for CoreML diarization. "
                "Install with: pip install senko"
            )

        # Map device names
        if device == "mlx":
            device = "auto"  # Senko uses 'auto' for best device selection
        elif device == "cuda":
            logger.warning("CUDA not supported by Senko, falling back to 'auto'")
            device = "auto"

        logger.info(f"Initializing Senko diarizer with device={device}")
        # Note: warmup=False to avoid OMP runtime conflicts when torch is loaded
        # The first diarization call will be slightly slower but avoids crashes
        self.diarizer = senko.Diarizer(device=device, warmup=False)
        self._device = device

    @property
    def backend_name(self) -> str:
        return "senko"

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[List[DiarizationSegment], Tuple[List[DiarizationSegment], Optional[Dict[str, np.ndarray]]]]:
        """Run speaker diarization on audio using Senko.

        Args:
            audio: Path to audio file or numpy array (16kHz mono)
            min_speakers: Minimum number of speakers (passed to Senko)
            max_speakers: Maximum number of speakers (passed to Senko)
            return_embeddings: Whether to return speaker embeddings

        Returns:
            List of DiarizationSegment objects with speaker labels.
            If return_embeddings is True, also returns speaker embeddings dict.
        """
        # Senko requires 16kHz mono WAV - convert if needed
        temp_file = None
        if isinstance(audio, np.ndarray):
            import soundfile as sf
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sf.write(temp_file.name, audio, 16000, subtype='PCM_16')
            audio_path = temp_file.name
        else:
            # Convert file to 16kHz mono if needed
            audio_path = self._convert_audio_for_senko(audio)

        try:
            # Build kwargs for Senko diarize call
            kwargs = {}
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

            # Run diarization
            logger.debug(f"Running Senko diarization on {audio_path}")
            result = self.diarizer.diarize(audio_path, **kwargs)

            # Convert Senko output to our segment format
            # Senko returns: {"merged_segments": [{"speaker_id": 0, "start": 0.0, "end": 2.5}, ...]}
            segments = self._convert_senko_output(result)

            # Normalize speaker IDs to SPEAKER_00 format
            segments = normalize_speaker_ids(segments)

            if return_embeddings:
                # Senko doesn't expose embeddings directly
                # Return None for embeddings
                embeddings = self._extract_embeddings(segments)
                return segments, embeddings

            return segments

        except Exception as e:
            logger.error(f"Senko diarization failed: {e}")
            if return_embeddings:
                return [], None
            return []

        finally:
            # Clean up temp files
            import os
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass
            # Clean up conversion temp file
            if hasattr(self, '_temp_audio_file') and self._temp_audio_file:
                try:
                    os.unlink(self._temp_audio_file)
                except OSError:
                    pass
                self._temp_audio_file = None

    def _convert_audio_for_senko(self, audio_path: str) -> str:
        """Convert audio file to 16kHz mono WAV format required by Senko.

        Args:
            audio_path: Path to input audio file

        Returns:
            Path to converted audio file (temp file if conversion needed)
        """
        import soundfile as sf

        # Read audio file
        try:
            audio_data, sample_rate = sf.read(audio_path)
        except Exception as e:
            logger.warning(f"Could not read audio with soundfile: {e}, trying torchaudio")
            import torchaudio
            waveform, sample_rate = torchaudio.load(audio_path)
            audio_data = waveform.numpy().T  # (channels, samples) -> (samples, channels)

        # Check if conversion is needed
        needs_conversion = False

        # Convert to mono if stereo
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
            needs_conversion = True
        elif audio_data.ndim > 1:
            audio_data = audio_data.flatten()
            needs_conversion = True

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            import torch
            import torchaudio.functional as F
            waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
            audio_data = F.resample(waveform, sample_rate, 16000).squeeze().numpy()
            needs_conversion = True

        if needs_conversion:
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio_data, 16000, subtype='PCM_16')
            self._temp_audio_file = temp_file.name  # Track for cleanup
            return temp_file.name

        return audio_path

    def _convert_senko_output(self, result: dict) -> List[DiarizationSegment]:
        """Convert Senko output format to DiarizationSegment list.

        Senko returns:
            {"merged_segments": [{"speaker_id": 0, "start": 0.0, "end": 2.5}, ...]}

        We convert to:
            [DiarizationSegment(start=0.0, end=2.5, speaker="0"), ...]
        """
        segments = []

        # Handle different possible output formats from Senko
        if "merged_segments" in result:
            raw_segments = result["merged_segments"]
        elif "segments" in result:
            raw_segments = result["segments"]
        else:
            logger.warning(f"Unexpected Senko output format: {result.keys()}")
            raw_segments = []

        for seg in raw_segments:
            # Senko uses speaker_id (int) or speaker (str)
            speaker_id = seg.get("speaker_id", seg.get("speaker", 0))
            segments.append(DiarizationSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=str(speaker_id),
            ))

        return segments

    def _extract_embeddings(
        self,
        segments: List[DiarizationSegment],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract speaker embeddings.

        Note: Senko doesn't expose speaker embeddings directly.
        This returns placeholder embeddings for API compatibility.
        """
        if not segments:
            return None

        # Return placeholder embeddings (Senko doesn't expose these)
        unique_speakers = set(seg.speaker for seg in segments)
        return {
            speaker: np.zeros(256)  # Placeholder embedding
            for speaker in unique_speakers
        }
