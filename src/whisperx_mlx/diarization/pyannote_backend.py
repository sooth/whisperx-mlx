"""
Pyannote diarization backend for WhisperX-MLX.

Pyannote-audio provides PyTorch-based speaker diarization that works
cross-platform but is slower than CoreML on Apple Silicon.

See: https://github.com/pyannote/pyannote-audio
"""

import logging
from typing import Optional, Union, List, Dict, Tuple

import numpy as np
import torch

from whisperx_mlx.diarization.base import (
    DiarizationBackend,
    DiarizationSegment,
    normalize_speaker_ids,
)

logger = logging.getLogger(__name__)


class PyannoteDiarizationPipeline(DiarizationBackend):
    """Speaker diarization pipeline using pyannote-audio.

    This pipeline identifies different speakers in an audio recording
    and segments the audio by speaker.
    """

    def __init__(
        self,
        model_name: str = "pyannote/speaker-diarization-3.1",
        use_auth_token: Optional[str] = None,
        device: str = "auto",
    ):
        """Initialize the pyannote diarization pipeline.

        Args:
            model_name: HuggingFace model name for diarization
            use_auth_token: HuggingFace authentication token (required)
            device: Device for computation ('auto', 'cpu', 'cuda', 'mps')
        """
        self.model_name = model_name
        self._use_auth_token = use_auth_token

        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                # MPS can be unstable with pyannote, prefer CPU on Mac
                device = "cpu"
            else:
                device = "cpu"
        elif device == "mlx":
            # pyannote doesn't support MLX, fall back to CPU
            device = "cpu"

        self._device = device
        self.pipeline = None

        try:
            from pyannote.audio import Pipeline

            self.pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=use_auth_token,
            )
            self.pipeline.to(torch.device(device))
            logger.info(f"Loaded pyannote diarization model: {model_name} on {device}")

        except Exception as e:
            logger.warning(
                f"Could not load diarization model: {e}\n"
                "Diarization requires pyannote-audio and a HuggingFace token.\n"
                "Install with: pip install pyannote-audio\n"
                "Get token from: https://huggingface.co/pyannote/speaker-diarization-3.1"
            )
            self.pipeline = None

    @property
    def backend_name(self) -> str:
        return "pyannote"

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[List[DiarizationSegment], Tuple[List[DiarizationSegment], Optional[Dict[str, np.ndarray]]]]:
        """Run speaker diarization on audio using pyannote.

        Args:
            audio: Path to audio file or numpy array (16kHz mono)
            min_speakers: Minimum number of speakers (optional hint)
            max_speakers: Maximum number of speakers (optional hint)
            return_embeddings: Whether to return speaker embeddings

        Returns:
            List of DiarizationSegment objects with speaker labels.
            If return_embeddings is True, also returns speaker embeddings dict.
        """
        if self.pipeline is None:
            logger.warning("Diarization model not loaded, returning empty segments")
            return [] if not return_embeddings else ([], None)

        # Handle numpy array input - need to save to temp file
        temp_file = None
        if isinstance(audio, np.ndarray):
            import tempfile
            import soundfile as sf
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio, 16000)
            audio_input = temp_file.name
        else:
            audio_input = audio

        try:
            # Prepare kwargs for diarization
            kwargs = {}
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

            # Run diarization
            logger.debug(f"Running pyannote diarization on {audio_input}")
            diarization = self.pipeline(audio_input, **kwargs)

            # Convert to segments
            segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                segments.append(DiarizationSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=speaker,
                ))

            # Normalize speaker IDs
            segments = normalize_speaker_ids(segments)

            if return_embeddings:
                embeddings = self._extract_embeddings(audio_input, segments)
                return segments, embeddings

            return segments

        except Exception as e:
            logger.error(f"Pyannote diarization failed: {e}")
            return [] if not return_embeddings else ([], None)

        finally:
            # Clean up temp file
            if temp_file is not None:
                import os
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    def _extract_embeddings(
        self,
        audio: Union[str, np.ndarray],
        segments: List[DiarizationSegment],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract speaker embeddings for each unique speaker.

        Args:
            audio: Audio data
            segments: Diarization segments

        Returns:
            Dictionary mapping speaker ID to embedding vector
        """
        if not segments:
            return None

        # Placeholder - full implementation would use a speaker embedding model
        unique_speakers = set(seg.speaker for seg in segments)
        embeddings = {
            speaker: np.zeros(256)  # Placeholder embedding
            for speaker in unique_speakers
        }
        return embeddings
