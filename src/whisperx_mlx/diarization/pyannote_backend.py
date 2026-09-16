"""
Pyannote diarization backend for WhisperX-MLX.

Pyannote-audio provides PyTorch-based speaker diarization that works
cross-platform but is slower than CoreML on Apple Silicon.

See: https://github.com/pyannote/pyannote-audio
"""

import logging
import os
from typing import Optional, Union, List, Dict, Tuple

# pyannote.audio 4.x sends OpenTelemetry usage data to otel.pyannote.ai by
# default. Disable unless the user has explicitly opted in; must be set
# before pyannote.audio is imported.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")

import numpy as np
import torch

from whisperx_mlx.diarization.base import (
    DiarizationBackend,
    DiarizationSegment,
    normalize_speaker_ids,
)

logger = logging.getLogger(__name__)

PYANNOTE_SAMPLE_RATE = 16000


def load_waveform(audio: Union[str, np.ndarray], target_sr: int = PYANNOTE_SAMPLE_RATE) -> Dict[str, object]:
    """Load audio into the ``{"waveform", "sample_rate"}`` dict pyannote 4.x accepts.

    pyannote-audio 4.x decodes file paths through torchcodec, which requires
    an FFmpeg shared-library build matching the torchcodec wheel. That pairing
    is fragile (Homebrew FFmpeg 8 vs. torchcodec expecting 4-7, torch/torchcodec
    version skew, etc.). Passing an in-memory waveform sidesteps torchcodec
    entirely, so we decode here with soundfile if present, else scipy.

    Returns a mono float32 tensor of shape (1, time) at ``target_sr``.
    """
    if isinstance(audio, np.ndarray):
        data = np.asarray(audio, dtype=np.float32)
        sr = target_sr  # numpy input is documented as 16 kHz mono
    else:
        try:
            import soundfile as sf
            data, sr = sf.read(audio, dtype="float32", always_2d=False)
        except ImportError:
            from scipy.io import wavfile
            sr, data = wavfile.read(audio)
            if data.dtype.kind == "i":
                data = data.astype(np.float32) / float(np.iinfo(data.dtype).max)
            elif data.dtype.kind == "u":  # 8-bit unsigned PCM
                data = (data.astype(np.float32) - 128.0) / 128.0
            else:
                data = data.astype(np.float32)

    # Downmix to mono: (time, ch) -> (time,)
    if data.ndim == 2:
        data = data.mean(axis=1)

    if sr != target_sr:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(target_sr))
        data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        sr = target_sr

    waveform = torch.from_numpy(np.ascontiguousarray(data)).unsqueeze(0)  # (1, time)
    return {"waveform": waveform, "sample_rate": sr}


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
            # pyannote doesn't support MLX; use Apple GPU via MPS if available
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._device = device
        self.pipeline = None

        try:
            from pyannote.audio import Pipeline

            self.pipeline = Pipeline.from_pretrained(
                model_name,
                token=use_auth_token,
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

        audio_label = audio if isinstance(audio, str) else f"<ndarray {getattr(audio, 'shape', '?')}>"

        try:
            # Prepare kwargs for diarization
            kwargs = {}
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

            # Decode to an in-memory waveform so pyannote never touches torchcodec.
            audio_input = load_waveform(audio)
            audio_input["waveform"] = audio_input["waveform"].to(torch.device(self._device))

            # Run diarization
            logger.debug(
                f"Running pyannote diarization on {audio_label} "
                f"({audio_input['waveform'].shape[-1] / audio_input['sample_rate']:.1f}s, in-memory waveform)"
            )
            diarization = self.pipeline(audio_input, **kwargs)

            # pyannote 4.x wraps the Annotation in an output object; unwrap it
            if not hasattr(diarization, "itertracks"):
                for attr in ("speaker_diarization", "annotation", "diarization"):
                    inner = getattr(diarization, attr, None)
                    if inner is not None and hasattr(inner, "itertracks"):
                        diarization = inner
                        break
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
                embeddings = self._extract_embeddings(audio, segments)
                return segments, embeddings

            return segments

        except Exception as e:
            logger.error(f"Pyannote diarization failed: {e}", exc_info=True)
            return [] if not return_embeddings else ([], None)

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
