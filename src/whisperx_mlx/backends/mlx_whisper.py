"""
MLX-Whisper backend for Apple Silicon GPU acceleration.

This module provides the MLXWhisperPipeline class that wraps mlx-whisper
for GPU-accelerated transcription on Apple Silicon Macs.
"""

from typing import Optional, Union, List, Dict, Any
import numpy as np
import logging

from whisperx_mlx.backends.base import ASRBackend
from whisperx_mlx.schema import TranscriptionResult, SingleSegment

logger = logging.getLogger(__name__)

# Model name mapping from WhisperX names to mlx-community HuggingFace repos
# See: https://huggingface.co/collections/mlx-community/whisper-663256f9964fbb1177db93dc
MLX_MODEL_MAPPING = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "tiny.en": "mlx-community/whisper-tiny.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "base.en": "mlx-community/whisper-base.en-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "medium.en": "mlx-community/whisper-medium.en-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "large-v1": "mlx-community/whisper-large-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "distil-large-v3": "mlx-community/distil-whisper-large-v3",
}


def get_mlx_model_path(model_name: str) -> str:
    """Convert a model name to the mlx-community HuggingFace repo path.

    Args:
        model_name: Short model name (e.g., 'large-v3') or full HF path

    Returns:
        Full HuggingFace model path for mlx-whisper
    """
    # If it's already a full path, return as-is
    if "/" in model_name:
        return model_name

    # Look up in mapping
    if model_name in MLX_MODEL_MAPPING:
        return MLX_MODEL_MAPPING[model_name]

    # Try with mlx-community prefix
    return f"mlx-community/whisper-{model_name}"


class MLXWhisperPipeline(ASRBackend):
    """MLX-Whisper backend for Apple Silicon GPU acceleration.

    This backend uses mlx-whisper for GPU-accelerated transcription on Apple Silicon.
    It integrates with WhisperX's VAD preprocessing and output normalization.

    Attributes:
        model_path: HuggingFace path to the MLX Whisper model
        vad_model: Voice Activity Detection model for preprocessing
        language: Preset language code (None for auto-detection)
        asr_options: Additional options passed to mlx-whisper
    """

    def __init__(
        self,
        model_name: str,
        vad_model: Optional[Any] = None,
        vad_params: Optional[Dict] = None,
        language: Optional[str] = None,
        asr_options: Optional[Dict] = None,
    ):
        """Initialize the MLX-Whisper pipeline.

        Args:
            model_name: Model name (e.g., 'large-v3') or full HF path
            vad_model: VAD model instance for speech detection
            vad_params: VAD parameters (chunk_size, vad_onset, vad_offset)
            language: Preset language code (None for auto-detection)
            asr_options: Additional options for transcription
        """
        # Lazy import mlx_whisper to avoid import errors on non-Apple platforms
        try:
            import mlx_whisper
            self._mlx_whisper = mlx_whisper
        except ImportError as e:
            raise ImportError(
                "mlx-whisper is required for the MLX backend. "
                "Install it with: pip install mlx-whisper"
            ) from e

        self._model_path = get_mlx_model_path(model_name)
        self._model_name = model_name
        self.vad_model = vad_model
        self._vad_params = vad_params or {
            "chunk_size": 30,
            "vad_onset": 0.5,
            "vad_offset": 0.363,
        }
        self.preset_language = language
        self.asr_options = asr_options or {}

        logger.info(f"Initialized MLX-Whisper backend with model: {self._model_path}")

    @property
    def device(self) -> str:
        """Return the device this backend is using."""
        return "mlx"

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

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
        """Transcribe audio using MLX-Whisper.

        Args:
            audio: Path to audio file or numpy array of audio samples (16kHz mono float32)
            batch_size: Not used by mlx-whisper (for API compatibility)
            language: Language code (e.g., 'en'). None for auto-detection.
            task: 'transcribe' or 'translate' (to English)
            chunk_size: Maximum duration (seconds) of audio chunks
            print_progress: Whether to print progress
            verbose: Whether to print detailed output

        Returns:
            TranscriptionResult with segments and detected language
        """
        # Load audio if path is provided
        if isinstance(audio, str):
            from whisperx_mlx.audio import load_audio
            audio = load_audio(audio)

        # Determine language
        effective_language = language or self.preset_language

        # If we have a VAD model, use it to segment the audio
        if self.vad_model is not None:
            return self._transcribe_with_vad(
                audio,
                language=effective_language,
                task=task,
                chunk_size=chunk_size,
                print_progress=print_progress,
                verbose=verbose,
            )

        # Otherwise, transcribe the entire audio directly
        return self._transcribe_direct(
            audio,
            language=effective_language,
            task=task,
            verbose=verbose,
        )

    def _transcribe_direct(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio directly without VAD preprocessing."""
        result = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            language=language,
            task=task,
            verbose=verbose,
            initial_prompt=self.asr_options.get("initial_prompt"),
            temperature=self.asr_options.get("temperatures", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
            compression_ratio_threshold=self.asr_options.get("compression_ratio_threshold", 2.4),
            logprob_threshold=self.asr_options.get("log_prob_threshold", -1.0),
            no_speech_threshold=self.asr_options.get("no_speech_threshold", 0.6),
            condition_on_previous_text=self.asr_options.get("condition_on_previous_text", True),
        )

        # Convert mlx-whisper output to WhisperX format
        segments = self._convert_segments(result.get("segments", []))
        detected_language = result.get("language", language or "en")

        return {
            "segments": segments,
            "language": detected_language,
        }

    def _transcribe_with_vad(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_size: int = 30,
        print_progress: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio with VAD preprocessing."""
        import torch
        from whisperx_mlx.audio import SAMPLE_RATE

        # Preprocess audio for VAD
        if hasattr(self.vad_model, 'preprocess_audio'):
            waveform = self.vad_model.preprocess_audio(audio)
        else:
            waveform = torch.from_numpy(audio).unsqueeze(0)

        # Run VAD
        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})

        # Merge VAD segments into chunks
        if hasattr(self.vad_model, 'merge_chunks'):
            merged_segments = self.vad_model.merge_chunks(
                vad_segments,
                chunk_size,
                onset=self._vad_params.get("vad_onset", 0.5),
                offset=self._vad_params.get("vad_offset", 0.363),
            )
        else:
            # Fallback: treat entire audio as one segment
            merged_segments = [{"start": 0, "end": len(audio) / SAMPLE_RATE}]

        if not merged_segments:
            logger.warning("No speech detected in audio")
            return {"segments": [], "language": language or "en"}

        # Transcribe each VAD segment
        all_segments: List[SingleSegment] = []
        detected_language = language

        for idx, seg in enumerate(merged_segments):
            # Extract audio chunk
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = int(seg["end"] * SAMPLE_RATE)
            audio_chunk = audio[start_sample:end_sample]

            if len(audio_chunk) < SAMPLE_RATE * 0.1:  # Skip very short segments
                continue

            # Transcribe chunk
            # NOTE: Never pass verbose=True to mlx_whisper when processing chunks,
            # because it prints timestamps relative to the chunk, not the full audio.
            # We handle verbose output ourselves with corrected timestamps below.
            result = self._mlx_whisper.transcribe(
                audio_chunk,
                path_or_hf_repo=self._model_path,
                language=language,
                task=task,
                verbose=False,  # Always False - we print with correct offsets below
                initial_prompt=self.asr_options.get("initial_prompt"),
                temperature=self.asr_options.get("temperatures", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)),
                compression_ratio_threshold=self.asr_options.get("compression_ratio_threshold", 2.4),
                logprob_threshold=self.asr_options.get("log_prob_threshold", -1.0),
                no_speech_threshold=self.asr_options.get("no_speech_threshold", 0.6),
            )

            # Use detected language from first chunk
            if detected_language is None and "language" in result:
                detected_language = result["language"]

            # Convert and offset timestamps
            chunk_segments = self._convert_segments(
                result.get("segments", []),
                time_offset=seg["start"]
            )
            all_segments.extend(chunk_segments)

            # Print verbose output with corrected timestamps
            if verbose:
                for s in chunk_segments:
                    start_ts = self._format_timestamp(s["start"])
                    end_ts = self._format_timestamp(s["end"])
                    print(f"[{start_ts} --> {end_ts}] {s['text']}")

            if print_progress:
                progress = ((idx + 1) / len(merged_segments)) * 100
                print(f"Transcription progress: {progress:.1f}%")

        return {
            "segments": all_segments,
            "language": detected_language or "en",
        }

    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS.mmm timestamp."""
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins:02d}:{secs:06.3f}"

    def _convert_segments(
        self,
        mlx_segments: List[Dict],
        time_offset: float = 0.0
    ) -> List[SingleSegment]:
        """Convert mlx-whisper segments to WhisperX format.

        Args:
            mlx_segments: Segments from mlx-whisper
            time_offset: Offset to add to timestamps (for VAD chunks)

        Returns:
            List of SingleSegment dictionaries
        """
        segments = []
        for seg in mlx_segments:
            text = seg.get("text", "").strip()
            if not text:
                continue

            segments.append({
                "start": round(seg.get("start", 0) + time_offset, 3),
                "end": round(seg.get("end", 0) + time_offset, 3),
                "text": text,
            })

        return segments

    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the language of the audio.

        Args:
            audio: Numpy array of audio samples (16kHz mono float32)

        Returns:
            ISO 639-1 language code
        """
        from whisperx_mlx.audio import SAMPLE_RATE, N_SAMPLES

        # Use first 30 seconds for language detection
        audio_sample = audio[:N_SAMPLES] if len(audio) > N_SAMPLES else audio

        result = self._mlx_whisper.transcribe(
            audio_sample,
            path_or_hf_repo=self._model_path,
        )

        return result.get("language", "en")


def load_mlx_model(
    model_name: str = "large-v3",
    language: Optional[str] = None,
    vad_model: Optional[Any] = None,
    vad_params: Optional[Dict] = None,
    asr_options: Optional[Dict] = None,
) -> MLXWhisperPipeline:
    """Load an MLX-Whisper model.

    Args:
        model_name: Model name (e.g., 'large-v3', 'turbo') or full HF path
        language: Preset language code (None for auto-detection)
        vad_model: VAD model for speech detection
        vad_params: VAD parameters
        asr_options: Additional ASR options

    Returns:
        Configured MLXWhisperPipeline instance
    """
    return MLXWhisperPipeline(
        model_name=model_name,
        vad_model=vad_model,
        vad_params=vad_params,
        language=language,
        asr_options=asr_options,
    )
