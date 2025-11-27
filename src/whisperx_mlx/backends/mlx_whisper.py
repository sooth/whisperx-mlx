"""
MLX-Whisper backend for Apple Silicon GPU acceleration.

This module provides the MLXWhisperPipeline class that wraps mlx-whisper
for GPU-accelerated transcription on Apple Silicon Macs.

Supports batched inference for significant speedup on longer audio files.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
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

        # Model caching for batched inference
        self._cached_model = None
        self._model_dtype = None

        logger.info(f"Initialized MLX-Whisper backend with model: {self._model_path}")

    @property
    def device(self) -> str:
        """Return the device this backend is using."""
        return "mlx"

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    @property
    def _model(self):
        """Lazily load and cache the Whisper model for batched inference."""
        if self._cached_model is None:
            import mlx.core as mx
            from mlx_whisper.load_models import load_model
            self._model_dtype = mx.float16
            logger.info(f"Loading MLX model for batched inference: {self._model_path}")
            self._cached_model = load_model(self._model_path, dtype=self._model_dtype)
            logger.info("MLX model loaded successfully")
        return self._cached_model

    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = 4,
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_size: int = 30,
        print_progress: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio using MLX-Whisper with batched inference.

        Args:
            audio: Path to audio file or numpy array of audio samples (16kHz mono float32)
            batch_size: Number of VAD chunks to process in parallel (default: 4)
                Higher values use more memory but are faster.
                Recommended: 4 for large models, 8-16 for smaller models.
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
        effective_batch_size = batch_size if batch_size is not None else 4

        # If we have a VAD model, use it to segment the audio
        if self.vad_model is not None:
            return self._transcribe_with_vad(
                audio,
                batch_size=effective_batch_size,
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
        batch_size: int = 4,
        language: Optional[str] = None,
        task: str = "transcribe",
        chunk_size: int = 30,
        print_progress: bool = False,
        verbose: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio with VAD preprocessing using BATCHED inference.

        Args:
            audio: Audio samples (16kHz mono float32)
            batch_size: Number of chunks to process in parallel (default: 4)
            language: Language code (None for auto-detection)
            task: 'transcribe' or 'translate'
            chunk_size: Max chunk duration for VAD merging
            print_progress: Show progress percentage
            verbose: Print transcribed text as it's generated

        Returns:
            TranscriptionResult with segments and language
        """
        import torch
        import mlx.core as mx
        from whisperx_mlx.audio import SAMPLE_RATE
        from mlx_whisper.decoding import detect_language as mlx_detect_language

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

        # Pre-compute ALL mel spectrograms
        logger.debug(f"Computing mel spectrograms for {len(merged_segments)} chunks...")
        mel_specs, chunks_metadata = self._compute_mel_batch(audio, merged_segments)

        if not mel_specs:
            return {"segments": [], "language": language or "en"}

        # Detect language from first chunk if needed
        effective_language = language
        if effective_language is None and self._model.is_multilingual:
            first_mel = mel_specs[0][None]  # Add batch dimension
            _, probs = mlx_detect_language(self._model, first_mel)
            effective_language = max(probs[0], key=probs[0].get)
            logger.info(f"Detected language: {effective_language}")
        elif effective_language is None:
            effective_language = "en"

        # Process in batches
        all_segments: List[SingleSegment] = []
        detected_language = effective_language
        total_chunks = len(mel_specs)

        logger.debug(f"Processing {total_chunks} chunks with batch_size={batch_size}")

        for batch_start in range(0, total_chunks, batch_size):
            batch_end = min(batch_start + batch_size, total_chunks)

            # Stack mel spectrograms into batch tensor
            batch_mels = mx.stack(mel_specs[batch_start:batch_end])
            batch_metadata = chunks_metadata[batch_start:batch_end]

            # Decode batch
            results = self._decode_mel_batch(
                batch_mels,
                language=effective_language,
                task=task,
            )

            # Process results
            batch_segments, _ = self._process_decode_results(results, batch_metadata)

            # Verbose output
            if verbose:
                for seg in batch_segments:
                    start_ts = self._format_timestamp(seg["start"])
                    end_ts = self._format_timestamp(seg["end"])
                    print(f"[{start_ts} --> {end_ts}] {seg['text']}")

            all_segments.extend(batch_segments)

            # Progress
            if print_progress:
                progress = (batch_end / total_chunks) * 100
                print(f"Transcription progress: {progress:.1f}%")

        return {
            "segments": all_segments,
            "language": detected_language,
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

    def _compute_mel_batch(
        self,
        audio: np.ndarray,
        merged_segments: List[Dict],
    ) -> Tuple[List[Any], List[Dict]]:
        """Pre-compute mel spectrograms for all VAD chunks.

        Args:
            audio: Full audio array (16kHz mono float32)
            merged_segments: Merged VAD segments with 'start'/'end' keys

        Returns:
            (mel_specs, chunks_metadata) where:
            - mel_specs: List of mel spectrogram arrays, each shape (80, N_FRAMES)
            - chunks_metadata: List of dicts with 'offset' and 'duration'
        """
        import mlx.core as mx
        from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
        from whisperx_mlx.audio import SAMPLE_RATE

        mel_specs = []
        chunks_metadata = []

        for seg in merged_segments:
            # Extract audio chunk
            start_sample = int(seg["start"] * SAMPLE_RATE)
            end_sample = int(seg["end"] * SAMPLE_RATE)
            audio_chunk = audio[start_sample:end_sample]

            # Skip very short segments (< 100ms)
            if len(audio_chunk) < SAMPLE_RATE * 0.1:
                continue

            # Compute mel spectrogram using mlx-whisper
            # log_mel_spectrogram returns shape (n_frames, n_mels)
            # MLX Conv1d expects (batch, length, in_channels) = (batch, n_frames, n_mels)
            mel = log_mel_spectrogram(audio_chunk, n_mels=self._model.dims.n_mels)

            # Pad/trim to N_FRAMES (3000 for 30s) on the time axis (axis=0)
            mel = pad_or_trim(mel, N_FRAMES, axis=0)

            mel_specs.append(mel)
            chunks_metadata.append({
                "offset": seg["start"],
                "duration": seg["end"] - seg["start"],
                "segment": seg,
            })

        return mel_specs, chunks_metadata

    def _decode_mel_batch(
        self,
        mel_batch: Any,  # mx.array
        language: Optional[str] = None,
        task: str = "transcribe",
    ) -> List[Any]:  # List[DecodingResult]
        """Decode a batch of mel spectrograms.

        Args:
            mel_batch: Batched mel spectrograms, shape (batch_size, 80, 3000)
            language: Language code
            task: 'transcribe' or 'translate'

        Returns:
            List of DecodingResult objects
        """
        from mlx_whisper.decoding import DecodingOptions, DecodingTask

        # Build decoding options from asr_options
        temperatures = self.asr_options.get("temperatures", (0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
        temp = temperatures[0] if isinstance(temperatures, (list, tuple)) else temperatures

        options = DecodingOptions(
            task=task,
            language=language,
            temperature=temp,
            fp16=True,
            suppress_tokens=self.asr_options.get("suppress_tokens", "-1"),
            without_timestamps=False,  # We need timestamps for segment splitting
        )

        # Run batched decode using mlx-whisper's DecodingTask
        task_runner = DecodingTask(self._model, options)
        results = task_runner.run(mel_batch)

        return results

    def _process_decode_results(
        self,
        results: List[Any],  # List[DecodingResult]
        chunks_metadata: List[Dict],
    ) -> Tuple[List[SingleSegment], Optional[str]]:
        """Process decode results into segments with corrected timestamps.

        Args:
            results: List of DecodingResult from batched decode
            chunks_metadata: Metadata with offset for each chunk

        Returns:
            (segments, detected_language)
        """
        from mlx_whisper.tokenizer import get_tokenizer

        all_segments = []
        detected_language = None

        # Time precision: 0.02 seconds per timestamp token (20ms)
        time_precision = 0.02

        for result, metadata in zip(results, chunks_metadata):
            time_offset = metadata["offset"]
            chunk_duration = metadata["duration"]

            # Get language from first result
            if detected_language is None:
                detected_language = result.language

            # Check no-speech threshold
            no_speech_threshold = self.asr_options.get("no_speech_threshold", 0.6)
            if no_speech_threshold and result.no_speech_prob > no_speech_threshold:
                continue

            # Check compression ratio threshold
            compression_threshold = self.asr_options.get("compression_ratio_threshold", 2.4)
            if compression_threshold and result.compression_ratio > compression_threshold:
                continue

            # Get tokenizer for this language
            tokenizer = get_tokenizer(
                self._model.is_multilingual,
                num_languages=self._model.num_languages,
                language=result.language,
                task="transcribe",
            )

            tokens = list(result.tokens)

            # Parse timestamp tokens to create segments
            # Timestamp tokens are >= tokenizer.timestamp_begin
            segments_in_chunk = []
            current_tokens = []
            last_timestamp = 0.0

            for token in tokens:
                if token >= tokenizer.timestamp_begin:
                    # This is a timestamp token
                    timestamp_pos = token - tokenizer.timestamp_begin
                    timestamp = timestamp_pos * time_precision

                    if current_tokens:
                        # Decode accumulated tokens and create segment
                        text = tokenizer.decode(current_tokens).strip()
                        if text:
                            segments_in_chunk.append({
                                "start": round(time_offset + last_timestamp, 3),
                                "end": round(time_offset + timestamp, 3),
                                "text": text,
                            })
                        current_tokens = []

                    last_timestamp = timestamp
                elif token < tokenizer.eot:
                    # Regular text token
                    current_tokens.append(token)

            # Handle remaining tokens
            if current_tokens:
                text = tokenizer.decode(current_tokens).strip()
                if text:
                    segments_in_chunk.append({
                        "start": round(time_offset + last_timestamp, 3),
                        "end": round(time_offset + min(chunk_duration, 30.0), 3),
                        "text": text,
                    })

            all_segments.extend(segments_in_chunk)

        return all_segments, detected_language

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
