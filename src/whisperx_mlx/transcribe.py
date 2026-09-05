"""
Main transcription pipeline for WhisperX-MLX.

This module provides the high-level transcribe() function that orchestrates
the full transcription workflow: VAD preprocessing, ASR, and optional
alignment/diarization.
"""

import gc
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Optional, Union, List, Dict, Any, Tuple

import numpy as np
import torch

from whisperx_mlx.audio import load_audio, SAMPLE_RATE
from whisperx_mlx.backends import load_model, get_default_device
from whisperx_mlx.schema import TranscriptionResult, AlignedTranscriptionResult

logger = logging.getLogger(__name__)


# Global executor for background model loading
_model_loader_executor: Optional[ThreadPoolExecutor] = None
_cache_lock = threading.Lock()
_ASR_CACHE: Dict[tuple, Any] = {}
_ALIGN_CACHE: Dict[tuple, Tuple[Any, dict]] = {}


def _get_model_loader_executor() -> ThreadPoolExecutor:
    """Get or create the background model loader executor."""
    global _model_loader_executor
    if _model_loader_executor is None:
        _model_loader_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="model_loader")
    return _model_loader_executor


def _freeze(value: Any) -> Any:
    """Hashable snapshot of asr_options / nested configs for the model cache key."""
    if isinstance(value, dict):
        return tuple(sorted((str(k), _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _asr_cache_key(
    model: Optional[str],
    backend: str,
    device: Optional[str],
    device_index: int,
    compute_type: str,
    language: Optional[str],
    task: str,
    vad_method: str,
    vad_onset: float,
    vad_offset: float,
    chunk_size: int,
    asr_options: Dict[str, Any],
) -> tuple:
    return (
        model,
        backend,
        device,
        device_index,
        compute_type,
        language,
        task,
        vad_method,
        round(float(vad_onset), 6),
        round(float(vad_offset), 6),
        chunk_size,
        _freeze(asr_options),
    )


def _get_cached_asr(
    *,
    model: Optional[str],
    backend: str,
    device: Optional[str],
    device_index: int,
    compute_type: str,
    language: Optional[str],
    vad_method: str,
    vad_onset: float,
    vad_offset: float,
    chunk_size: int,
    asr_options: Dict[str, Any],
):
    key = _asr_cache_key(
        model,
        backend,
        device,
        device_index,
        compute_type,
        language,
        "transcribe",
        vad_method,
        vad_onset,
        vad_offset,
        chunk_size,
        asr_options,
    )
    with _cache_lock:
        hit = _ASR_CACHE.get(key)
    if hit is not None:
        return hit
    logger.info(f"Loading model: {model} (backend: {backend})")
    pipe = load_model(
        model_name=model,
        backend=backend,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        language=language,
        vad_method=vad_method,
        vad_options={
            "chunk_size": chunk_size,
            "vad_onset": vad_onset,
            "vad_offset": vad_offset,
        },
        asr_options=asr_options,
    )
    with _cache_lock:
        return _ASR_CACHE.setdefault(key, pipe)


def _get_cached_align_model(language_code: str, device: str, model_name: Optional[str]):
    from whisperx_mlx.alignment import load_align_model

    key = (language_code, model_name, device)
    with _cache_lock:
        hit = _ALIGN_CACHE.get(key)
    if hit is not None:
        return hit
    logger.info(f"Loading alignment model for language: {language_code}")
    align_model_obj, align_metadata = load_align_model(
        language_code,
        device,
        model_name=model_name,
    )
    if align_model_obj is not None and hasattr(align_model_obj, "eval"):
        align_model_obj.eval()
    packed = (align_model_obj, align_metadata)
    if align_model_obj is None:
        return packed
    with _cache_lock:
        return _ALIGN_CACHE.setdefault(key, packed)


def _load_diarization_model_background(
    hf_token: Optional[str],
    device: str,
    backend: str,
) -> "DiarizationPipeline":
    """Load diarization model in background thread."""
    from whisperx_mlx.diarize import DiarizationPipeline
    logger.debug("Background: Starting diarization model load...")
    model = DiarizationPipeline(
        use_auth_token=hf_token,
        device=device,
        backend=backend,
    )
    logger.debug("Background: Diarization model loaded!")
    return model


def transcribe(
    audio: Union[str, np.ndarray],
    model: Optional[str] = "large-v3",
    backend: str = "auto",
    device: Optional[str] = None,
    device_index: int = 0,
    compute_type: str = "float16",
    batch_size: int = 16,
    chunk_size: int = 30,
    language: Optional[str] = None,
    task: str = "transcribe",
    vad_method: str = "silero",
    vad_onset: float = 0.5,
    vad_offset: float = 0.363,
    asr_options: Optional[Dict[str, Any]] = None,
    print_progress: bool = False,
    verbose: bool = False,
) -> TranscriptionResult:
    """Transcribe audio to text with timestamps.

    This is the main entry point for transcription. It handles:
    - Loading the audio file (if a path is provided)
    - Running VAD to detect speech regions
    - Transcribing each region using the selected backend
    - Returning normalized results

    Args:
        audio: Path to audio file or numpy array of audio samples (16kHz mono)
        model: Model name (e.g., 'large-v3', 'turbo', 'small.en')
        backend: Backend to use: 'auto', 'mlx', or 'faster-whisper'
        device: Device to use (auto-detected if None)
        device_index: GPU index for CUDA (ignored for MLX)
        compute_type: Compute precision ('float16', 'float32', 'int8')
        batch_size: Batch size for processing (backend-specific)
        chunk_size: Maximum audio chunk duration in seconds
        language: Language code (None for auto-detection)
        task: 'transcribe' or 'translate' (to English)
        vad_method: VAD method: 'pyannote' or 'silero'
        vad_onset: VAD onset threshold
        vad_offset: VAD offset threshold
        asr_options: Additional ASR options (initial_prompt, temperatures, etc.)
        print_progress: Whether to print progress during transcription
        verbose: Whether to print detailed output

    Returns:
        TranscriptionResult with segments and detected language

    Example:
        >>> result = transcribe("audio.mp3", model="large-v3")
        >>> for segment in result["segments"]:
        ...     print(f"{segment['start']:.2f} - {segment['end']:.2f}: {segment['text']}")
    """
    # Determine device
    if device is None:
        device = get_default_device()

    # Load audio if path is provided. Copy ndarrays so timed calls always
    # pay Silero (no identity-based VAD skip) while models stay cached.
    if isinstance(audio, str):
        logger.info(f"Loading audio: {audio}")
        audio_data = load_audio(audio)
    else:
        audio_data = np.array(audio, copy=True, dtype=np.float32)

    # Set up default ASR options
    default_asr_options = {
        "temperatures": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "compression_ratio_threshold": 2.4,
        "log_prob_threshold": -1.0,
        "no_speech_threshold": 0.6,
        "condition_on_previous_text": True,
    }
    if asr_options:
        default_asr_options.update(asr_options)

    asr_model = _get_cached_asr(
        model=model,
        backend=backend,
        device=device,
        device_index=device_index,
        compute_type=compute_type,
        language=language,
        vad_method=vad_method,
        vad_onset=vad_onset,
        vad_offset=vad_offset,
        chunk_size=chunk_size,
        asr_options=default_asr_options,
    )

    # Transcribe
    logger.info("Performing transcription...")
    result = asr_model.transcribe(
        audio_data,
        batch_size=batch_size,
        language=language,
        task=task,
        chunk_size=chunk_size,
        print_progress=print_progress,
        verbose=verbose,
    )

    return result


def transcribe_with_alignment(
    audio: Union[str, np.ndarray],
    model: str = "large-v3",
    align_model: Optional[str] = None,
    backend: str = "auto",
    device: Optional[str] = None,
    language: Optional[str] = None,
    **kwargs,
) -> AlignedTranscriptionResult:
    """Transcribe audio with word-level alignment.

    This performs transcription followed by forced alignment using wav2vec2
    to get precise word-level timestamps.

    Args:
        audio: Path to audio file or numpy array
        model: Whisper model name
        align_model: Alignment model name (auto-selected if None)
        backend: Backend to use
        device: Device to use
        language: Language code (affects alignment model selection)
        **kwargs: Additional arguments passed to transcribe()

    Returns:
        AlignedTranscriptionResult with word-level timestamps
    """
    from whisperx_mlx.alignment import align

    # Determine device
    if device is None:
        device = get_default_device()

    # Load audio if needed
    if isinstance(audio, str):
        audio_data = load_audio(audio)
    else:
        audio_data = np.array(audio, copy=True, dtype=np.float32)

    # First, transcribe (ASR pipeline is process-cached). MLX wav2vec2 must
    # be constructed on this thread — a worker-thread load has no GPU stream
    # for later mx.eval.
    result = transcribe(
        audio_data,
        model=model,
        backend=backend,
        device=device,
        language=language,
        **kwargs,
    )

    if not result.get("segments"):
        logger.warning("No segments to align")
        return {"segments": [], "word_segments": []}

    # Determine alignment language
    align_language = language or result.get("language", "en")

    align_model_obj, align_metadata = _get_cached_align_model(
        align_language, device, align_model
    )

    if align_model_obj is None:
        logger.warning(f"No alignment model available for {align_language}")
        return {"segments": result["segments"], "word_segments": []}

    # Perform alignment
    logger.info("Performing alignment...")
    aligned_result = align(
        result["segments"],
        align_model_obj,
        align_metadata,
        audio_data,
        device,
    )

    return aligned_result


def transcribe_with_diarization(
    audio: Union[str, np.ndarray],
    model: str = "large-v3",
    hf_token: Optional[str] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    diarization_backend: str = "auto",
    backend: str = "auto",
    device: Optional[str] = None,
    align: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """Transcribe audio with speaker diarization.

    This performs transcription, optional alignment, and speaker diarization
    to identify who said what.

    The diarization model is loaded in a background thread while transcription
    and alignment are running, significantly reducing total processing time.

    Args:
        audio: Path to audio file or numpy array
        model: Whisper model name
        hf_token: HuggingFace token for diarization model (required for pyannote)
        min_speakers: Minimum number of speakers (optional hint)
        max_speakers: Maximum number of speakers (optional hint)
        diarization_backend: Diarization backend: 'auto', 'senko', or 'pyannote'
            'auto' uses Senko (CoreML) on Apple Silicon, pyannote elsewhere
        backend: ASR backend to use
        device: Device to use
        align: Whether to perform alignment before diarization
        **kwargs: Additional arguments passed to transcribe()

    Returns:
        Transcription result with speaker labels assigned to segments
    """
    from whisperx_mlx.diarize import assign_word_speakers

    # Determine device
    if device is None:
        device = get_default_device()

    # Start loading diarization model in background thread IMMEDIATELY
    # This runs in parallel with transcription and alignment
    logger.info("Starting diarization model load in background...")
    executor = _get_model_loader_executor()
    diarize_model_future: Future = executor.submit(
        _load_diarization_model_background,
        hf_token,
        device,
        diarization_backend,
    )

    # Load audio
    if isinstance(audio, str):
        audio_path = audio
        audio_data = load_audio(audio)
    else:
        audio_path = None
        audio_data = audio

    # Transcribe (with or without alignment)
    # This runs while diarization model loads in background
    if align:
        result = transcribe_with_alignment(
            audio_data,
            model=model,
            backend=backend,
            device=device,
            **kwargs,
        )
    else:
        result = transcribe(
            audio_data,
            model=model,
            backend=backend,
            device=device,
            **kwargs,
        )

    if not result.get("segments"):
        logger.warning("No segments to diarize")
        # Cancel the background load if possible
        diarize_model_future.cancel()
        return result

    # Wait for diarization model to finish loading
    # By now, it should be ready (or close to ready) since transcription takes ~7-11s
    logger.info("Waiting for diarization model...")
    diarize_model = diarize_model_future.result()
    logger.info(f"Performing diarization (backend: {diarization_backend})...")

    # Diarization needs the audio path or we need to save temporarily
    if audio_path:
        diarize_input = audio_path
    else:
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio_data, SAMPLE_RATE)
            diarize_input = f.name

    diarize_segments = diarize_model(
        diarize_input,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    # Assign speakers to segments
    result = assign_word_speakers(diarize_segments, result)

    # Cleanup
    del diarize_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
