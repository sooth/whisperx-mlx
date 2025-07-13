"""
WhisperX ASR Module - MLX Backend Only

This module provides the main ASR interface for WhisperX using only the MLX backend.
Supports both standard and batch-optimized processing.
"""
import os
import platform
from typing import Optional, Union, Dict, Any, List

import numpy as np
import torch

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, SingleSegment
from whisperx.vads import Vad, Silero, Pyannote


class MLXWhisperPipeline:
    """
    Pipeline wrapper for MLX Whisper model to maintain API compatibility.
    """
    
    def __init__(self, backend, vad_model=None):
        self.backend = backend
        self.vad_model = vad_model
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: int = 8,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio using the MLX backend.
        
        If VAD is enabled, audio will be segmented first and then transcribed.
        """
        # Load audio if path provided
        if isinstance(audio, str):
            audio = load_audio(audio)
        
        # If no VAD, transcribe directly
        if self.vad_model is None:
            # Check if backend supports align_words parameter
            if hasattr(self.backend, '_align_words') and kwargs.get('word_timestamps', False):
                kwargs['align_words'] = True
                kwargs.pop('word_timestamps', None)
            return self.backend.transcribe(
                audio,
                batch_size=batch_size,
                num_workers=0,
                print_progress=print_progress,
                combined_progress=combined_progress,
                verbose=verbose,
                **kwargs
            )
        
        # Segment audio with VAD
        segments = self._segment_audio_with_vad(audio, chunk_size)
        
        # Process segments
        if hasattr(self.backend, 'transcribe_batch') and batch_size > 1:
            # Use batch processing if available
            # Add audio to each segment for batch processing
            for segment in segments:
                start_sample = int(segment['start'] * SAMPLE_RATE)
                end_sample = int(segment['end'] * SAMPLE_RATE)
                segment['audio'] = audio[start_sample:end_sample]
            
            # Check if backend supports align_words parameter
            if hasattr(self.backend, '_align_words') and kwargs.get('word_timestamps', False):
                kwargs['align_words'] = True
                kwargs.pop('word_timestamps', None)
            
            return self.backend.transcribe_batch(
                segments,
                batch_size=batch_size,
                print_progress=print_progress,
                combined_progress=combined_progress,
                verbose=verbose,
                **kwargs
            )
        else:
            # Standard sequential processing
            all_segments = []
            language = None
            
            for segment in segments:
                start_sample = int(segment['start'] * SAMPLE_RATE)
                end_sample = int(segment['end'] * SAMPLE_RATE)
                segment_audio = audio[start_sample:end_sample]
                
                # Transcribe segment
                result = self.backend.transcribe(
                    segment_audio,
                    batch_size=1,
                    print_progress=False,
                    verbose=verbose,
                    **kwargs
                )
                
                # Extract language from first segment
                if language is None and result.get('language'):
                    language = result['language']
                
                # Adjust timestamps and add segments
                for seg in result.get('segments', []):
                    seg['start'] += segment['start']
                    seg['end'] += segment['start']
                    all_segments.append(seg)
            
            return {
                'segments': all_segments,
                'language': language or 'en'
            }
    
    def _segment_audio_with_vad(self, audio: np.ndarray, chunk_size: int) -> List[Dict]:
        """Segment audio using VAD."""
        # Pre-process audio for VAD
        if issubclass(type(self.vad_model), Vad):
            waveform = self.vad_model.preprocess_audio(audio)
            merge_chunks = self.vad_model.merge_chunks
        else:
            waveform = Pyannote.preprocess_audio(audio)
            merge_chunks = Pyannote.merge_chunks
        
        # Get VAD segments
        vad_segments = self.vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        
        # Merge chunks
        vad_segments = merge_chunks(
            vad_segments,
            chunk_size,
            onset=self.vad_model.vad_onset if hasattr(self.vad_model, 'vad_onset') else 0.5,
            offset=self.vad_model.vad_offset if hasattr(self.vad_model, 'vad_offset') else 0.363,
        )
        
        return vad_segments
    
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect language of audio."""
        return self.backend.detect_language(audio)


def load_model(
    whisper_arch: str,
    device: str = "cpu",
    device_index: int = 0,
    compute_type: str = "float16",
    asr_options: Optional[dict] = None,
    language: Optional[str] = None,
    vad_method: str = "pyannote",
    vad_options: Optional[dict] = None,
    task: str = "transcribe",
    download_root: Optional[str] = None,
    local_files_only: bool = False,
    threads: int = 4,
    backend: str = "auto",
    batch_size: int = 8,
    **kwargs
) -> MLXWhisperPipeline:
    """
    Load a Whisper model for inference with MLX backend.
    
    Args:
        whisper_arch: The name of the Whisper model to load.
        device: The device to load the model on (always 'mlx' for this fork).
        device_index: Device index (not used for MLX).
        compute_type: Compute type (float16, float32, int4).
        asr_options: ASR options to pass to the model.
        language: Model language.
        vad_method: VAD method to use ('pyannote' or 'silero').
        vad_options: VAD options.
        task: Task to perform ('transcribe' or 'translate').
        download_root: Directory to download models to.
        local_files_only: If True, only use local files.
        threads: Number of threads (not used for MLX).
        backend: Backend to use ('auto', 'mlx', 'standard', 'batch').
        batch_size: Batch size for processing.
        
    Returns:
        MLXWhisperPipeline: The loaded model pipeline.
    """
    # Force MLX backend for this fork
    if device == "cuda" and platform.system() == "Darwin":
        device = "cpu"  # Use CPU for VAD on macOS
    
    # Determine which MLX backend to use
    if backend == "auto":
        # Auto-select based on batch size
        backend = "batch" if batch_size > 1 else "standard"
    
    # Load the appropriate MLX backend
    if backend in ["lightning", "mlx_lightning"]:
        from whisperx.backends.mlx_lightning import WhisperMLXLightning
        # Remove word_timestamps from kwargs as it's handled via align_words
        kwargs_filtered = {k: v for k, v in kwargs.items() if k != 'word_timestamps'}
        mlx_backend = WhisperMLXLightning(
            model_name=whisper_arch,
            compute_type=compute_type,
            **kwargs_filtered
        )
    elif backend in ["batch", "mlx_batch"]:
        from whisperx.backends.mlx_simple import SimpleMLXWhisperBackend
        mlx_backend = SimpleMLXWhisperBackend(
            model_name=whisper_arch,
            batch_size=batch_size,
            device="mlx",
            compute_type=compute_type,
            asr_options=asr_options,
            **kwargs
        )
    else:
        # Standard MLX backend
        from whisperx.backends.mlx_whisper import MlxWhisperBackend
        mlx_backend = MlxWhisperBackend(
            model=whisper_arch,
            device="mlx",
            device_index=device_index,
            compute_type=compute_type,
            download_root=download_root,
            local_files_only=local_files_only,
            threads=threads,
            asr_options=asr_options,
            vad_method=vad_method,
            vad_options=vad_options,
            language=language,
            task=task,
            batch_size=batch_size,
            **kwargs
        )
    
    # Initialize VAD if needed
    vad_model = None
    if vad_method and vad_method != "none":
        default_vad_options = {
            "chunk_size": 30,
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }
        
        if vad_options is not None:
            default_vad_options.update(vad_options)
        
        if vad_method == "silero":
            vad_model = Silero(**default_vad_options)
        elif vad_method == "pyannote":
            # VAD runs on CPU for now
            vad_device = "cpu"
            # Remove device from options if it exists to avoid duplicate argument
            pyannote_options = default_vad_options.copy()
            pyannote_options.pop('device', None)
            vad_model = Pyannote(vad_device, use_auth_token=None, **pyannote_options)
    
    # Return pipeline with VAD
    return MLXWhisperPipeline(backend=mlx_backend, vad_model=vad_model)