import os
from typing import List, Optional, Union
from dataclasses import replace
import warnings

# Threading limits are already set in __main__.py or by the user
# Don't override them here as it may cause conflicts

import numpy as np
import torch
import mlx.core as mx
import mlx_whisper

from whisperx.audio import N_SAMPLES, SAMPLE_RATE, load_audio
from whisperx.types import SingleSegment, TranscriptionResult
from whisperx.vads import Vad, Silero, Pyannote
from .base import WhisperBackend


class MlxWhisperBackend(WhisperBackend):
    """Backend implementation for mlx-whisper on Apple Silicon."""
    
    def __init__(
        self,
        model: str,
        device: str = "mlx",  # MLX always runs on Apple Silicon
        device_index: int = 0,  # Not used for MLX
        compute_type: str = "float16",
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        threads: int = 4,  # Not used for MLX
        asr_options: Optional[dict] = None,
        vad_method: str = "pyannote",
        vad_options: Optional[dict] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        batch_size: int = 8,
        **kwargs
    ):
        # Convert model name to MLX format if needed
        if not model.startswith("mlx-community/"):
            # Map common names to MLX model names
            model_map = {
                "tiny": "mlx-community/whisper-tiny-mlx",
                "base": "mlx-community/whisper-base-mlx", 
                "small": "mlx-community/whisper-small-mlx",
                "medium": "mlx-community/whisper-medium-mlx",
                "large": "mlx-community/whisper-large-mlx",
                "large-v2": "mlx-community/whisper-large-v2-mlx",
                "large-v3": "mlx-community/whisper-large-v3-mlx",
            }
            # Also handle quantized versions
            if compute_type in ["int4", "q4"]:
                model_map = {k: v + "-4bit" for k, v in model_map.items()}
            
            self.model_path = model_map.get(model, model)
        else:
            self.model_path = model
        self.batch_size = batch_size
        self.dtype = compute_type
        self.language = language
        self.task = task
        
        # Convert compute_type to MLX dtype
        self.mlx_dtype = mx.float16 if compute_type == "float16" else mx.float32
        
        # Don't load model here - mlx_whisper.transcribe loads its own model
        # Model loading was causing hanging issues
        self.model = None
        self._model_loaded = False
        
        # Setup ASR options - separate transcribe options from decoding options
        self.default_asr_options = {
            # Options for mlx_whisper.transcribe
            "temperature": 0.01,  # Use small temperature > 0 to avoid beam search
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "condition_on_previous_text": False,
            "initial_prompt": None,
            "word_timestamps": False,  # Disabled by default due to performance issues
            "prepend_punctuations": "\"'\u2018\u00BF([{-",
            "append_punctuations": "\"'.。,，!！?？:：\")]}、",
            "hallucination_silence_threshold": None,
            # DecodingOptions parameters
            "language": language,
            "task": task,
            "fp16": compute_type == "float16",
        }
        
        if asr_options is not None:
            self.default_asr_options.update(asr_options)
            # Warn if word_timestamps is enabled
            if asr_options.get("word_timestamps", False):
                warnings.warn(
                    "word_timestamps=True may cause performance issues with MLX backend. "
                    "Consider using word_timestamps=False for better performance.",
                    UserWarning
                )
        
        # Store VAD parameters but don't initialize VAD here
        # VAD will be initialized by the pipeline in asr.py
        default_vad_options = {
            "chunk_size": 30,
            "vad_onset": 0.500,
            "vad_offset": 0.363
        }
        
        if vad_options is not None:
            default_vad_options.update(vad_options)
            
        self.vad_params = default_vad_options
        self.vad_model = None  # VAD is handled by the pipeline
    
    def _lazy_init(self):
        """Lazy initialization of the MLX model."""
        # Model is now loaded in __init__, so this is a no-op
        pass
    
    def transcribe(
        self,
        audio: Union[str, np.ndarray],
        batch_size: Optional[int] = None,
        num_workers: int = 0,
        language: Optional[str] = None,
        task: Optional[str] = None,
        chunk_size: int = 30,
        print_progress: bool = False,
        combined_progress: bool = False,
        verbose: bool = False,
        **kwargs
    ) -> TranscriptionResult:
        """Transcribe audio using MLX Whisper backend.
        
        This method transcribes raw audio without VAD segmentation.
        VAD segmentation should be handled by the pipeline if needed.
        """
        self._lazy_init()
        
        # Load audio if path is provided
        if isinstance(audio, str):
            audio = load_audio(audio)
        
        # Update language and task if provided
        if language is not None:
            self.default_asr_options["language"] = language
        elif self.language is not None:
            self.default_asr_options["language"] = self.language
            
        if task is not None:
            self.default_asr_options["task"] = task
        elif self.task is not None:
            self.default_asr_options["task"] = self.task
        
        # Prepare transcription options
        transcribe_options = self.default_asr_options.copy()
        transcribe_options.update(kwargs)
        
        # Remove verbose from options if it exists, pass it separately
        transcribe_options.pop('verbose', None)
        
        # Convert 'temperatures' to 'temperature' (MLX expects singular)
        if 'temperatures' in transcribe_options:
            temps = transcribe_options.pop('temperatures')
            if isinstance(temps, (list, tuple)) and len(temps) > 0:
                transcribe_options['temperature'] = temps[0]
            else:
                transcribe_options['temperature'] = temps
                
        # Convert log_prob_threshold to logprob_threshold (MLX uses no underscore)
        if 'log_prob_threshold' in transcribe_options:
            transcribe_options['logprob_threshold'] = transcribe_options.pop('log_prob_threshold')
                
        # MLX doesn't support beam search yet, so remove beam_size when temperature is 0
        if transcribe_options.get('temperature', 0.0) == 0.0:
            transcribe_options.pop('beam_size', None)
            transcribe_options.pop('patience', None)
            transcribe_options.pop('best_of', None)
            
        # Remove unsupported options
        for unsupported in ['suppress_numerals', 'max_new_tokens', 'clip_timestamps', 
                          'repetition_penalty', 'no_repeat_ngram_size', 
                          'prompt_reset_on_temperature', 'prefix', 'suppress_blank',
                          'suppress_tokens', 'without_timestamps', 'max_initial_timestamp',
                          'multilingual', 'hotwords', 'batch_size', 'num_workers',
                          'vad_segments', 'combined_progress', 'chunk_size', 'print_progress']:
            transcribe_options.pop(unsupported, None)
        
        # Transcribe the entire audio
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self.model_path,
            verbose=verbose,
            **transcribe_options
        )
        
        # Extract segments and language
        segments = result.get("segments", [])
        detected_language = result.get("language", self.default_asr_options.get("language", "en"))
        
        return {"segments": segments, "language": detected_language}
    
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect the language of the audio using MLX."""
        self._lazy_init()
        
        if audio.shape[0] < N_SAMPLES:
            print("Warning: audio is shorter than 30s, language detection may be inaccurate.")
        
        # Use MLX transcribe with language detection
        result = mlx_whisper.transcribe(
            audio[:N_SAMPLES],  # Use first 30 seconds
            path_or_hf_repo=self.model_path,
            verbose=False,
            language=None,  # Auto-detect language
            fp16=self.mlx_dtype == mx.float16
        )
        
        detected_language = result.get("language", "en")
        print(f"Detected language: {detected_language}")
        return detected_language
    
    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        # MLX Whisper supports all Whisper languages
        return list(mlx_whisper.tokenizer.LANGUAGES.keys())
    
    @property  
    def is_multilingual(self) -> bool:
        """Return whether the model supports multiple languages."""
        # Check model name to determine if it's multilingual
        model_name = self.model_path.lower()
        return not model_name.endswith('.en')