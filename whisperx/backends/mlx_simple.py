#!/usr/bin/env python3
"""
Simple optimized MLX Whisper backend
"""

import os
import time
import logging
from typing import List, Dict, Optional, Any, Union
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# Disable numba JIT
os.environ['NUMBA_DISABLE_JIT'] = '1'

import mlx.core as mx
import mlx_whisper

from .base import WhisperBackend


class SimpleMLXWhisperBackend(WhisperBackend):
    """
    Simple optimized MLX backend that leverages mlx_whisper directly
    
    Key optimizations:
    - Model caching across instances
    - Efficient memory management with mx.clear_cache()
    - Greedy decoding by default (temperature=0.0)
    - Minimal overhead
    """
    
    # Class-level model path cache
    _model_cache = {}
    
    def __init__(self, 
                 model_name: str = "large-v3",
                 batch_size: int = 1,  # Not used, for compatibility
                 device: str = "mlx",
                 compute_type: str = "float16",
                 asr_options: Optional[Dict] = None,
                 **kwargs):
        """Initialize simple MLX backend"""
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.asr_options = asr_options or {}
        self.kwargs = kwargs
        
        # Model path
        self.model_path = self._get_model_path(model_name)
        
        # Cache model path
        self._model_cache[model_name] = self.model_path
        
    def _get_model_path(self, model_name: str) -> str:
        """Get model path from name"""
        # Check cache first
        if model_name in self._model_cache:
            return self._model_cache[model_name]
            
        model_map = {
            "tiny": "mlx-community/whisper-tiny",
            "base": "mlx-community/whisper-base-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
        }
        return model_map.get(model_name, model_name)
    
    def transcribe(self,
                  audio: Union[str, np.ndarray],
                  language: Optional[str] = None,
                  task: str = "transcribe",
                  verbose: bool = False,
                  **kwargs) -> Dict[str, Any]:
        """
        Transcribe audio using mlx_whisper
        """
        # Clear cache before processing
        mx.clear_cache()
        
        # Prepare audio
        if isinstance(audio, str):
            # mlx_whisper can handle file paths directly
            audio_input = audio
        else:
            # Ensure float32 numpy array
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            audio_input = audio
        
        # Start timing
        start_time = time.time()
        
        # Filter out invalid kwargs
        valid_kwargs = {}
        invalid_keys = ['batch_size', 'num_workers', 'print_progress', 'combined_progress']
        for k, v in kwargs.items():
            if k not in invalid_keys:
                valid_kwargs[k] = v
        
        # Transcribe with optimized settings
        result = mlx_whisper.transcribe(
            audio_input,
            path_or_hf_repo=self.model_path,
            task=task,
            language=language,
            temperature=0.0,  # Greedy decoding for speed
            word_timestamps=self.asr_options.get('word_timestamps', False),
            fp16=(self.compute_type == "float16"),
            verbose=verbose,
            condition_on_previous_text=False,  # Avoid getting stuck
            **valid_kwargs
        )
        
        # Report performance
        if verbose:
            elapsed = time.time() - start_time
            if isinstance(audio_input, np.ndarray):
                audio_duration = len(audio_input) / 16000
                rtf = audio_duration / elapsed
                logger.info(f"Transcribed {audio_duration:.1f}s in {elapsed:.2f}s (RTF: {rtf:.2f}x)")
        
        return {
            'text': result.get('text', '').strip(),
            'segments': result.get('segments', []),
            'language': result.get('language', language or 'en')
        }
    
    def transcribe_batch(self, 
                        segments: List[Dict[str, Any]], 
                        language: Optional[str] = None,
                        task: str = "transcribe",
                        verbose: bool = False,
                        **kwargs) -> Dict[str, Any]:
        """
        Transcribe multiple segments
        
        For compatibility with VAD pipeline
        """
        if not segments:
            return {'segments': [], 'language': language or 'en'}
        
        all_segments = []
        detected_language = language
        total_duration = 0
        start_time = time.time()
        
        for i, seg in enumerate(segments):
            # Clear cache periodically
            if i > 0 and i % 10 == 0:
                mx.clear_cache()
            
            audio = seg['audio']
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            segment_duration = len(audio) / 16000
            total_duration += segment_duration
            
            # Filter out invalid kwargs
            valid_kwargs = {}
            invalid_keys = ['batch_size', 'num_workers', 'print_progress', 'combined_progress']
            for k, v in kwargs.items():
                if k not in invalid_keys:
                    valid_kwargs[k] = v
            
            # Transcribe segment
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_path,
                task=task,
                language=language,
                temperature=0.0,
                fp16=(self.compute_type == "float16"),
                verbose=False,
                condition_on_previous_text=False,
                **valid_kwargs
            )
            
            # Extract language
            if detected_language is None and result.get('language'):
                detected_language = result['language']
            
            # Adjust timestamps
            for s in result.get('segments', []):
                s['start'] += seg.get('start', 0)
                s['end'] += seg.get('start', 0)
                all_segments.append(s)
        
        # Report performance
        if verbose:
            elapsed = time.time() - start_time
            rtf = total_duration / elapsed
            logger.info(f"Transcribed {len(segments)} segments ({total_duration:.1f}s) in {elapsed:.2f}s")
            logger.info(f"Overall RTF: {rtf:.2f}x")
        
        return {
            'segments': all_segments,
            'language': detected_language or 'en'
        }
    
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect language"""
        # Use first 30 seconds
        sample = audio[:30 * 16000] if len(audio) > 30 * 16000 else audio
        
        result = mlx_whisper.transcribe(
            sample,
            path_or_hf_repo=self.model_path,
            fp16=(self.compute_type == "float16"),
            verbose=False
        )
        
        return result.get('language', 'en')
    
    @property
    def supported_languages(self) -> List[str]:
        """Supported languages"""
        return [
            "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl",
            "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk",
            "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr",
            "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn",
            "sr", "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne",
            "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si", "km", "sn",
            "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi",
            "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my",
            "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"
        ]
    
    @property
    def is_multilingual(self) -> bool:
        """Check if model is multilingual"""
        return not self.model_name.endswith(".en")