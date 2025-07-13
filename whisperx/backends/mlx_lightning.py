"""
Lightning-inspired MLX backend with optimized transcription and optional word alignment
Combines fast transcription with optional WhisperX-style word timestamps
"""
import numpy as np
import mlx.core as mx
from typing import List, Dict, Any, Optional

from mlx_whisper.load_models import load_model
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES, SAMPLE_RATE
from mlx_whisper.decoding import decode, DecodingOptions

from .base import WhisperBackend


# Singleton model cache
_model_cache = {}


class WhisperMLXLightning(WhisperBackend):
    """
    Lightning-inspired implementation with key optimizations:
    1. Single mel computation for entire audio
    2. Greedy decoding (temperature=0.0) 
    3. Model caching
    4. Sequential processing (avoids MLX threading issues)
    5. Optional word-level timestamps via wav2vec2 alignment
    
    This is the unified Lightning backend combining all functionality.
    """
    
    def __init__(self, 
                 model_name: str,
                 compute_type: str = "float16",
                 device: str = "auto",
                 download_root: str = None,
                 **kwargs):
        
        self.model_name = model_name
        self.compute_type = compute_type
        
        # Cache model
        cache_key = f"{model_name}_{compute_type}"
        if cache_key not in _model_cache:
            dtype = mx.float16 if compute_type == "float16" else mx.float32
            # Convert model name to MLX format if needed
            if not model_name.startswith("mlx-community/"):
                # Handle distil models
                if "distil" in model_name:
                    if model_name.startswith("distil-whisper-"):
                        # Format: distil-whisper-large-v3 -> mlx-community/distil-whisper-large-v3
                        model_name = f"mlx-community/{model_name}"
                    elif model_name.startswith("distil-"):
                        # Format: distil-large-v3 -> mlx-community/distil-whisper-large-v3
                        model_name = f"mlx-community/distil-whisper-{model_name[7:]}"
                    else:
                        # Format: large-v3-distil -> mlx-community/distil-whisper-large-v3
                        model_name = f"mlx-community/distil-whisper-{model_name}"
                else:
                    # Regular whisper models
                    model_name = f"mlx-community/whisper-{model_name}-mlx"
            _model_cache[cache_key] = load_model(model_name, dtype=dtype)
        self.model = _model_cache[cache_key]
        
        # Lightning default: greedy decoding
        self.temperature = kwargs.get('temperature', 0.0)
        
        # Alignment model cache for word timestamps
        self.align_model_cache = {}
        
    def transcribe_batch(self, 
                        segments: List[Dict[str, Any]],
                        batch_size: int = 8,
                        align_words: bool = False,
                        **kwargs) -> Dict[str, Any]:
        """Batch transcription of VAD segments with optional word alignment"""
        all_segments = []
        language = None
        
        # First pass: transcribe all segments
        for segment in segments:
            audio = segment.get('audio')
            if audio is None:
                continue
                
            # Transcribe this segment
            result = self._transcribe_core(audio, **kwargs)
            
            # Adjust timestamps and add segments
            for seg in result.get('segments', []):
                seg['start'] += segment['start']
                seg['end'] = min(seg['end'] + segment['start'], segment['end'])
                all_segments.append(seg)
            
            # Get language from first segment
            if language is None and result.get('language'):
                language = result['language']
        
        result = {
            'segments': all_segments,
            'language': language or 'en'
        }
        
        # Second pass: word alignment if requested
        if align_words and segments:
            result = self._align_batch_words(result, segments)
        
        return result
    
    def transcribe(self,
                  audio: np.ndarray,
                  task: str = "transcribe",
                  language: str = None,
                  align_words: bool = False,
                  batch_size: int = None,
                  **kwargs) -> Dict[str, Any]:
        """
        Transcribe with Lightning optimizations and optional word alignment.
        
        Args:
            audio: Audio data as numpy array
            task: Task to perform ("transcribe" or "translate")
            language: Language code or None for auto-detection
            align_words: Whether to perform word-level alignment
            batch_size: Batch size (ignored, kept for compatibility)
            **kwargs: Additional arguments
            
        Returns:
            Transcription result with segments and optional word timestamps
        """
        # Remove batch_size from kwargs to avoid passing it to decode
        kwargs.pop('batch_size', None)
        
        # First, do fast transcription
        result = self._transcribe_core(audio, task, language, **kwargs)
        
        # If word alignment requested, perform it
        if align_words:
            result = self._align_words(result, audio, language)
        
        return result
    
    def _transcribe_core(self,
                        audio: np.ndarray,
                        task: str = "transcribe",
                        language: str = None,
                        **kwargs) -> Dict[str, Any]:
        """
        Core transcription logic with Lightning optimizations
        """
        # Key optimization 1: Compute mel ONCE for entire audio
        mel_full = log_mel_spectrogram(audio, n_mels=self.model.dims.n_mels)
        total_frames = mel_full.shape[0]
        
        # Process segments
        all_text = []
        all_segments = []
        
        # Key optimization 2: Process with minimal overhead
        seek = 0
        segment_idx = 0
        
        while seek < total_frames:
            # Extract segment
            segment_end = min(seek + N_FRAMES, total_frames)
            mel_segment = mel_full[seek:segment_end]
            
            # Pad if needed
            if mel_segment.shape[0] < N_FRAMES:
                mel_segment = pad_or_trim(mel_segment, N_FRAMES, axis=0)
            
            # Convert to MLX
            mel_mx = mx.array(mel_segment, dtype=mx.float16 if self.compute_type == "float16" else mx.float32)
            
            # Key optimization 3: Greedy decoding
            options = DecodingOptions(
                task=task,
                language=language,
                temperature=self.temperature,  # 0.0 for greedy
                without_timestamps=False,
                fp16=(self.compute_type == "float16"),
            )
            
            # Decode single segment
            result = decode(self.model, mel_mx[None], options)[0]
            
            # Process result
            text = result.text.strip()
            if text:
                all_text.append(text)
                
                # Create segment
                segment_start = seek / 100.0  # Convert frames to seconds
                segment_end_time = segment_end / 100.0
                
                all_segments.append({
                    'start': segment_start,
                    'end': segment_end_time,
                    'text': text,
                    'id': segment_idx
                })
                segment_idx += 1
            
            # Move to next segment (no overlap for speed)
            seek = segment_end
        
        # Get language
        detected_language = "en"
        if hasattr(result, 'language'):
            detected_language = result.language
        
        return {
            "text": " ".join(all_text),
            "segments": all_segments,
            "language": language or detected_language,
        }
    
    def _align_words(self, 
                    transcription_result: Dict[str, Any],
                    audio: np.ndarray,
                    language: str = None) -> Dict[str, Any]:
        """
        Perform word-level alignment using WhisperX's approach.
        """
        try:
            import whisperx
            
            # Get language if not provided
            if not language:
                language = transcription_result.get("language", "en")
            
            # Load alignment model (cached)
            cache_key = f"align_{language}"
            if cache_key not in self.align_model_cache:
                model_a, metadata = whisperx.load_align_model(
                    language_code=language,
                    device="cpu"  # MLX doesn't need CUDA
                )
                self.align_model_cache[cache_key] = (model_a, metadata)
            else:
                model_a, metadata = self.align_model_cache[cache_key]
            
            # Perform alignment
            aligned_result = whisperx.align(
                transcription_result["segments"],
                model_a,
                metadata,
                audio,
                "cpu",
                return_char_alignments=False
            )
            
            # Update segments with word-level data
            transcription_result["segments"] = aligned_result["segments"]
            
            # Ensure consistent format
            for segment in transcription_result["segments"]:
                if "words" not in segment:
                    segment["words"] = []
                
                # Normalize word format
                for word in segment.get("words", []):
                    if "text" in word and "word" not in word:
                        word["word"] = word["text"]
                    if "score" in word and "probability" not in word:
                        word["probability"] = word["score"]
            
        except Exception as e:
            print(f"Warning: Word alignment failed: {e}")
            print("Returning transcription without word-level timestamps")
            
            # Add empty words arrays for compatibility
            for segment in transcription_result["segments"]:
                if "words" not in segment:
                    segment["words"] = []
        
        return transcription_result
    
    def _align_batch_words(self,
                          result: Dict[str, Any],
                          segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Align words for batch transcription"""
        try:
            import whisperx
            
            # Get language from transcription
            language = result.get('language', 'en')
            
            # Load alignment model (cached)
            cache_key = f"align_{language}"
            if cache_key not in self.align_model_cache:
                model_a, metadata = whisperx.load_align_model(
                    language_code=language,
                    device="cpu"
                )
                self.align_model_cache[cache_key] = (model_a, metadata)
            else:
                model_a, metadata = self.align_model_cache[cache_key]
            
            # Process each segment that has audio data
            aligned_segments = []
            segment_idx = 0
            
            for vad_segment in segments:
                segment_audio = vad_segment.get('audio')
                if segment_audio is None:
                    continue
                
                # Find transcribed segments that belong to this VAD segment
                segment_transcriptions = []
                vad_start = vad_segment['start']
                vad_end = vad_segment['end']
                
                # Collect segments within this VAD segment's time range
                while segment_idx < len(result['segments']):
                    seg = result['segments'][segment_idx]
                    if seg['start'] >= vad_start and seg['end'] <= vad_end:
                        # Create a copy with relative timestamps for alignment
                        seg_copy = seg.copy()
                        seg_copy['start'] -= vad_start
                        seg_copy['end'] -= vad_start
                        segment_transcriptions.append(seg_copy)
                        segment_idx += 1
                    else:
                        break
                
                if segment_transcriptions:
                    # Perform alignment on this segment
                    aligned = whisperx.align(
                        segment_transcriptions,
                        model_a,
                        metadata,
                        segment_audio,
                        "cpu",
                        return_char_alignments=False
                    )
                    
                    # Adjust timestamps back to absolute and add to results
                    for aligned_seg in aligned.get('segments', []):
                        # Adjust segment timestamps
                        aligned_seg['start'] += vad_start
                        aligned_seg['end'] += vad_start
                        
                        # Adjust word timestamps
                        for word in aligned_seg.get('words', []):
                            word['start'] += vad_start
                            word['end'] += vad_start
                        
                        aligned_segments.append(aligned_seg)
            
            # Replace segments with aligned versions
            result['segments'] = aligned_segments
            
        except Exception as e:
            print(f"Warning: Batch word alignment failed: {e}")
            print("Returning transcription without word-level timestamps")
        
        return result
    
    def detect_language(self, audio: np.ndarray) -> str:
        """Detect language from audio"""
        # Use first 30 seconds
        sample = audio[:30 * SAMPLE_RATE]
        
        # Quick decode to get language
        mel = log_mel_spectrogram(sample, n_mels=self.model.dims.n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=0)
        mel_mx = mx.array(mel, dtype=mx.float16 if self.compute_type == "float16" else mx.float32)
        
        options = DecodingOptions(
            task="transcribe",
            temperature=0.0,
            fp16=(self.compute_type == "float16"),
        )
        
        result = decode(self.model, mel_mx[None], options)[0]
        return getattr(result, 'language', 'en')
    
    @property
    def supported_languages(self) -> List[str]:
        """Return list of supported languages"""
        return ["en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", 
                "pl", "ca", "nl", "ar", "sv", "it", "id", "hi", "fi", "vi",
                "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
                "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk",
                "te", "fa", "lv", "bn", "sr", "az", "sl", "kn", "et", "mk",
                "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
                "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",
                "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
                "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
                "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su"]
    
    @property
    def is_multilingual(self) -> bool:
        """Check if model is multilingual"""
        return not self.model_name.endswith(".en")