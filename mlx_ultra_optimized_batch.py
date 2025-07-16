#!/usr/bin/env python3
"""
Ultra-optimized batch processing with word timestamps.

This implementation:
1. Performs batch decoding WITH timestamps enabled (using our fix)
2. Extracts word timestamps directly from the batch results
3. Achieves true single-pass processing
"""

import mlx.core as mx
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
from dataclasses import dataclass
import warnings

from mlx_whisper.load_models import load_model
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim
from mlx_whisper.decoding import DecodingOptions, DecodingResult
from mlx_whisper.tokenizer import get_tokenizer
from mlx_whisper.timing import add_word_timestamps

# Import and patch
import mlx_whisper.decoding as mlx_decoding
from mlx_whisper_batch_decoder import batch_decode


def install_broadcasting_fix():
    """Install a permanent fix for the broadcasting bug."""
    # Check if already patched
    if hasattr(mlx_decoding.ApplyTimestampRules, '_original_apply'):
        return
        
    # Save original
    mlx_decoding.ApplyTimestampRules._original_apply = mlx_decoding.ApplyTimestampRules.apply
    
    def fixed_apply(self, logits: mx.array, tokens: mx.array) -> mx.array:
        """Fixed version that handles batch dimensions correctly."""
        if self.tokenizer.no_timestamps is not None:
            # Ensure we have the mask
            if not hasattr(self, 'mask'):
                mask = np.ones(logits.shape[-1], dtype=np.float32)
                mask[: self.tokenizer.timestamp_begin] = -np.inf
                self.mask = mask
                
            mask = mx.array(self.mask)
            
            # Fix: add keepdims=True for proper broadcasting
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            
            # Timestamp probability
            timestamp_logprob = logprobs[:, self.tokenizer.timestamp_begin :].logsumexp(
                axis=-1, keepdims=True
            )
            
            # Max text token probability
            max_text_token_logprob = logprobs[:, : self.tokenizer.timestamp_begin].max(
                axis=-1, keepdims=True
            )
            
            # Apply mask conditionally
            mask = mx.where(
                timestamp_logprob > max_text_token_logprob,
                mask,
                mx.ones_like(mask)
            )
            
            logits = logits + mask
            
        return logits
    
    # Install the fix
    mlx_decoding.ApplyTimestampRules.apply = fixed_apply
    print("✓ Broadcasting fix installed")


@dataclass
class UltraOptimizedResult:
    """Result with integrated word timestamps."""
    text: str
    words: List[Dict]
    tokens: List[int]
    language: str = "en"
    processing_time: float = 0.0


class UltraOptimizedBatchProcessor:
    """
    Ultra-optimized processor that truly processes everything in a single pass.
    """
    
    def __init__(self, model_name: str = "mlx-community/whisper-large-v3-mlx"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        # Install fix on initialization
        install_broadcasting_fix()
        
    def load(self):
        """Load model and tokenizer."""
        if self.model is None:
            print("Loading model...")
            self.model = load_model(self.model_name, dtype=mx.float16)
            self.tokenizer = get_tokenizer(
                self.model.is_multilingual,
                num_languages=getattr(self.model, 'num_languages', 99),
                language="en",
                task="transcribe",
            )
            
    def process_batch_ultra(
        self,
        mel_batch: mx.array,
        extract_words: bool = True
    ) -> List[UltraOptimizedResult]:
        """
        Ultra-optimized batch processing in a true single pass.
        """
        batch_size = mel_batch.shape[0]
        
        # Single-pass batch decode WITH timestamps
        print(f"  Single-pass batch decoding {batch_size} samples...")
        start_time = time.time()
        
        # Enable timestamps for word alignment
        options = DecodingOptions(
            language="en",
            temperature=0.0,
            without_timestamps=False if extract_words else True,
            suppress_blank=False,
            suppress_tokens=[]
        )
        
        # Batch decode with timestamps enabled (using our fix)
        decode_results = batch_decode(self.model, mel_batch, options)
        
        decode_time = time.time() - start_time
        
        # Extract word timestamps if requested
        results = []
        
        if extract_words and not options.without_timestamps:
            print(f"  Extracting word timestamps from batch results...")
            words_start = time.time()
            
            # Process word timestamps in batch
            for i, result in enumerate(decode_results):
                # Create segments from batch result
                segments = self._create_segments_from_result(result)
                
                if segments and len(result.tokens) > 0:
                    try:
                        # Add word timestamps using the mel for this sample
                        add_word_timestamps(
                            segments=segments,
                            model=self.model,
                            tokenizer=self.tokenizer,
                            mel=mel_batch[i],
                            num_frames=mel_batch[i].shape[-1],
                            last_speech_timestamp=0.0
                        )
                    except Exception as e:
                        warnings.warn(f"Failed to extract words for sample {i}: {e}")
                
                # Extract words from segments
                words = []
                for segment in segments:
                    if "words" in segment:
                        words.extend(segment["words"])
                
                results.append(UltraOptimizedResult(
                    text=result.text,
                    words=words,
                    tokens=result.tokens,
                    language=result.language,
                    processing_time=time.time() - start_time
                ))
            
            words_time = time.time() - words_start
            total_time = time.time() - start_time
            
            print(f"  Decode: {decode_time:.1f}s, Words: {words_time:.1f}s")
            print(f"  Total: {total_time:.1f}s ({batch_size * 30 / total_time:.1f}x realtime)")
        else:
            # No word extraction
            for result in decode_results:
                results.append(UltraOptimizedResult(
                    text=result.text,
                    words=[],
                    tokens=result.tokens,
                    language=result.language,
                    processing_time=decode_time
                ))
            print(f"  Decode time: {decode_time:.1f}s ({batch_size * 30 / decode_time:.1f}x realtime)")
        
        return results
    
    def _create_segments_from_result(self, result: DecodingResult) -> List[Dict]:
        """Create segments from decoding result for word timestamp extraction."""
        # Look for timestamp tokens to create proper segments
        tokens = result.tokens
        
        if not tokens:
            return []
            
        # Simple segmentation - could be improved
        segments = []
        current_tokens = []
        start_time = 0.0
        
        for token in tokens:
            if token >= self.tokenizer.timestamp_begin:
                # End current segment
                if current_tokens:
                    segments.append({
                        "seek": 0,
                        "tokens": current_tokens,
                        "text": self.tokenizer.decode(current_tokens),
                        "start": start_time,
                        "end": (token - self.tokenizer.timestamp_begin) * 0.02
                    })
                    current_tokens = []
                start_time = (token - self.tokenizer.timestamp_begin) * 0.02
            else:
                current_tokens.append(token)
        
        # Final segment
        if current_tokens:
            segments.append({
                "seek": 0,
                "tokens": current_tokens,
                "text": self.tokenizer.decode(current_tokens),
                "start": start_time,
                "end": 30.0  # Default to chunk duration
            })
        
        # If no timestamp tokens found, create single segment
        if not segments and tokens:
            segments = [{
                "seek": 0,
                "tokens": tokens,
                "text": result.text,
                "start": 0.0,
                "end": 30.0
            }]
        
        return segments
    
    def process_audio_file(
        self,
        audio_path: str,
        batch_size: int = 8,
        chunk_duration: float = 30.0
    ) -> Dict:
        """Process audio file with ultra-optimized single-pass approach."""
        import librosa
        import json
        
        self.load()
        
        print(f"\nUltra-Optimized Processing: {audio_path}")
        print(f"Batch size: {batch_size}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        duration = len(audio) / sr
        print(f"Duration: {duration/60:.1f} minutes")
        
        # Create chunks
        chunk_samples = int(sr * chunk_duration)
        chunks = []
        
        for start in range(0, len(audio), chunk_samples):
            end = min(start + chunk_samples, len(audio))
            chunk = audio[start:end]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunks.append(chunk.astype(np.float32))
        
        print(f"Created {len(chunks)} chunks")
        
        # Process in batches
        all_text = []
        all_words = []
        num_batches = (len(chunks) + batch_size - 1) // batch_size
        
        print(f"\nProcessing {num_batches} batches...")
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]
            
            print(f"\nBatch {batch_idx + 1}/{num_batches} ({len(batch_chunks)} chunks)")
            
            # Prepare mel spectrograms
            mels = []
            for chunk in batch_chunks:
                mel = log_mel_spectrogram(chunk, n_mels=self.model.dims.n_mels)
                mel = pad_or_trim(mel, 3000, axis=0)
                mels.append(mel)
            
            mel_batch = mx.array(np.stack(mels))
            
            # Process batch in single pass
            results = self.process_batch_ultra(mel_batch, extract_words=True)
            
            # Collect results
            for i, result in enumerate(results):
                all_text.append(result.text)
                
                # Adjust word timestamps for chunk position
                chunk_offset = (start_idx + i) * chunk_duration
                for word in result.words:
                    word_copy = word.copy()
                    word_copy["start"] += chunk_offset
                    word_copy["end"] += chunk_offset
                    all_words.append(word_copy)
        
        total_time = time.time() - start_time
        throughput = duration / total_time
        
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {total_time:.1f}s")
        print(f"Throughput: {throughput:.1f}x realtime")
        print(f"Total words: {len(all_words)}")
        
        # Save results
        full_text = " ".join(all_text)
        
        output_data = {
            "method": "ultra_optimized_single_pass",
            "audio_file": audio_path,
            "duration_seconds": duration,
            "processing_time": total_time,
            "throughput": throughput,
            "text": full_text,
            "word_count": len(all_words),
            "words": all_words
        }
        
        base_name = audio_path.rsplit('.', 1)[0]
        output_file = f"{base_name}_ultra_optimized.json"
        
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Show sample
        if all_words:
            print(f"\nFirst 10 words:")
            for w in all_words[:10]:
                print(f"  {w['word']} [{w['start']:.2f}s - {w['end']:.2f}s]")
        
        return output_data


def benchmark_ultra_optimized():
    """Benchmark the ultra-optimized approach."""
    processor = UltraOptimizedBatchProcessor()
    
    # Test on short.wav first
    print("Testing on short.wav...")
    result1 = processor.process_audio_file("short.wav", batch_size=8)
    
    print(f"\n\nSummary:")
    print(f"- Method: Ultra-optimized single-pass")
    print(f"- Throughput: {result1['throughput']:.1f}x realtime")
    print(f"- Words extracted: {result1['word_count']}")
    
    return result1


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        processor = UltraOptimizedBatchProcessor()
        processor.process_audio_file(sys.argv[1], batch_size=8)
    else:
        benchmark_ultra_optimized()