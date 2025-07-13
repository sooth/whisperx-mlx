#!/usr/bin/env python3
"""
Hybrid VAD implementation that intelligently chooses between CPU and MLX
Based on profiling results showing CPU is optimal for VAD
"""

import time
import platform
from typing import List, Dict, Optional, Union
import numpy as np

from whisperx.vads.vad import Vad
from whisperx.vads.silero import Silero
from whisperx.diarize import Segment as SegmentX
from whisperx.audio import SAMPLE_RATE


class HybridVAD(Vad):
    """
    Intelligent VAD that uses the optimal backend based on context
    
    Key findings from profiling:
    - CPU Silero: 317x realtime for single streams
    - MLX VAD: 216x realtime (slower due to GPU overhead)
    - MLX shines with batch processing (5.5x speedup)
    
    Strategy:
    - Use CPU Silero for single audio streams
    - Use MLX for batch processing (if implemented)
    - Use CPU for real-time/streaming
    - Consider MLX for very long audio (>60s) where we can batch frames
    """
    
    def __init__(self, 
                 prefer_mlx: bool = False,
                 batch_threshold: int = 4,
                 **kwargs):
        """
        Initialize Hybrid VAD
        
        Args:
            prefer_mlx: Force MLX usage (for testing)
            batch_threshold: Minimum batch size to prefer MLX
            **kwargs: VAD options
        """
        print(">>Initializing Hybrid VAD...")
        super().__init__(kwargs.get('vad_onset', 0.5))
        
        self.prefer_mlx = prefer_mlx
        self.batch_threshold = batch_threshold
        self.vad_options = kwargs
        
        # Initialize CPU VAD (always available)
        self.cpu_vad = Silero(**kwargs)
        
        # Try to initialize MLX VAD if on macOS
        self.mlx_vad = None
        self.mlx_available = False
        
        if platform.system() == "Darwin":
            try:
                from whisperx.integrate_mlx_vad import SileroMLXVAD
                self.mlx_vad = SileroMLXVAD(**kwargs)
                self.mlx_available = True
                print("  ✓ MLX VAD available for batch processing")
            except ImportError:
                print("  ✗ MLX VAD not available, using CPU only")
        
        # Statistics
        self.stats = {
            "cpu_calls": 0,
            "mlx_calls": 0,
            "cpu_time": 0.0,
            "mlx_time": 0.0,
            "total_audio_duration": 0.0
        }
    
    def __call__(self, audio_dict: Dict, **kwargs) -> List[SegmentX]:
        """
        Process audio with optimal VAD backend
        
        Args:
            audio_dict: Dict with 'waveform' and 'sample_rate'
            
        Returns:
            List of speech segments
        """
        # Calculate audio duration
        waveform = audio_dict['waveform']
        if hasattr(waveform, 'shape'):
            if len(waveform.shape) > 1:
                audio_length = waveform.shape[-1]
            else:
                audio_length = waveform.shape[0]
        else:
            audio_length = len(waveform)
        
        duration = audio_length / audio_dict['sample_rate']
        self.stats['total_audio_duration'] += duration
        
        # Choose backend
        use_mlx = self._should_use_mlx(duration, kwargs.get('batch_size', 1))
        
        # Process with chosen backend
        start_time = time.time()
        
        if use_mlx and self.mlx_available:
            segments = self.mlx_vad(audio_dict, **kwargs)
            elapsed = time.time() - start_time
            self.stats['mlx_calls'] += 1
            self.stats['mlx_time'] += elapsed
            backend = "MLX"
        else:
            segments = self.cpu_vad(audio_dict, **kwargs)
            elapsed = time.time() - start_time
            self.stats['cpu_calls'] += 1
            self.stats['cpu_time'] += elapsed
            backend = "CPU"
        
        # Report performance
        realtime_factor = duration / elapsed
        print(f"  Hybrid VAD: {backend} backend, {len(segments)} segments, {realtime_factor:.1f}x realtime")
        
        return segments
    
    def _should_use_mlx(self, audio_duration: float, batch_size: int) -> bool:
        """
        Decide whether to use MLX based on profiling results
        
        Args:
            audio_duration: Audio length in seconds
            batch_size: Number of audio streams to process
            
        Returns:
            True if MLX should be used
        """
        if self.prefer_mlx:
            return True
        
        if not self.mlx_available:
            return False
        
        # Use MLX for batch processing
        if batch_size >= self.batch_threshold:
            return True
        
        # For single streams, CPU is always faster
        # (based on profiling: CPU 317x vs MLX 216x realtime)
        return False
    
    def process_batch(self, audio_batch: List[Dict], **kwargs) -> List[List[SegmentX]]:
        """
        Process multiple audio streams efficiently
        
        Args:
            audio_batch: List of audio dicts
            
        Returns:
            List of segment lists
        """
        batch_size = len(audio_batch)
        
        if batch_size >= self.batch_threshold and self.mlx_available:
            # Use MLX for true batch processing
            # TODO: Implement true batch VAD in MLX backend
            print(f"  Hybrid VAD: Processing batch of {batch_size} with MLX")
            
        # For now, process sequentially
        results = []
        for audio_dict in audio_batch:
            segments = self(audio_dict, batch_size=batch_size, **kwargs)
            results.append(segments)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        stats = self.stats.copy()
        
        # Calculate realtime factors
        if stats['cpu_time'] > 0:
            cpu_audio = stats['total_audio_duration'] * (stats['cpu_calls'] / (stats['cpu_calls'] + stats['mlx_calls']))
            stats['cpu_realtime_factor'] = cpu_audio / stats['cpu_time']
        else:
            stats['cpu_realtime_factor'] = 0
        
        if stats['mlx_time'] > 0:
            mlx_audio = stats['total_audio_duration'] * (stats['mlx_calls'] / (stats['cpu_calls'] + stats['mlx_calls']))
            stats['mlx_realtime_factor'] = mlx_audio / stats['mlx_time']
        else:
            stats['mlx_realtime_factor'] = 0
        
        return stats
    
    @staticmethod
    def preprocess_audio(audio):
        """Preprocess audio for VAD"""
        # Use CPU Silero preprocessing
        return Silero.preprocess_audio(audio)
    
    @staticmethod
    def merge_chunks(segments_list, chunk_size, onset=0.5, offset=None):
        """Merge segments into chunks"""
        return Silero.merge_chunks(segments_list, chunk_size, onset, offset)


def optimize_vad_for_whisperx():
    """
    Recommendations for optimal VAD usage in WhisperX
    Based on comprehensive profiling
    """
    print("\nVAD Optimization Strategy for WhisperX")
    print("=" * 50)
    
    print("\n1. Single Audio Stream (Most Common):")
    print("   → Use CPU Silero (317x realtime)")
    print("   → MLX adds 0.13ms overhead per operation")
    print("   → CPU SIMD optimizations beat GPU for small models")
    
    print("\n2. Batch Processing (Multiple Files):")
    print("   → Consider MLX if batch size ≥ 4")
    print("   → 5.5x speedup potential with batching")
    print("   → Implement true batch VAD operations")
    
    print("\n3. Real-time/Streaming:")
    print("   → Always use CPU for low latency")
    print("   → MLX initialization overhead too high")
    print("   → CPU provides consistent performance")
    
    print("\n4. Very Long Audio (>5 minutes):")
    print("   → Still use CPU (consistent 317x realtime)")
    print("   → MLX benefit minimal even for long audio")
    print("   → Consider chunking for memory efficiency")
    
    print("\n5. Integration Recommendation:")
    print("   → Default to CPU Silero VAD")
    print("   → Remove MLX VAD to reduce complexity")
    print("   → Focus MLX optimization on ASR model instead")


if __name__ == "__main__":
    # Test hybrid VAD
    print("Testing Hybrid VAD")
    print("-" * 50)
    
    # Initialize
    vad = HybridVAD(vad_onset=0.5, chunk_size=30)
    
    # Test with different audio lengths
    for duration in [1, 10, 30]:
        samples = int(duration * SAMPLE_RATE)
        audio = np.random.randn(samples) * 0.1
        
        result = vad({
            "waveform": audio,
            "sample_rate": SAMPLE_RATE
        })
        
    # Show statistics
    print("\nPerformance Statistics:")
    stats = vad.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Show recommendations
    optimize_vad_for_whisperx()