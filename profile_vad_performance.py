#!/usr/bin/env python3
"""
Profile VAD performance to understand why MLX is slower than CPU
"""

import time
import numpy as np
import mlx.core as mx
import torch
from whisperx.vads.silero import Silero
from integrate_mlx_vad import SileroMLXVAD, MLXOptimizedVAD
from whisperx.audio import SAMPLE_RATE

class VADProfiler:
    """Profile different VAD implementations"""
    
    def __init__(self):
        self.results = {}
        
    def profile_silero_cpu(self, audio_lengths=[1, 10, 30, 60]):
        """Profile CPU Silero VAD"""
        print("\n" + "="*60)
        print("Profiling CPU Silero VAD")
        print("="*60)
        
        # Initialize once
        vad = Silero(vad_onset=0.5, chunk_size=30)
        
        for duration in audio_lengths:
            print(f"\nTesting {duration}s audio...")
            
            # Generate test audio
            samples = int(duration * SAMPLE_RATE)
            audio = np.random.randn(samples) * 0.1
            waveform = torch.from_numpy(audio).unsqueeze(0)
            
            # Warmup
            _ = vad({"waveform": waveform, "sample_rate": SAMPLE_RATE})
            
            # Time multiple runs
            num_runs = 10
            times = []
            
            for _ in range(num_runs):
                start = time.perf_counter()
                segments = vad({"waveform": waveform, "sample_rate": SAMPLE_RATE})
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            realtime_factor = duration / avg_time
            
            print(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
            print(f"  Realtime factor: {realtime_factor:.1f}x")
            print(f"  Segments: {len(segments)}")
            
            self.results[f"silero_cpu_{duration}s"] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "realtime_factor": realtime_factor,
                "num_segments": len(segments)
            }
    
    def profile_mlx_vad(self, audio_lengths=[1, 10, 30, 60]):
        """Profile MLX VAD"""
        print("\n" + "="*60)
        print("Profiling MLX VAD")
        print("="*60)
        
        # Initialize once
        vad = SileroMLXVAD(vad_onset=0.5)
        
        for duration in audio_lengths:
            print(f"\nTesting {duration}s audio...")
            
            # Generate test audio
            samples = int(duration * SAMPLE_RATE)
            audio = np.random.randn(samples) * 0.1
            
            # Warmup
            _ = vad({"waveform": audio, "sample_rate": SAMPLE_RATE})
            
            # Time multiple runs
            num_runs = 10
            times = []
            
            for _ in range(num_runs):
                start = time.perf_counter()
                segments = vad({"waveform": audio, "sample_rate": SAMPLE_RATE})
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            std_time = np.std(times)
            realtime_factor = duration / avg_time
            
            print(f"  Average time: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
            print(f"  Realtime factor: {realtime_factor:.1f}x")
            print(f"  Segments: {len(segments)}")
            
            self.results[f"mlx_vad_{duration}s"] = {
                "avg_time": avg_time,
                "std_time": std_time,
                "realtime_factor": realtime_factor,
                "num_segments": len(segments)
            }
    
    def profile_mlx_operations(self):
        """Profile individual MLX operations"""
        print("\n" + "="*60)
        print("Profiling MLX Operations")
        print("="*60)
        
        # Test different audio lengths
        for samples in [16000, 160000, 480000]:  # 1s, 10s, 30s
            duration = samples / SAMPLE_RATE
            print(f"\nTesting {duration}s audio ({samples} samples)...")
            
            # Create test data
            audio = mx.random.normal((samples,))
            
            # Profile Conv1d
            conv = mx.nn.Conv1d(1, 40, kernel_size=512, stride=256)
            audio_conv = audio[None, :, None]  # Add batch and channel dims
            
            # Warmup
            _ = conv(audio_conv)
            mx.eval(_)
            
            # Time Conv1d
            times = []
            for _ in range(10):
                start = time.perf_counter()
                out = conv(audio_conv)
                mx.eval(out)  # Force evaluation
                end = time.perf_counter()
                times.append(end - start)
            
            conv_time = np.mean(times)
            print(f"  Conv1d: {conv_time*1000:.2f}ms")
            
            # Profile GRU
            seq_len = out.shape[1]
            gru = mx.nn.GRU(40, 64)
            
            # Warmup
            _ = gru(out)
            mx.eval(_)
            
            # Time GRU
            times = []
            for _ in range(10):
                start = time.perf_counter()
                gru_out = gru(out)
                mx.eval(gru_out)
                end = time.perf_counter()
                times.append(end - start)
            
            gru_time = np.mean(times)
            print(f"  GRU: {gru_time*1000:.2f}ms")
            
            # Profile memory transfer
            numpy_audio = np.random.randn(samples)
            
            times = []
            for _ in range(10):
                start = time.perf_counter()
                mx_array = mx.array(numpy_audio)
                mx.eval(mx_array)
                end = time.perf_counter()
                times.append(end - start)
            
            transfer_time = np.mean(times)
            print(f"  NumPy→MLX transfer: {transfer_time*1000:.2f}ms")
            
            # Total overhead
            total_time = conv_time + gru_time + transfer_time
            overhead_factor = duration / total_time
            print(f"  Total MLX ops: {total_time*1000:.2f}ms ({overhead_factor:.1f}x realtime)")
    
    def analyze_bottlenecks(self):
        """Analyze performance bottlenecks"""
        print("\n" + "="*60)
        print("Performance Analysis")
        print("="*60)
        
        # Compare realtime factors
        print("\nRealtime Factor Comparison:")
        print("-" * 40)
        
        durations = [1, 10, 30, 60]
        for duration in durations:
            cpu_key = f"silero_cpu_{duration}s"
            mlx_key = f"mlx_vad_{duration}s"
            
            if cpu_key in self.results and mlx_key in self.results:
                cpu_rt = self.results[cpu_key]["realtime_factor"]
                mlx_rt = self.results[mlx_key]["realtime_factor"]
                slowdown = cpu_rt / mlx_rt
                
                print(f"{duration}s audio:")
                print(f"  CPU Silero: {cpu_rt:.1f}x realtime")
                print(f"  MLX VAD: {mlx_rt:.1f}x realtime")
                print(f"  MLX slowdown: {slowdown:.2f}x slower")
        
        print("\nLikely bottlenecks:")
        print("1. Small model size - GPU overhead exceeds benefit")
        print("2. Memory transfers between CPU and GPU")
        print("3. MLX lazy evaluation not properly utilized")
        print("4. Architecture not optimized for MLX operations")

def main():
    """Run VAD performance profiling"""
    print("VAD Performance Profiling")
    print("=" * 60)
    
    profiler = VADProfiler()
    
    # Profile implementations
    profiler.profile_silero_cpu()
    profiler.profile_mlx_vad()
    profiler.profile_mlx_operations()
    
    # Analyze results
    profiler.analyze_bottlenecks()
    
    print("\nRecommendations:")
    print("1. For small audio chunks (<10s), use CPU Silero")
    print("2. Investigate batch processing for MLX efficiency")
    print("3. Consider model quantization to reduce memory bandwidth")
    print("4. Profile with larger models where GPU shines")

if __name__ == "__main__":
    main()