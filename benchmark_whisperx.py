#!/usr/bin/env python3
"""Benchmark script for whisperx-mlx implementation"""
import time
import psutil
import mlx.core as mx
import os

def benchmark_whisperx():
    """Benchmark whisperx-mlx on 30m.wav"""
    print("=== WhisperX-MLX Benchmark ===")
    print(f"Audio file: 30m.wav (30 minutes)")
    print(f"MLX device: {mx.default_device()}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()
    
    # Test different configurations
    configs = [
        ("small", "mlx", 1),
        ("small", "mlx", 16),
        ("base", "mlx", 1),
        ("base", "mlx", 16),
        ("large-v3", "mlx", 1),
        ("large-v3", "mlx", 16),
    ]
    
    results = []
    
    for model, backend, batch_size in configs:
        print(f"\nTesting: model={model}, backend={backend}, batch_size={batch_size}")
        
        # Clear MLX cache
        mx.clear_cache()
        
        cmd = f"python -m whisperx 30m.wav --model {model} --backend {backend} --batch_size {batch_size} --output_dir tmp_benchmark"
        
        start_time = time.time()
        os.system(cmd)
        end_time = time.time()
        
        elapsed = end_time - start_time
        rtf = 1800 / elapsed  # 30 minutes = 1800 seconds
        
        print(f"Time: {elapsed:.2f}s, RTF: {rtf:.2f}x")
        results.append({
            "model": model,
            "backend": backend, 
            "batch_size": batch_size,
            "time": elapsed,
            "rtf": rtf
        })
        
        # Clean up
        os.system("rm -rf tmp_benchmark")
    
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['model']:10} batch={r['batch_size']:2d}: {r['time']:6.2f}s ({r['rtf']:5.2f}x realtime)")
    
    return results

if __name__ == "__main__":
    benchmark_whisperx()