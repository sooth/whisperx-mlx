#!/usr/bin/env python3
"""Benchmark optimized WhisperX-MLX implementation"""
import time
import mlx.core as mx
from datetime import datetime
import os

def benchmark_optimized():
    """Benchmark the optimized implementation"""
    print(f"=== Optimized WhisperX-MLX Benchmark ===")
    print(f"Started at: {datetime.now()}")
    print(f"Audio file: 30m.wav (30 minutes)")
    print(f"MLX device: {mx.default_device()}")
    print()
    
    # Test configurations
    configs = [
        ("small", "standard", 1, None),      # Baseline
        ("small", "batch", 16, None),        # Optimized
        ("base", "batch", 16, None),         # Base model optimized
        ("tiny", "batch", 16, None),         # Tiny model optimized
    ]
    
    results = []
    
    for model, backend, batch_size, quant in configs:
        config_name = f"{model} {backend} batch={batch_size}"
        if quant:
            config_name += f" {quant}"
        
        print(f"\nTesting: {config_name}")
        
        # Clear MLX cache
        mx.clear_cache()
        
        # Build command
        cmd = f"python -m whisperx 30m.wav --model {model} --backend {backend} --batch_size {batch_size}"
        if quant:
            cmd += f" --quantization {quant}"
        cmd += " --output_dir tmp_benchmark > /dev/null 2>&1"
        
        # Time execution
        start_time = time.time()
        result = os.system(cmd)
        end_time = time.time()
        
        elapsed = end_time - start_time
        rtf = 1800 / elapsed  # 30 minutes = 1800 seconds
        
        if result == 0:
            print(f"Success! Time: {elapsed:.2f}s, RTF: {rtf:.2f}x")
            results.append({
                "config": config_name,
                "time": elapsed,
                "rtf": rtf,
                "success": True
            })
        else:
            print(f"Failed!")
            results.append({
                "config": config_name,
                "time": elapsed,
                "rtf": rtf,
                "success": False
            })
        
        # Clean up
        os.system("rm -rf tmp_benchmark")
    
    # Summary
    print("\n=== Summary ===")
    print(f"{'Configuration':30} {'Time':>8} {'RTF':>8} {'Status':>8}")
    print("-" * 60)
    
    baseline_time = None
    for r in results:
        status = "OK" if r['success'] else "FAIL"
        print(f"{r['config']:30} {r['time']:>6.2f}s {r['rtf']:>6.2f}x {status:>8}")
        
        if baseline_time is None and r['success']:
            baseline_time = r['time']
        elif baseline_time and r['success']:
            speedup = baseline_time / r['time']
            print(f"  → {speedup:.2f}x faster than baseline")
    
    return results

def quick_test():
    """Quick test with tiny model"""
    print("Running quick test with tiny model...")
    mx.clear_cache()
    
    start = time.time()
    os.system("python -m whisperx 30m.wav --model tiny --backend batch --batch_size 16 --output_dir tmp_test > /dev/null 2>&1")
    elapsed = time.time() - start
    
    rtf = 1800 / elapsed
    print(f"Tiny model test: {elapsed:.2f}s ({rtf:.2f}x realtime)")
    os.system("rm -rf tmp_test")

if __name__ == "__main__":
    # Run quick test first
    quick_test()
    print("\n" + "="*60 + "\n")
    
    # Run full benchmark
    benchmark_optimized()