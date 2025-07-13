#!/usr/bin/env python3
"""Quick benchmark comparison between whisperx-mlx and lightning-whisper-mlx"""
import time
import os
import sys
import mlx.core as mx
from datetime import datetime

def run_whisperx_benchmark():
    """Run whisperx-mlx benchmark with small model"""
    print("=== WhisperX-MLX Benchmark (small model) ===")
    
    # Clear cache
    mx.clear_cache()
    
    # Remove old output if exists
    os.system("rm -rf tmp_benchmark")
    
    # Time the execution
    start_time = time.time()
    result = os.system("python -m whisperx 30m.wav --model small --backend mlx --batch_size 16 --output_dir tmp_benchmark > /dev/null 2>&1")
    end_time = time.time()
    
    elapsed = end_time - start_time
    rtf = 1800 / elapsed  # 30 minutes = 1800 seconds
    
    # Clean up
    os.system("rm -rf tmp_benchmark")
    
    return elapsed, rtf, (result == 0)

def run_lightning_benchmark():
    """Run lightning-whisper-mlx benchmark with small model"""
    print("\n=== Lightning-Whisper-MLX Benchmark (small model) ===")
    
    # Add lightning to path
    sys.path.insert(0, '/Users/dmalson/whisperx-mlx/lightning-whisper-mlx/lightning-whisper-mlx-env/lib/python3.11/site-packages')
    
    try:
        from lightning_whisper_mlx import LightningWhisperMLX
        
        # Clear cache
        mx.clear_cache()
        
        # Initialize and time
        start_init = time.time()
        whisper = LightningWhisperMLX(model="small", batch_size=12)
        init_time = time.time() - start_init
        
        start_time = time.time()
        result = whisper.transcribe("30m.wav")
        end_time = time.time()
        
        elapsed = end_time - start_time
        rtf = 1800 / elapsed
        
        return elapsed, rtf, True
    except Exception as e:
        print(f"Error: {e}")
        return None, None, False

def main():
    print(f"Starting benchmark at {datetime.now()}")
    print(f"Audio file: 30m.wav (30 minutes)")
    print(f"MLX device: {mx.default_device()}")
    print()
    
    # Run whisperx benchmark
    wx_time, wx_rtf, wx_success = run_whisperx_benchmark()
    if wx_success:
        print(f"WhisperX-MLX: {wx_time:.2f}s ({wx_rtf:.2f}x realtime)")
    else:
        print("WhisperX-MLX: Failed")
    
    # Run lightning benchmark
    lw_time, lw_rtf, lw_success = run_lightning_benchmark()
    if lw_success:
        print(f"Lightning-Whisper-MLX: {lw_time:.2f}s ({lw_rtf:.2f}x realtime)")
    else:
        print("Lightning-Whisper-MLX: Failed")
    
    # Compare
    if wx_success and lw_success:
        speedup = wx_time / lw_time
        print(f"\n=== Comparison ===")
        print(f"Lightning is {speedup:.2f}x faster than current WhisperX-MLX")
        print(f"WhisperX-MLX: {wx_rtf:.2f}x realtime")
        print(f"Lightning-Whisper-MLX: {lw_rtf:.2f}x realtime")

if __name__ == "__main__":
    main()