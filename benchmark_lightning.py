#!/usr/bin/env python3
"""Benchmark script for lightning-whisper-mlx implementation"""
import sys
import time
import psutil
import mlx.core as mx

# Add the virtual environment to the path
sys.path.insert(0, '/Users/dmalson/whisperx-mlx/lightning-whisper-mlx/lightning-whisper-mlx-env/lib/python3.11/site-packages')

from lightning_whisper_mlx import LightningWhisperMLX

def benchmark_lightning():
    """Benchmark lightning-whisper-mlx on 30m.wav"""
    print("=== Lightning-Whisper-MLX Benchmark ===")
    print(f"Audio file: 30m.wav (30 minutes)")
    print(f"MLX device: {mx.default_device()}")
    print(f"CPU cores: {psutil.cpu_count()}")
    print(f"Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()
    
    # Test different configurations
    configs = [
        ("small", 12, None),  # Default quantization
        ("small", 1, None),
        ("small", 24, None),
        ("base", 12, None),
        ("base", 1, None),
        ("base", 24, None),
        ("large-v3", 12, None),
        ("large-v3", 1, None),
        ("large-v3", 6, None),
        ("distil-large-v3", 12, None),  # Distilled model
        ("small", 12, "4bit"),  # 4-bit quantization
        ("base", 12, "4bit"),
    ]
    
    results = []
    
    for model, batch_size, quant in configs:
        print(f"\nTesting: model={model}, batch_size={batch_size}, quantization={quant}")
        
        # Clear MLX cache
        mx.clear_cache()
        
        try:
            # Initialize model
            start_init = time.time()
            whisper = LightningWhisperMLX(model=model, batch_size=batch_size, quant=quant)
            init_time = time.time() - start_init
            print(f"Model initialization: {init_time:.2f}s")
            
            # Transcribe
            start_time = time.time()
            result = whisper.transcribe("30m.wav")
            end_time = time.time()
            
            elapsed = end_time - start_time
            rtf = 1800 / elapsed  # 30 minutes = 1800 seconds
            
            print(f"Transcription time: {elapsed:.2f}s, RTF: {rtf:.2f}x")
            print(f"Text preview: {result['text'][:100]}...")
            
            results.append({
                "model": model,
                "batch_size": batch_size,
                "quantization": quant or "none",
                "init_time": init_time,
                "time": elapsed,
                "rtf": rtf
            })
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    print("\n=== Summary ===")
    for r in results:
        print(f"{r['model']:15} batch={r['batch_size']:2d} quant={r['quantization']:4}: {r['time']:6.2f}s ({r['rtf']:5.2f}x realtime)")
    
    return results

if __name__ == "__main__":
    benchmark_lightning()