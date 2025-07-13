#!/usr/bin/env python3
"""Quick test of Turbo model performance"""

import time
import whisperx
import torch

def test_turbo():
    """Quick turbo model test"""
    print("Testing Whisper Turbo Model")
    print("=" * 60)
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Test models
    models = [
        ("large-v3", "Standard large model"),
        ("turbo", "Turbo optimized model"),
        ("distil-large-v3", "Distilled model")
    ]
    
    # Load test audio
    try:
        audio = whisperx.load_audio("30m.wav")
        duration = len(audio) / 16000
        print(f"Audio duration: {duration/60:.1f} minutes\n")
    except:
        audio = whisperx.load_audio("short.wav")
        duration = len(audio) / 16000
        print(f"Audio duration: {duration:.1f} seconds\n")
    
    results = []
    
    for model_name, desc in models:
        print(f"\n{model_name}: {desc}")
        print("-" * 40)
        
        try:
            # Load model
            start = time.time()
            model = whisperx.load_model(
                model_name,
                device=device,
                compute_type="float16",
                backend="lightning"
            )
            load_time = time.time() - start
            print(f"Load time: {load_time:.2f}s")
            
            # Transcribe
            start = time.time()
            result = model.transcribe(audio, batch_size=16)
            trans_time = time.time() - start
            
            # Calculate metrics
            realtime_factor = duration / trans_time
            segments = result.get('segments', [])
            words = sum(len(s.get('text', '').split()) for s in segments)
            
            print(f"Transcription time: {trans_time:.2f}s")
            print(f"Realtime factor: {realtime_factor:.1f}x")
            print(f"Words: {words}")
            
            results.append({
                "model": model_name,
                "time": trans_time,
                "rtf": realtime_factor,
                "words": words
            })
            
        except Exception as e:
            print(f"Error: {e}")
            results.append({
                "model": model_name,
                "error": str(e)
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    valid_results = [r for r in results if "error" not in r]
    if len(valid_results) >= 2:
        # Compare turbo to large-v3
        large = next((r for r in valid_results if r["model"] == "large-v3"), None)
        turbo = next((r for r in valid_results if r["model"] == "turbo"), None)
        
        if large and turbo:
            speedup = turbo["rtf"] / large["rtf"]
            print(f"\nTurbo vs Large-v3:")
            print(f"  Turbo: {turbo['rtf']:.1f}x realtime")
            print(f"  Large: {large['rtf']:.1f}x realtime")
            print(f"  Speedup: {speedup:.2f}x faster")
            print(f"  Time saved: {large['time'] - turbo['time']:.1f}s")

if __name__ == "__main__":
    test_turbo()