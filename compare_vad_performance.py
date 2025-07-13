#!/usr/bin/env python3
"""Compare VAD performance impact on 30m.wav transcription"""

import time
import whisperx
import torch

def test_vad_performance(vad_method, model_size="tiny"):
    """Test transcription with specific VAD"""
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"\nTesting {vad_method} VAD with {model_size} model...")
    
    # Load audio
    audio = whisperx.load_audio("30m.wav")
    duration = len(audio) / 16000
    
    # Time the full pipeline
    start_time = time.time()
    
    # Load model with specific VAD
    model = whisperx.load_model(
        model_size,
        device,
        compute_type="float32",
        backend="lightning",
        vad_method=vad_method
    )
    
    # Transcribe
    result = model.transcribe(audio, batch_size=16)
    
    total_time = time.time() - start_time
    realtime_factor = duration / total_time
    
    # Get segment count
    segments = len(result.get('segments', []))
    
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    print(f"  Segments: {segments}")
    
    return {
        'vad': vad_method,
        'time': total_time,
        'realtime': realtime_factor,
        'segments': segments
    }

def main():
    """Compare VAD methods on 30m audio"""
    
    print("VAD Performance Comparison on 30m.wav")
    print("=" * 60)
    
    try:
        # Test both VAD methods
        results = []
        
        # Test Silero (new default)
        silero_result = test_vad_performance("silero")
        results.append(silero_result)
        
        # Test PyAnnote (old default)
        pyannote_result = test_vad_performance("pyannote")
        results.append(pyannote_result)
        
        # Calculate improvement
        if len(results) == 2:
            speedup = pyannote_result['time'] / silero_result['time']
            
            print(f"\n{'='*60}")
            print("SUMMARY")
            print(f"{'='*60}")
            print(f"\nSilero VAD:   {silero_result['realtime']:.2f}x realtime")
            print(f"PyAnnote VAD: {pyannote_result['realtime']:.2f}x realtime")
            print(f"\n✅ Silero is {speedup:.2f}x faster than PyAnnote!")
            print(f"   Time saved: {pyannote_result['time'] - silero_result['time']:.1f} seconds")
            
    except FileNotFoundError:
        print("\n30m.wav not found. Creating comparison with short.wav instead...")
        
        # Use short.wav for comparison
        audio = whisperx.load_audio("short.wav")
        duration = len(audio) / 16000
        print(f"Using short.wav ({duration:.1f}s)")
        
        # Quick comparison
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        
        # Silero
        start = time.time()
        model = whisperx.load_model("tiny", device, backend="lightning", vad_method="silero")
        result = model.transcribe(audio)
        silero_time = time.time() - start
        
        # PyAnnote
        start = time.time()
        model = whisperx.load_model("tiny", device, backend="lightning", vad_method="pyannote")
        result = model.transcribe(audio)
        pyannote_time = time.time() - start
        
        print(f"\nSilero:   {silero_time:.2f}s ({duration/silero_time:.1f}x realtime)")
        print(f"PyAnnote: {pyannote_time:.2f}s ({duration/pyannote_time:.1f}x realtime)")
        print(f"\n✅ Silero is {pyannote_time/silero_time:.2f}x faster!")

if __name__ == "__main__":
    main()