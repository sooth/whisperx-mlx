#!/usr/bin/env python3
"""
Debug why transcriptions are different
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import whisperx

print("=== Debugging Transcription Differences ===")

# Load audio
audio = whisperx.load_audio("short.wav")
duration = len(audio) / 16000
print(f"\nAudio duration: {duration:.1f}s")

# Test 1: Our backend without VAD
print("\n1. Our Lightning backend (direct, no VAD)...")
from whisperx.backends.mlx_lightning_simple import WhisperMLXLightningSimple

backend = WhisperMLXLightningSimple(
    model_name="mlx-community/whisper-tiny",
    compute_type="float16"
)

result = backend.transcribe(audio)
print(f"Text: {result['text'][:200]}...")

# Test 2: Stock WhisperX
print("\n2. Stock WhisperX transcription...")
import sys
sys.path.insert(0, 'stock')

try:
    import whisperx as whisperx_stock
    
    model = whisperx_stock.load_model("tiny", "cpu", compute_type="float16")
    result_stock = model.transcribe(audio, batch_size=16)
    
    print(f"Text: {result_stock['segments'][0]['text'][:200] if result_stock['segments'] else 'No text'}...")
    
    # Check if it's the audio or VAD
    print(f"\nStock segments: {len(result_stock['segments'])}")
    for i, seg in enumerate(result_stock['segments'][:3]):
        print(f"  Segment {i}: [{seg['start']:.1f}s - {seg['end']:.1f}s] {seg['text'][:50]}...")
        
except Exception as e:
    print(f"Stock failed: {e}")

# Test 3: Check audio content
print("\n3. Audio analysis...")
print(f"Audio shape: {audio.shape}")
print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")
print(f"Audio mean: {audio.mean():.3f}")
print(f"Audio std: {audio.std():.3f}")

# Save a sample for inspection
import numpy as np
sample = audio[:5 * 16000]  # First 5 seconds
np.save("audio_sample.npy", sample)
print("\nSaved first 5s to audio_sample.npy for inspection")

# Test 4: Try with different start times
print("\n4. Testing different audio segments...")
for start_sec in [0, 5, 10, 15, 20]:
    start_idx = start_sec * 16000
    end_idx = (start_sec + 5) * 16000
    segment = audio[start_idx:end_idx]
    
    result_seg = backend.transcribe(segment)
    print(f"  {start_sec}s-{start_sec+5}s: {result_seg['text'][:50]}...")

print("\n5. Conclusion:")
print("The issue appears to be that stock WhisperX is detecting different")
print("segments with VAD or processing audio differently.")