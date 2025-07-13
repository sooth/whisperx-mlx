#!/usr/bin/env python3
"""
Debug word timestamp issue
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print("1. Testing direct mlx_whisper call...")
import mlx_whisper

try:
    result = mlx_whisper.transcribe(
        "short.wav",
        path_or_hf_repo="mlx-community/whisper-tiny",
        word_timestamps=True,
        verbose=False
    )
    print("✓ Direct call works!")
    print(f"  Segments: {len(result.get('segments', []))}")
    if result.get('segments'):
        print(f"  First segment has words: {'words' in result['segments'][0]}")
except Exception as e:
    print(f"✗ Direct call failed: {e}")

print("\n2. Testing our backend...")
from whisperx.backends.mlx_lightning_words import WhisperMLXLightningWords
import whisperx

try:
    audio = whisperx.load_audio("short.wav")
    backend = WhisperMLXLightningWords(
        model_name="mlx-community/whisper-tiny",
        compute_type="float16",
        word_timestamps=True
    )
    
    # Test just the word transcription method
    result = backend._transcribe_with_words(audio)
    print("✓ Backend works!")
    print(f"  Segments: {len(result.get('segments', []))}")
    if result.get('segments'):
        print(f"  First segment has words: {len(result['segments'][0].get('words', []))} words")
except Exception as e:
    print(f"✗ Backend failed: {e}")
    import traceback
    traceback.print_exc()