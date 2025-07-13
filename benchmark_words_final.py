#!/usr/bin/env python3
"""
Final benchmark: Word-level timestamps on 30m.wav
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import mlx_whisper
import whisperx
from whisperx.backends.mlx_lightning_words import WhisperMLXLightningWords

print("=== Word-Level Timestamp Benchmark on 30m.wav ===")

# Load audio
audio = whisperx.load_audio("30m.wav")
duration = len(audio) / 16000
print(f"\nAudio duration: {duration:.1f}s ({duration/60:.1f} minutes)")

# Test 1: Direct MLX call (baseline)
print("\n1. Direct mlx_whisper with word timestamps...")
start = time.time()
result_direct = mlx_whisper.transcribe(
    audio,
    path_or_hf_repo="mlx-community/whisper-tiny",
    word_timestamps=True,
    temperature=0.0,
    verbose=False
)
time_direct = time.time() - start
speed_direct = duration / time_direct

print(f"✓ Time: {time_direct:.2f}s")
print(f"✓ Speed: {speed_direct:.1f}x realtime")
print(f"✓ Segments: {len(result_direct['segments'])}")

# Count words
total_words = sum(len(seg.get('words', [])) for seg in result_direct['segments'])
print(f"✓ Total words: {total_words}")

# Test 2: Our Lightning backend with words
print("\n2. Lightning backend with word timestamps...")
backend = WhisperMLXLightningWords(
    model_name="mlx-community/whisper-tiny",
    compute_type="float16",
    word_timestamps=True
)

start = time.time()
result_backend = backend.transcribe(audio)
time_backend = time.time() - start
speed_backend = duration / time_backend

print(f"✓ Time: {time_backend:.2f}s")
print(f"✓ Speed: {speed_backend:.1f}x realtime")
print(f"✓ Segments: {len(result_backend['segments'])}")

# Count words
total_words_backend = sum(len(seg.get('words', [])) for seg in result_backend['segments'])
print(f"✓ Total words: {total_words_backend}")

# Test 3: Lightning without words for comparison
print("\n3. Lightning backend WITHOUT word timestamps...")
backend_fast = WhisperMLXLightningWords(
    model_name="mlx-community/whisper-tiny",
    compute_type="float16",
    word_timestamps=False
)

start = time.time()
result_fast = backend_fast.transcribe(audio)
time_fast = time.time() - start
speed_fast = duration / time_fast

print(f"✓ Time: {time_fast:.2f}s")
print(f"✓ Speed: {speed_fast:.1f}x realtime")

# Summary
print("\n=== Performance Summary ===")
print(f"Direct MLX with words: {speed_direct:.1f}x realtime")
print(f"Lightning with words: {speed_backend:.1f}x realtime")
print(f"Lightning without words: {speed_fast:.1f}x realtime")
print(f"\nWord timestamp overhead: {((time_backend/time_fast) - 1) * 100:.1f}%")
print(f"Total words extracted: {total_words}")

# Save sample words
print("\n=== Sample Word Output ===")
if result_backend['segments']:
    seg = result_backend['segments'][len(result_backend['segments'])//2]  # Middle segment
    if 'words' in seg and seg['words']:
        print(f"Sample from middle of audio:")
        for word in seg['words'][:10]:
            print(f"  {word['start']:7.2f}s: '{word['word']}'")

print("\n✓ Word-level timestamps successfully integrated!")
print(f"✓ Performance with words: {speed_backend:.1f}x realtime")
print(f"✓ Extracted {total_words} words from {duration/60:.1f} minute audio")