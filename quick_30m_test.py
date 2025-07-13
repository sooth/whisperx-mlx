#!/usr/bin/env python3
"""Quick speed test on 30m.wav for Lightning with word alignment"""

import time
import whisperx
import torch

# Test parameters
audio_file = "30m.wav"
model_size = "tiny"
device = "mps" if torch.backends.mps.is_available() else "cpu"
compute_type = "float32"

print(f"Quick Speed Test: Lightning with Batch Word Alignment on {audio_file}")
print("="*80)

# Load audio
print("Loading audio...")
audio = whisperx.load_audio(audio_file)
duration = len(audio) / 16000
print(f"Audio duration: {duration/60:.2f} minutes")

# Test Lightning with batch word alignment
print("\nTesting Lightning with batch word alignment...")
start_time = time.time()

model = whisperx.load_model(
    model_size,
    device,
    compute_type=compute_type,
    backend="mlx_lightning",
    word_timestamps=True
)

result = model.transcribe(
    audio, 
    batch_size=16,
    word_timestamps=True
)

elapsed_time = time.time() - start_time

# Count results
word_count = 0
for seg in result.get("segments", []):
    word_count += len(seg.get("words", []))

realtime_factor = duration / elapsed_time

print("\nResults:")
print(f"  Processing time: {elapsed_time:.2f}s")
print(f"  Realtime factor: {realtime_factor:.2f}x")
print(f"  Segments: {len(result.get('segments', []))}")
print(f"  Words detected: {word_count}")

# Show sample words
if word_count > 0:
    print("\nSample words with timestamps:")
    all_words = []
    for seg in result.get("segments", []):
        all_words.extend(seg.get("words", []))
    
    for i, word in enumerate(all_words[:5]):
        print(f"  {i+1}. '{word['word']}' at {word['start']:.2f}s")

print(f"\n✅ Lightning with word alignment achieves {realtime_factor:.2f}x realtime speed on 30-minute audio!")