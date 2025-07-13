#!/usr/bin/env python3
"""Final comparison of word-level timestamps between implementations"""

import json
import time
import numpy as np
import whisperx
import torch

# Test parameters
audio_file = "short.wav"
model_size = "tiny"
device = "mps" if torch.backends.mps.is_available() else "cpu"
compute_type = "float32"

print(f"Final Word-Level Timestamp Comparison on {audio_file}")
print(f"Model: {model_size}, Device: {device}")
print("="*80)

# Load audio once
audio = whisperx.load_audio(audio_file)

results = {}

# Test 1: Stock WhisperX with VAD
print("\n1. Stock WhisperX (with VAD + alignment):")
start_time = time.time()

model = whisperx.load_model(model_size, device, compute_type=compute_type, backend="mlx")
result = model.transcribe(audio, batch_size=16)
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result_aligned = whisperx.align(result["segments"], model_a, metadata, audio, device)

stock_vad_time = time.time() - start_time
stock_vad_words = []
for seg in result_aligned.get("segments", []):
    stock_vad_words.extend(seg.get("words", []))

print(f"  Time: {stock_vad_time:.2f}s")
print(f"  Words: {len(stock_vad_words)}")
print(f"  Transcription: {' '.join([w['word'] for w in stock_vad_words[:10]])}...")

results["stock_with_vad"] = {
    "time": stock_vad_time,
    "word_count": len(stock_vad_words),
    "first_10_words": [w['word'] for w in stock_vad_words[:10]]
}

# Test 2: Lightning without VAD (only way to get word timestamps currently)
print("\n2. Lightning (no VAD + integrated alignment):")
start_time = time.time()

model = whisperx.load_model(model_size, device, compute_type=compute_type, 
                          backend="mlx_lightning", word_timestamps=True, vad_method=None)
result = model.transcribe(audio, align_words=True)

lightning_no_vad_time = time.time() - start_time
lightning_words = []
for seg in result.get("segments", []):
    lightning_words.extend(seg.get("words", []))

print(f"  Time: {lightning_no_vad_time:.2f}s")
print(f"  Words: {len(lightning_words)}")
print(f"  Transcription: {' '.join([w['word'] for w in lightning_words[:10]])}...")

results["lightning_no_vad"] = {
    "time": lightning_no_vad_time,
    "word_count": len(lightning_words),
    "first_10_words": [w['word'] for w in lightning_words[:10]]
}

# Test 3: Lightning without word alignment (fast mode)
print("\n3. Lightning (with VAD, no word alignment - fast mode):")
start_time = time.time()

model = whisperx.load_model(model_size, device, compute_type=compute_type, backend="mlx_lightning")
result = model.transcribe(audio, batch_size=16)

lightning_fast_time = time.time() - start_time
segments_text = ' '.join([seg['text'].strip() for seg in result.get('segments', [])])

print(f"  Time: {lightning_fast_time:.2f}s")
print(f"  Segments: {len(result.get('segments', []))}")
print(f"  Transcription: {segments_text[:100]}...")

results["lightning_fast"] = {
    "time": lightning_fast_time,
    "segment_count": len(result.get('segments', [])),
    "sample_text": segments_text[:100]
}

# Summary
print("\n" + "="*80)
print("SUMMARY:")
print("="*80)

print(f"\n1. Stock WhisperX (VAD + align): {stock_vad_time:.2f}s for {len(stock_vad_words)} words")
print(f"2. Lightning (no VAD + align): {lightning_no_vad_time:.2f}s for {len(lightning_words)} words")
print(f"3. Lightning (VAD, no align): {lightning_fast_time:.2f}s (transcription only)")

print(f"\nSpeedup vs Stock:")
print(f"  - Lightning with alignment: {stock_vad_time/lightning_no_vad_time:.2f}x")
print(f"  - Lightning fast mode: {stock_vad_time/lightning_fast_time:.2f}x")

# Save results
with open("final_word_comparison.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nResults saved to final_word_comparison.json")

# Show transcription differences
print("\n" + "="*80)
print("TRANSCRIPTION COMPARISON (first 10 words):")
print("="*80)
print(f"Stock:     {' '.join([w['word'] for w in stock_vad_words[:10]])}")
print(f"Lightning: {' '.join([w['word'] for w in lightning_words[:10]])}")