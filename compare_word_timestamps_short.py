#!/usr/bin/env python3
"""Compare word-level timestamps between stock WhisperX and our Lightning implementation on short.wav"""

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

print(f"Comparing word-level timestamps on {audio_file}")
print(f"Model: {model_size}, Device: {device}")
print("="*80)

# Test 1: Stock WhisperX with word timestamps
print("\n1. Stock WhisperX (with word-level timestamps):")
start_time = time.time()

# Load model
model = whisperx.load_model(model_size, device, compute_type=compute_type, backend="mlx")

# Transcribe with word timestamps
audio = whisperx.load_audio(audio_file)
result_stock = model.transcribe(audio, batch_size=16)

# Align for word-level timestamps
model_a, metadata = whisperx.load_align_model(language_code=result_stock["language"], device=device)
result_stock_aligned = whisperx.align(result_stock["segments"], model_a, metadata, audio, device, return_char_alignments=False)

stock_time = time.time() - start_time
print(f"Time: {stock_time:.2f}s")
print(f"Detected language: {result_stock['language']}")

# Test 2: Our Lightning implementation with word timestamps
print("\n2. Our Lightning Implementation (with word-level timestamps):")
start_time = time.time()

# Load Lightning model with word timestamps
model_lightning = whisperx.load_model(model_size, device, compute_type=compute_type, backend="mlx_lightning", word_timestamps=True)

# Transcribe with word timestamps
result_lightning = model_lightning.transcribe(audio, batch_size=16, align_words=True)

lightning_time = time.time() - start_time
print(f"Time: {lightning_time:.2f}s")
print(f"Detected language: {result_lightning['language']}")

# Compare results
print("\n" + "="*80)
print("WORD-LEVEL COMPARISON:")
print("="*80)

# Extract words from stock aligned result
stock_words = []
for segment in result_stock_aligned.get("segments", []):
    for word in segment.get("words", []):
        stock_words.append({
            "word": word["word"],
            "start": word["start"],
            "end": word["end"]
        })

# Extract words from lightning result
lightning_words = []
for segment in result_lightning.get("segments", []):
    for word in segment.get("words", []):
        lightning_words.append({
            "word": word["word"],
            "start": word["start"],
            "end": word["end"]
        })

# Display side-by-side comparison
print(f"\nStock WhisperX: {len(stock_words)} words")
print(f"Lightning:      {len(lightning_words)} words")
print("\n{:<30} | {:<30}".format("Stock WhisperX", "Lightning Implementation"))
print("-"*61)

max_words = max(len(stock_words), len(lightning_words))
for i in range(max_words):
    stock_str = ""
    lightning_str = ""
    
    if i < len(stock_words):
        w = stock_words[i]
        stock_str = f"{w['word']} [{w['start']:.2f}-{w['end']:.2f}]"
    
    if i < len(lightning_words):
        w = lightning_words[i]
        lightning_str = f"{w['word']} [{w['start']:.2f}-{w['end']:.2f}]"
    
    print("{:<30} | {:<30}".format(stock_str, lightning_str))

# Calculate timing differences
print("\n" + "="*80)
print("TIMING ANALYSIS:")
print("="*80)

if len(stock_words) == len(lightning_words):
    timing_diffs = []
    for i in range(len(stock_words)):
        start_diff = abs(stock_words[i]["start"] - lightning_words[i]["start"])
        end_diff = abs(stock_words[i]["end"] - lightning_words[i]["end"])
        timing_diffs.extend([start_diff, end_diff])
    
    avg_diff = np.mean(timing_diffs)
    max_diff = np.max(timing_diffs)
    print(f"Average timing difference: {avg_diff:.3f}s")
    print(f"Maximum timing difference: {max_diff:.3f}s")
else:
    print("Different number of words detected - cannot compute timing differences")

# Performance comparison
print("\n" + "="*80)
print("PERFORMANCE SUMMARY:")
print("="*80)
print(f"Stock WhisperX:  {stock_time:.2f}s")
print(f"Lightning:       {lightning_time:.2f}s")
print(f"Speedup:         {stock_time/lightning_time:.2f}x")

# Save detailed results
results = {
    "audio_file": audio_file,
    "model_size": model_size,
    "stock": {
        "time": stock_time,
        "words": stock_words,
        "segments": result_stock_aligned.get("segments", [])
    },
    "lightning": {
        "time": lightning_time,
        "words": lightning_words,
        "segments": result_lightning.get("segments", [])
    },
    "comparison": {
        "speedup": stock_time/lightning_time,
        "stock_word_count": len(stock_words),
        "lightning_word_count": len(lightning_words)
    }
}

with open("word_timestamps_comparison_short.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nDetailed results saved to word_timestamps_comparison_short.json")