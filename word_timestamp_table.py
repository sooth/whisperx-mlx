#!/usr/bin/env python3
"""Create a detailed word-by-word timestamp comparison table"""

import json
import time
import numpy as np
import whisperx
import torch
from tabulate import tabulate

# Test parameters
audio_file = "short.wav"
model_size = "tiny"
device = "mps" if torch.backends.mps.is_available() else "cpu"
compute_type = "float32"

print(f"Generating word-by-word timestamp comparison for {audio_file}")
print("="*80)

# Load audio once
audio = whisperx.load_audio(audio_file)

# Get stock WhisperX results with VAD
print("\nProcessing Stock WhisperX...")
model_stock = whisperx.load_model(model_size, device, compute_type=compute_type, backend="mlx")
result_stock = model_stock.transcribe(audio, batch_size=16)
model_a, metadata = whisperx.load_align_model(language_code=result_stock["language"], device=device)
result_stock_aligned = whisperx.align(result_stock["segments"], model_a, metadata, audio, device)

stock_words = []
for seg in result_stock_aligned.get("segments", []):
    for word in seg.get("words", []):
        stock_words.append({
            "word": word["word"],
            "start": word["start"],
            "end": word["end"],
            "duration": word["end"] - word["start"]
        })

# Get Lightning results without VAD (to get word timestamps)
print("Processing Lightning WhisperX...")
model_lightning = whisperx.load_model(model_size, device, compute_type=compute_type, 
                                    backend="mlx_lightning", word_timestamps=True, vad_method=None)
result_lightning = model_lightning.transcribe(audio, align_words=True)

lightning_words = []
for seg in result_lightning.get("segments", []):
    for word in seg.get("words", []):
        lightning_words.append({
            "word": word["word"],
            "start": word["start"],
            "end": word["end"],
            "duration": word["end"] - word["start"]
        })

# Create comparison table
print("\n" + "="*120)
print("WORD-BY-WORD TIMESTAMP COMPARISON")
print("="*120)

# Prepare table data
table_data = []
max_len = max(len(stock_words), len(lightning_words))

for i in range(max_len):
    row = []
    
    # Stock column
    if i < len(stock_words):
        w = stock_words[i]
        row.extend([
            i + 1,
            w["word"],
            f"{w['start']:.3f}",
            f"{w['end']:.3f}",
            f"{w['duration']:.3f}"
        ])
    else:
        row.extend([i + 1, "", "", "", ""])
    
    # Lightning column
    if i < len(lightning_words):
        w = lightning_words[i]
        row.extend([
            w["word"],
            f"{w['start']:.3f}",
            f"{w['end']:.3f}",
            f"{w['duration']:.3f}"
        ])
    else:
        row.extend(["", "", "", ""])
    
    table_data.append(row)

# Print table
headers = [
    "#", 
    "Stock Word", "Start", "End", "Duration",
    "Lightning Word", "Start", "End", "Duration"
]

print(tabulate(table_data[:40], headers=headers, tablefmt="grid"))

if len(table_data) > 40:
    print(f"\n... showing first 40 of {len(table_data)} rows ...")

# Summary statistics
print("\n" + "="*120)
print("SUMMARY STATISTICS")
print("="*120)

print(f"\nStock WhisperX:")
print(f"  Total words: {len(stock_words)}")
if stock_words:
    print(f"  Time range: {stock_words[0]['start']:.3f}s - {stock_words[-1]['end']:.3f}s")
    print(f"  Average word duration: {np.mean([w['duration'] for w in stock_words]):.3f}s")

print(f"\nLightning:")
print(f"  Total words: {len(lightning_words)}")
if lightning_words:
    print(f"  Time range: {lightning_words[0]['start']:.3f}s - {lightning_words[-1]['end']:.3f}s")
    print(f"  Average word duration: {np.mean([w['duration'] for w in lightning_words]):.3f}s")

# Find matching words
if stock_words and lightning_words:
    print("\n" + "="*120)
    print("MATCHING WORDS ANALYSIS")
    print("="*120)
    
    # Create word maps
    stock_text = ' '.join([w['word'] for w in stock_words])
    lightning_text = ' '.join([w['word'] for w in lightning_words])
    
    print(f"\nStock text sample: {stock_text[:100]}...")
    print(f"Lightning text sample: {lightning_text[:100]}...")
    
    # Find common words
    stock_word_set = set([w['word'].lower() for w in stock_words])
    lightning_word_set = set([w['word'].lower() for w in lightning_words])
    common_words = stock_word_set.intersection(lightning_word_set)
    
    print(f"\nCommon words: {len(common_words)} out of {len(stock_word_set)} stock / {len(lightning_word_set)} lightning")
    print(f"Sample common words: {list(common_words)[:10]}")

# Save detailed comparison
comparison_data = {
    "stock_words": stock_words,
    "lightning_words": lightning_words,
    "summary": {
        "stock_count": len(stock_words),
        "lightning_count": len(lightning_words),
        "stock_time_range": f"{stock_words[0]['start']:.3f}-{stock_words[-1]['end']:.3f}" if stock_words else "N/A",
        "lightning_time_range": f"{lightning_words[0]['start']:.3f}-{lightning_words[-1]['end']:.3f}" if lightning_words else "N/A"
    }
}

with open("word_timestamp_detailed.json", "w") as f:
    json.dump(comparison_data, f, indent=2)

print("\nDetailed data saved to word_timestamp_detailed.json")