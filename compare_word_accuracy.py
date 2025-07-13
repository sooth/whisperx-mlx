#!/usr/bin/env python3
"""
Compare word timestamp accuracy between our implementation and stock WhisperX
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import json
import numpy as np
from difflib import SequenceMatcher
import whisperx

print("=== Word Timestamp Accuracy Comparison ===")

# Load test audio
audio_file = "short.wav"
audio = whisperx.load_audio(audio_file)
duration = len(audio) / 16000
print(f"\nTest audio: {audio_file} ({duration:.1f}s)")

# Test 1: Stock WhisperX with alignment
print("\n1. Testing stock WhisperX with wav2vec2 alignment...")
try:
    # Load standard WhisperX model
    model_stock = whisperx.load_model(
        "mlx-community/whisper-tiny",
        device="cpu",
        compute_type="float16",
        backend="standard"  # Force standard backend
    )
    
    start = time.time()
    result_stock = model_stock.transcribe(audio_file)
    
    # Load alignment model
    model_a, metadata = whisperx.load_align_model(
        language_code=result_stock["language"], 
        device="cpu"
    )
    
    # Align
    result_aligned = whisperx.align(
        result_stock["segments"], 
        model_a, 
        metadata, 
        audio, 
        "cpu",
        return_char_alignments=False
    )
    
    time_stock = time.time() - start
    
    # Count words
    words_stock = []
    for seg in result_aligned["segments"]:
        if "words" in seg:
            words_stock.extend(seg["words"])
    
    print(f"✓ Time: {time_stock:.2f}s ({duration/time_stock:.1f}x realtime)")
    print(f"✓ Words aligned: {len(words_stock)}")
    
except Exception as e:
    print(f"✗ Stock WhisperX failed: {e}")
    result_aligned = None
    words_stock = []

# Test 2: Our Lightning implementation
print("\n2. Testing our Lightning backend with word timestamps...")
model_ours = whisperx.load_model(
    "mlx-community/whisper-tiny",
    device="cpu",
    compute_type="float16",
    backend="lightning",
    word_timestamps=True
)

start = time.time()
# Direct backend call to avoid VAD
result_ours = model_ours.backend.transcribe(audio)
time_ours = time.time() - start

# Count words
words_ours = []
for seg in result_ours["segments"]:
    if "words" in seg:
        words_ours.extend(seg["words"])

print(f"✓ Time: {time_ours:.2f}s ({duration/time_ours:.1f}x realtime)")
print(f"✓ Words extracted: {len(words_ours)}")

# Compare accuracy
print("\n3. Accuracy Comparison:")

if words_stock and words_ours:
    # Compare word texts
    text_stock = " ".join([w.get("word", w.get("text", "")).strip() for w in words_stock])
    text_ours = " ".join([w["word"].strip() for w in words_ours])
    
    similarity = SequenceMatcher(None, text_stock.lower(), text_ours.lower()).ratio()
    print(f"Text similarity: {similarity:.2%}")
    
    # Compare timing accuracy (sample first 10 words)
    print("\nTiming comparison (first 10 words):")
    print(f"{'Word':<20} {'Stock Start':<12} {'Our Start':<12} {'Diff (s)':<10}")
    print("-" * 60)
    
    min_words = min(len(words_stock), len(words_ours), 10)
    timing_diffs = []
    
    for i in range(min_words):
        w_stock = words_stock[i]
        w_ours = words_ours[i]
        
        stock_word = w_stock.get("word", w_stock.get("text", "")).strip()
        our_word = w_ours["word"].strip()
        stock_start = w_stock.get("start", 0)
        our_start = w_ours["start"]
        
        diff = abs(stock_start - our_start)
        timing_diffs.append(diff)
        
        print(f"{stock_word:<20} {stock_start:>12.2f} {our_start:>12.2f} {diff:>10.2f}")
    
    avg_timing_diff = np.mean(timing_diffs) if timing_diffs else 0
    print(f"\nAverage timing difference: {avg_timing_diff:.3f}s")
    
    # Check if we meet accuracy requirements
    if avg_timing_diff < 0.1:  # Within 100ms
        print("✓ Timing accuracy: EXCELLENT (< 0.1s average difference)")
    elif avg_timing_diff < 0.2:
        print("✓ Timing accuracy: GOOD (< 0.2s average difference)")
    else:
        print("✗ Timing accuracy: NEEDS IMPROVEMENT (> 0.2s average difference)")

# Performance comparison
print("\n4. Performance Comparison:")
if result_aligned:
    print(f"Stock WhisperX: {duration/time_stock:.1f}x realtime")
print(f"Our Lightning: {duration/time_ours:.1f}x realtime")
if result_aligned:
    print(f"Speed improvement: {time_stock/time_ours:.1f}x faster")

# Save detailed comparison
print("\n5. Saving detailed results...")
comparison = {
    "audio_duration": duration,
    "stock_whisperx": {
        "time": time_stock if result_aligned else None,
        "speed": duration/time_stock if result_aligned else None,
        "word_count": len(words_stock),
        "sample_words": [
            {
                "word": w.get("word", w.get("text", "")),
                "start": w.get("start", 0),
                "end": w.get("end", 0)
            }
            for w in words_stock[:5]
        ] if words_stock else []
    },
    "our_lightning": {
        "time": time_ours,
        "speed": duration/time_ours,
        "word_count": len(words_ours),
        "sample_words": [
            {
                "word": w["word"],
                "start": float(w["start"]),
                "end": float(w["end"]),
                "probability": float(w.get("probability", 1.0))
            }
            for w in words_ours[:5]
        ]
    },
    "comparison": {
        "text_similarity": similarity if words_stock and words_ours else None,
        "avg_timing_difference": avg_timing_diff if words_stock and words_ours else None,
        "speed_improvement": time_stock/time_ours if result_aligned else None
    }
}

with open("word_accuracy_comparison.json", "w") as f:
    json.dump(comparison, f, indent=2)

print("✓ Results saved to word_accuracy_comparison.json")

# Test with larger audio
print("\n6. Testing with larger audio sample...")
# Use first 2 minutes of 30m.wav
audio_long = whisperx.load_audio("30m.wav")[:120 * 16000]
duration_long = len(audio_long) / 16000

print(f"Testing {duration_long:.1f}s audio...")

start = time.time()
result_long = model_ours.backend.transcribe(audio_long)
time_long = time.time() - start

words_long = sum(len(seg.get("words", [])) for seg in result_long["segments"])
print(f"✓ Time: {time_long:.2f}s ({duration_long/time_long:.1f}x realtime)")
print(f"✓ Words extracted: {words_long}")

print("\n=== Summary ===")
if words_stock and words_ours:
    if similarity > 0.95 and avg_timing_diff < 0.2:
        print("✓ Accuracy matches stock WhisperX!")
    else:
        print("⚠ Accuracy needs improvement")
        print(f"  Text similarity: {similarity:.2%} (target: >95%)")
        print(f"  Timing difference: {avg_timing_diff:.3f}s (target: <0.2s)")
else:
    print("⚠ Could not compare with stock WhisperX")

print(f"\nPerformance: {duration/time_ours:.1f}x realtime")
if duration/time_ours < 50:
    print("⚠ Performance below target (50x+ realtime)")
else:
    print("✓ Performance meets target!")