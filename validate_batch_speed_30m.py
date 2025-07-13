#!/usr/bin/env python3
"""Validate batch word alignment speed on 30m.wav"""

import time
import whisperx
import torch
import os

# Test parameters
audio_file = "30m.wav"
model_size = "tiny"
device = "mps" if torch.backends.mps.is_available() else "cpu"
compute_type = "float32"

# Check if file exists
if not os.path.exists(audio_file):
    print(f"Error: {audio_file} not found!")
    exit(1)

print(f"Validating Batch Word Alignment Speed on {audio_file}")
print("="*80)

# Load audio once
print("Loading audio...")
audio = whisperx.load_audio(audio_file)
duration = len(audio) / 16000  # 16kHz sample rate
print(f"Audio duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")

# Test configurations
tests = [
    {
        "name": "Lightning (transcription only)",
        "backend": "mlx_lightning",
        "word_timestamps": False,
        "batch_size": 16
    },
    {
        "name": "Lightning (with batch word alignment)",
        "backend": "mlx_lightning", 
        "word_timestamps": True,
        "batch_size": 16
    },
    {
        "name": "Stock WhisperX (transcription + manual alignment)",
        "backend": "mlx",
        "word_timestamps": False,
        "batch_size": 16,
        "manual_align": True
    }
]

results = []

for test in tests:
    print(f"\n{test['name']}")
    print("-" * 60)
    
    start_time = time.time()
    
    # Load model
    print("Loading model...")
    model = whisperx.load_model(
        model_size,
        device,
        compute_type=compute_type,
        backend=test["backend"],
        word_timestamps=test.get("word_timestamps", False)
    )
    
    # Transcribe
    print("Transcribing...")
    result = model.transcribe(
        audio, 
        batch_size=test["batch_size"],
        word_timestamps=test.get("word_timestamps", False)
    )
    
    # Manual alignment for stock WhisperX
    if test.get("manual_align", False):
        print("Performing word alignment...")
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"], 
            device=device
        )
        result = whisperx.align(
            result["segments"], 
            model_a, 
            metadata, 
            audio, 
            device
        )
    
    elapsed_time = time.time() - start_time
    
    # Count words and segments
    word_count = 0
    total_text = []
    for seg in result.get("segments", []):
        word_count += len(seg.get("words", []))
        total_text.append(seg.get("text", "").strip())
    
    # Calculate metrics
    realtime_factor = duration / elapsed_time
    
    # Store results
    test_result = {
        "name": test["name"],
        "time": elapsed_time,
        "segments": len(result.get("segments", [])),
        "words": word_count,
        "realtime_factor": realtime_factor,
        "text_length": len(' '.join(total_text))
    }
    results.append(test_result)
    
    print(f"  Processing time: {elapsed_time:.2f}s")
    print(f"  Realtime factor: {realtime_factor:.2f}x")
    print(f"  Segments: {test_result['segments']}")
    print(f"  Words detected: {test_result['words']}")
    print(f"  Total text length: {test_result['text_length']} chars")

# Summary
print("\n" + "="*80)
print("PERFORMANCE SUMMARY (30-minute audio)")
print("="*80)

print(f"\n{'Method':<50} {'Time':>10} {'Speed':>12} {'Words':>10}")
print("-" * 85)

for r in results:
    print(f"{r['name']:<50} {r['time']:>10.2f}s {r['realtime_factor']:>11.2f}x {r['words']:>10}")

# Compare specific metrics
if len(results) >= 2:
    print("\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    # Lightning vs Stock comparison
    lightning_base = results[0]
    lightning_words = results[1]
    stock = results[2] if len(results) > 2 else None
    
    print(f"\nLightning word alignment overhead:")
    overhead_time = lightning_words['time'] - lightning_base['time']
    overhead_pct = (overhead_time / lightning_base['time']) * 100
    print(f"  Additional time: {overhead_time:.2f}s ({overhead_pct:.1f}%)")
    print(f"  Speed with words: {lightning_words['realtime_factor']:.2f}x realtime")
    
    if stock:
        print(f"\nLightning vs Stock WhisperX:")
        speedup = stock['time'] / lightning_words['time']
        print(f"  Lightning is {speedup:.2f}x faster")
        print(f"  Lightning: {lightning_words['realtime_factor']:.2f}x realtime")
        print(f"  Stock: {stock['realtime_factor']:.2f}x realtime")
        
        # Word count comparison
        if lightning_words['words'] > 0 and stock['words'] > 0:
            word_diff = abs(lightning_words['words'] - stock['words'])
            word_pct = (word_diff / stock['words']) * 100
            print(f"\n  Word detection comparison:")
            print(f"  Lightning: {lightning_words['words']} words")
            print(f"  Stock: {stock['words']} words")
            print(f"  Difference: {word_diff} words ({word_pct:.1f}%)")

print("\n✅ Validation complete!")
print("\nKey findings:")
print(f"- Lightning with word alignment achieves {results[1]['realtime_factor']:.2f}x realtime on 30-minute audio")
print(f"- Successfully detects {results[1]['words']} words with precise timestamps")
if len(results) > 2:
    print(f"- {results[2]['time'] / results[1]['time']:.2f}x faster than stock WhisperX with alignment")