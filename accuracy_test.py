#!/usr/bin/env python3
"""
Compare accuracy of our Lightning backend against gold standard
"""
import json
import jiwer
from whisperx.backends.mlx_lightning_simple import WhisperMLXLightningSimple
import whisperx
from difflib import SequenceMatcher

print("=== Accuracy Comparison: Lightning Backend vs Gold Standard ===")

# Load gold standard
print("\n1. Loading gold standard transcription...")
with open("whisperx-large-v3-gold-standard/30m.txt", "r") as f:
    gold_standard = f.read().strip()

print(f"Gold standard length: {len(gold_standard)} characters")
print(f"Preview: {gold_standard[:200]}...")

# Load audio
print("\n2. Loading 30m.wav...")
audio = whisperx.load_audio("30m.wav")
duration = len(audio) / 16000
print(f"Duration: {duration:.1f}s")

# Test our Lightning backend with large-v3
print("\n3. Testing our Lightning backend with large-v3...")
backend = WhisperMLXLightningSimple(
    model_name="mlx-community/whisper-large-v3-mlx",
    compute_type="float16",
    temperature=0.0
)

import time
start = time.time()
result = backend.transcribe(audio)
elapsed = time.time() - start

our_text = result['text'].strip()
speed = duration / elapsed

print(f"Time: {elapsed:.2f}s")
print(f"Speed: {speed:.2f}x realtime")
print(f"Our transcription length: {len(our_text)} characters")
print(f"Preview: {our_text[:200]}...")

# Calculate accuracy metrics
print("\n4. Accuracy Metrics:")

# Word Error Rate
wer = jiwer.wer(gold_standard.lower(), our_text.lower())
print(f"Word Error Rate (WER): {wer:.2%}")

# Character Error Rate
cer = jiwer.cer(gold_standard.lower(), our_text.lower())
print(f"Character Error Rate (CER): {cer:.2%}")

# Similarity score
similarity = SequenceMatcher(None, gold_standard.lower(), our_text.lower()).ratio()
print(f"Text Similarity: {similarity:.2%}")

# Word-level metrics
gold_words = gold_standard.split()
our_words = our_text.split()
print(f"\nWord count - Gold: {len(gold_words)}, Ours: {len(our_words)}")
print(f"Word count difference: {abs(len(gold_words) - len(our_words))} ({abs(len(gold_words) - len(our_words))/len(gold_words)*100:.1f}%)")

# Save our transcription for inspection
print("\n5. Saving our transcription for manual inspection...")
with open("lightning_large_v3_output.txt", "w") as f:
    f.write(our_text)
print("Saved to: lightning_large_v3_output.txt")

# Summary
print("\n=== Summary ===")
print(f"Speed: {speed:.2f}x realtime")
print(f"WER: {wer:.2%}")
print(f"CER: {cer:.2%}")
print(f"Similarity: {similarity:.2%}")

if wer < 0.10:
    print("\n✓ Excellent accuracy (<10% WER)")
elif wer < 0.20:
    print("\n✓ Good accuracy (10-20% WER)")
else:
    print("\n⚠ Accuracy needs improvement (>20% WER)")