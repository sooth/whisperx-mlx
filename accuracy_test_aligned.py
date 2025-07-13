#!/usr/bin/env python3
"""
Compare accuracy of our Lightning backend against gold standard - aligned content
"""
import jiwer
from difflib import SequenceMatcher

print("=== Accuracy Comparison: Aligned Content ===")

# Load files
with open("whisperx-large-v3-gold-standard/30m.txt", "r") as f:
    gold_full = f.read().strip()

with open("lightning_large_v3_output.txt", "r") as f:
    our_full = f.read().strip()

# Find where the actual podcast starts in gold standard
podcast_start = "What is up guys, it's Andy"
gold_start_idx = gold_full.find(podcast_start)
if gold_start_idx == -1:
    print("Error: Could not find podcast start in gold standard")
    exit(1)

# Extract just the podcast portion from gold standard
gold_podcast = gold_full[gold_start_idx:]

# Our transcription starts with the podcast
our_podcast = our_full

# Truncate to same length for fair comparison
min_len = min(len(gold_podcast), len(our_podcast))
gold_podcast = gold_podcast[:min_len]
our_podcast = our_podcast[:min_len]

print(f"Comparing {min_len} characters of aligned content")
print(f"\nGold preview: {gold_podcast[:200]}...")
print(f"\nOurs preview: {our_podcast[:200]}...")

# Calculate metrics
print("\n=== Accuracy Metrics ===")

# Word Error Rate
wer = jiwer.wer(gold_podcast.lower(), our_podcast.lower())
print(f"Word Error Rate (WER): {wer:.2%}")

# Character Error Rate  
cer = jiwer.cer(gold_podcast.lower(), our_podcast.lower())
print(f"Character Error Rate (CER): {cer:.2%}")

# Similarity
similarity = SequenceMatcher(None, gold_podcast.lower(), our_podcast.lower()).ratio()
print(f"Text Similarity: {similarity:.2%}")

# Word count comparison
gold_words = gold_podcast.split()
our_words = our_podcast.split()
print(f"\nWord count - Gold: {len(gold_words)}, Ours: {len(our_words)}")

# Find first major difference
print("\n=== First Major Differences ===")
gold_sentences = gold_podcast.split('.')
our_sentences = our_podcast.split('.')

for i in range(min(5, len(gold_sentences), len(our_sentences))):
    if gold_sentences[i].strip().lower() != our_sentences[i].strip().lower():
        print(f"\nSentence {i+1}:")
        print(f"Gold: {gold_sentences[i].strip()}")
        print(f"Ours: {our_sentences[i].strip()}")

# Summary
print("\n=== Summary ===")
if wer < 0.05:
    print("✓ Excellent accuracy (<5% WER)")
elif wer < 0.10:
    print("✓ Very good accuracy (5-10% WER)")
elif wer < 0.15:
    print("✓ Good accuracy (10-15% WER)")
else:
    print("⚠ Accuracy needs improvement (>15% WER)")

print(f"\nSpeed: 20.64x realtime (from previous run)")
print(f"Model: Large-v3 with Lightning backend")