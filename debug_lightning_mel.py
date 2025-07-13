#!/usr/bin/env python3
"""
Debug Lightning's exact mel processing to understand tensor shapes
"""
import numpy as np
import mlx.core as mx
from mlx_whisper.load_models import load_model
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
from mlx_whisper.decoding import decode, DecodingOptions
import whisperx

print("=== Debugging Lightning Mel Processing ===")

# Load test audio
audio = whisperx.load_audio("short.wav")
print(f"Audio shape: {audio.shape} ({len(audio)/16000:.1f}s)")

# Load model
model = load_model("mlx-community/whisper-tiny", dtype=mx.float16)

# Step 1: Understand mel computation
print("\n1. Mel Spectrogram Computation:")
mel = log_mel_spectrogram(audio, n_mels=80)
print(f"   Raw mel shape: {mel.shape}")
print(f"   Mel dtype: {mel.dtype}")

# Step 2: How Lightning creates segments
print("\n2. Lightning's Segment Creation:")
segments = []
for i in range(0, mel.shape[0], N_FRAMES):
    segment = mel[i:i + N_FRAMES]
    if segment.shape[0] < N_FRAMES:
        segment = pad_or_trim(segment, N_FRAMES, axis=0)
    segments.append(segment)
    print(f"   Segment {len(segments)}: shape {segment.shape}")

# Step 3: Test single segment decode
print("\n3. Single Segment Decode:")
single_segment = mx.array(segments[0], dtype=mx.float16)
print(f"   Single segment shape: {single_segment.shape}")

# The key: MLX decode expects mel in specific format
# Let's test what works
options = DecodingOptions(temperature=0.0, fp16=True)

try:
    # Test with batch dimension
    single_with_batch = single_segment[None]  # Add batch dim
    print(f"   With batch dim: {single_with_batch.shape}")
    
    result = decode(model, single_with_batch, options)
    print(f"   ✓ Single decode works! Text: '{result[0].text[:50]}'")
except Exception as e:
    print(f"   ✗ Single decode failed: {e}")

# Step 4: Test batch decode
print("\n4. Batch Decode Test:")
if len(segments) >= 2:
    try:
        # Stack segments
        batch = mx.stack([mx.array(s, dtype=mx.float16) for s in segments[:2]], axis=0)
        print(f"   Batch shape: {batch.shape}")
        
        results = decode(model, batch, options)
        print(f"   ✓ Batch decode works! Got {len(results)} results")
        for i, r in enumerate(results):
            print(f"      Result {i+1}: '{r.text[:30]}'")
    except Exception as e:
        print(f"   ✗ Batch decode failed: {str(e)[:200]}")
        
        # Let's debug the exact failure
        print("\n   Debugging the failure:")
        
        # Check model encoder
        try:
            batch = mx.stack([mx.array(s, dtype=mx.float16) for s in segments[:2]], axis=0)
            encoder_out = model.encoder(batch)
            print(f"   ✓ Encoder accepts batch! Output shape: {encoder_out.shape}")
        except Exception as e2:
            print(f"   ✗ Encoder failed: {e2}")

# Step 5: Understand the exact issue
print("\n5. Understanding the Issue:")
print("   The problem is in the decode function's batch handling")
print("   Let's check mlx_whisper's source to understand...")

# Import and check
import mlx_whisper
import inspect

# Get decode source
decode_source = inspect.getsource(mlx_whisper.decoding.decode)
if "for mel_segment in mel" in decode_source:
    print("   ✓ Found issue: decode loops over mel segments")
    print("   This means batch dim is treated as iteration, not parallel!")
else:
    print("   Need to investigate decode implementation further")

print("\n6. Solution:")
print("   We need to either:")
print("   a) Fix mlx_whisper's decode to handle batches properly")
print("   b) Implement our own batch decode logic")
print("   c) Use parallel processing with single decodes")