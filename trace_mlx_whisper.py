#!/usr/bin/env python3
"""
Step 2: Trace through mlx_whisper to understand exact processing
"""
import numpy as np
import mlx.core as mx
from mlx_whisper.load_models import load_model
from mlx_whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES
from mlx_whisper.decoding import decode, DecodingOptions, DecodingTask
from mlx_whisper import transcribe

# Let's look at the actual transcribe.py to understand the flow
import inspect
import mlx_whisper.transcribe

print("=== Tracing MLX Whisper Flow ===")

# Check the transcribe function
print("\n1. Examining transcribe function...")
source_lines = inspect.getsourcelines(mlx_whisper.transcribe.transcribe)[0]

# Find key processing steps
for i, line in enumerate(source_lines):
    if "mel" in line and "=" in line:
        print(f"Line {i}: {line.strip()}")

# Now let's look at the decode_with_fallback function
print("\n2. Examining decode_with_fallback...")
decode_func = mlx_whisper.transcribe.decode_with_fallback
source_lines = inspect.getsourcelines(decode_func)[0]

for i, line in enumerate(source_lines[:20]):  # First 20 lines
    if "decode" in line.lower():
        print(f"Line {i}: {line.strip()}")

# The key is in the transcribe.py file - let's understand the mel processing
print("\n3. Understanding mel processing in transcribe.py...")

# Create test audio
audio = np.random.randn(16000 * 5).astype(np.float32) * 0.1

# Load model 
model = load_model("mlx-community/whisper-tiny", dtype=mx.float16)

# What transcribe.py does:
print("\n4. Reproducing transcribe.py flow...")

# Step 1: Compute mel
mel = log_mel_spectrogram(audio, n_mels=model.dims.n_mels)
print(f"Step 1 - mel shape: {mel.shape}")

# Step 2: Pad to 30 seconds of mel frames
mel = pad_or_trim(mel, N_FRAMES)
print(f"Step 2 - padded mel shape: {mel.shape}")

# Step 3: Add batch dimension for single
mel_single = mel[None]
print(f"Step 3 - single mel shape: {mel_single.shape}")

# The issue is when we try to batch!
print("\n5. The batch problem...")

# Lightning must handle batching differently
# Let's check their exact approach

# They create segments like this:
segments = []
seek = 0
while seek < mel.shape[0]:
    segment = mel[seek:seek + N_FRAMES]
    if segment.shape[0] < N_FRAMES:
        segment = pad_or_trim(segment, N_FRAMES)
    segments.append(segment)
    seek += N_FRAMES
    if len(segments) >= 2:  # Just test with 2
        break

print(f"Created {len(segments)} segments")

# Now the critical part - how to batch them?
print("\n6. Testing batching approaches...")

# Approach 1: Stack along axis 0 (what we tried)
try:
    batch1 = mx.stack(segments, axis=0)
    print(f"Approach 1 - Stack axis=0: {batch1.shape}")
    # This gives (2, 3000, 80)
    # But decode expects something else for batch!
except Exception as e:
    print(f"Approach 1 failed: {e}")

# Let's check what decode actually does with batches
print("\n7. Understanding decode's batch handling...")

# Look at the DecodingTask
task_source = inspect.getsourcelines(DecodingTask)[0]
for i, line in enumerate(task_source[:50]):
    if "batch" in line.lower() or "mel.shape" in line:
        print(f"Line {i}: {line.strip()}")

# The key insight: decode expects mel in a specific format
# Let's test the exact format that works

print("\n8. Finding the working format...")

# We know single works with shape (3000, 80)
# The batch dimension must be handled differently

# Test with DecodingTask directly
options = DecodingOptions(temperature=0.0, fp16=True)

# Single segment
single_mel = mx.array(segments[0], dtype=mx.float16)
print(f"Single segment shape: {single_mel.shape}")

try:
    # Add batch dim
    single_batch = single_mel[None]  # (1, 3000, 80)
    print(f"Single with batch dim: {single_batch.shape}")
    
    # Try decode
    task = DecodingTask(model, options)
    audio_features = task._get_audio_features(single_batch)
    print(f"✓ Audio features shape: {audio_features.shape}")
    
except Exception as e:
    print(f"✗ Failed: {str(e)[:100]}")

print("\n9. Key finding:")
print("The issue is that batch decode has a bug in MLX whisper")
print("We need to either fix it or work around it")