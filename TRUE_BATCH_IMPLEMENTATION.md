# Technical Implementation: True Batch Processing

## The Core Problem

Lightning achieves 65.5x realtime while we achieve 8.66x. The difference is **true batch processing**.

### Lightning's Approach (Simplified)
```python
# 1. Compute mel once
mel = compute_mel(entire_audio)  # Shape: (total_frames, 80)

# 2. Create batches of segments
batch = []
for i in range(0, total_frames, stride):
    segment = mel[i:i+3000]  # 30 second segments
    batch.append(segment)
    if len(batch) == 12:  # Process 12 at once
        results = decode_batch(batch)  # TRUE PARALLEL PROCESSING
        batch = []
```

### Our Current Approach
```python
# Process each segment individually
for segment in segments:
    mel = compute_mel(segment.audio)  # Redundant computation
    result = decode(mel)  # Sequential processing
```

## Step-by-Step Integration

### Step 1: Fix Mel Spectrogram Computation

```python
def compute_mel_once(audio: np.ndarray) -> mx.array:
    """Compute mel spectrogram for entire audio at once"""
    # Current issue: We compute mel per segment
    # Solution: Compute once, slice later
    
    mel = log_mel_spectrogram(audio, n_mels=80)
    # mel shape: (n_frames, 80)
    
    # Critical: Understand the shape expected by decoder
    # MLX expects: (batch, n_frames, n_mels) for conv1d
    # We need to maintain this throughout
    
    return mel
```

### Step 2: Implement Sliding Window Batching

```python
def create_batched_segments(mel: mx.array, batch_size: int = 12) -> List[mx.array]:
    """Create batched segments with proper padding"""
    segments = []
    
    # Sliding window with 30-second segments
    for i in range(0, mel.shape[0], N_FRAMES):
        segment = mel[i:i+N_FRAMES]
        
        # Pad last segment if needed
        if segment.shape[0] < N_FRAMES:
            segment = pad_or_trim(segment, N_FRAMES, axis=0)
        
        segments.append(segment)
    
    # Group into batches
    batches = []
    for i in range(0, len(segments), batch_size):
        batch = segments[i:i+batch_size]
        # Stack along batch dimension
        batch_tensor = mx.stack(batch, axis=0)
        # Shape: (batch_size, N_FRAMES, 80)
        batches.append(batch_tensor)
    
    return batches
```

### Step 3: Fix Decoder Integration

```python
def decode_batch_properly(model, batch: mx.array) -> List[DecodingResult]:
    """Decode a batch with proper tensor handling"""
    
    # Current issue: Dimension mismatch
    # Solution: Ensure correct shape transformations
    
    # The decoder expects different format internally
    # We need to match Lightning's exact approach
    
    options = DecodingOptions(
        temperature=0.0,  # Greedy decoding
        without_timestamps=False,
        fp16=True
    )
    
    # Critical: The decode function needs proper format
    # Lightning somehow makes this work - we need to study how
    results = decode(model, batch, options)
    
    return results
```

### Step 4: Merge Results Efficiently

```python
def merge_batch_results(batch_results: List[List[DecodingResult]], 
                       segment_times: List[Tuple[float, float]]) -> Dict:
    """Merge batch results into final transcription"""
    all_text = []
    all_segments = []
    
    for batch_idx, batch in enumerate(batch_results):
        for seg_idx, result in enumerate(batch):
            # Calculate actual timestamp
            start_time = segment_times[batch_idx * len(batch) + seg_idx][0]
            
            # Add to segments with proper timestamps
            for segment in result.segments:
                segment['start'] += start_time
                segment['end'] += start_time
                all_segments.append(segment)
            
            all_text.append(result.text)
    
    return {
        'text': ' '.join(all_text),
        'segments': all_segments
    }
```

## Critical Technical Issues to Solve

### 1. Tensor Dimension Mismatch

**Problem**: When we try batch decode, we get:
```
ValueError: [conv] Expect the input channels in the input and weight array to match but got shapes - input: (4,80,3000) and weight: (384,3,80)
```

**Investigation Needed**:
```python
# Test different tensor layouts
test_shapes = [
    (batch, n_mels, n_frames),    # (4, 80, 3000)
    (batch, n_frames, n_mels),    # (4, 3000, 80)
    (n_mels, batch, n_frames),    # (80, 4, 3000)
]

# Find which one actually works with MLX decoder
```

### 2. Memory Layout Optimization

**Problem**: Unnecessary copies between operations

**Solution**:
```python
# Use MLX's lazy evaluation
with mx.stream():
    mel = compute_mel(audio)
    batches = create_batches(mel)
    results = decode_batches(batches)
# All operations execute efficiently
```

### 3. VAD Integration Without Breaking Batching

**Problem**: VAD creates irregular segments

**Solution**:
```python
def batch_vad_segments(vad_segments, target_batch_duration=30.0):
    """Group VAD segments into batch-friendly chunks"""
    batched_segments = []
    current_batch = []
    current_duration = 0
    
    for segment in vad_segments:
        duration = segment['end'] - segment['start']
        
        if current_duration + duration > target_batch_duration:
            # Process current batch
            batched_segments.append(merge_segments(current_batch))
            current_batch = [segment]
            current_duration = duration
        else:
            current_batch.append(segment)
            current_duration += duration
    
    return batched_segments
```

## Validation Tests

### Test 1: Tensor Shape Compatibility
```python
def test_tensor_shapes():
    # Create minimal test case
    audio = np.random.randn(16000 * 5)  # 5 seconds
    mel = log_mel_spectrogram(audio)
    
    # Test single decode
    single = decode(model, mel[None], options)
    
    # Test batch decode with different shapes
    batch2 = mx.stack([mel, mel])
    batch_results = decode(model, batch2, options)
    
    assert len(batch_results) == 2
```

### Test 2: Performance Validation
```python
def benchmark_batch_vs_sequential():
    audio = load_audio("30m.wav")
    
    # Sequential
    start = time.time()
    for i in range(0, len(audio), 30*16000):
        segment = audio[i:i+30*16000]
        result = transcribe(segment)
    seq_time = time.time() - start
    
    # Batch
    start = time.time()
    result = transcribe_batched(audio)
    batch_time = time.time() - start
    
    print(f"Speedup: {seq_time / batch_time}x")
```

## Integration Path

1. **Create New Backend**: `mlx_true_batch.py` that implements this approach
2. **Test in Isolation**: Verify 50x+ performance before integration
3. **Add Compatibility Layer**: Handle VAD segments properly
4. **Progressive Rollout**: Add `--experimental-batch` flag first
5. **Make Default**: Once stable, make it the default backend

The key is to exactly replicate Lightning's tensor handling while maintaining WhisperX's API compatibility.