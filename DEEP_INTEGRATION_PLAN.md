# Deep Integration Plan: Lightning Optimizations → WhisperX-MLX

## Current Gap Analysis

### What Lightning Does (65.5x realtime)
1. **One-time Mel Computation**: Computes mel spectrogram ONCE for entire audio
2. **True Batch Processing**: Processes 12 segments in parallel through decoder
3. **Efficient Memory Layout**: Maintains consistent tensor shapes throughout
4. **Optimized Seek Strategy**: Pre-computed sliding windows with -3000 frame offset
5. **Greedy Decoding**: Default temperature=0.0 (no beam search)
6. **Model Singleton**: Caches model across all instances
7. **Simplified Pipeline**: Minimal post-processing overhead

### What We're Missing
1. ❌ **True Batch Decoding**: Our attempts failed due to tensor dimension mismatches
2. ❌ **Single Mel Computation**: We compute mel per segment
3. ❌ **Proper Tensor Layout**: Confusion between (frames, mels) vs (mels, frames)
4. ❌ **Batch Seek Strategy**: We process segments individually
5. ✓ **Greedy Decoding**: We implemented this
6. ✓ **Model Caching**: We implemented this
7. ❌ **Direct Integration**: We wrapped existing APIs instead of deep integration

## Deep Integration Strategy

### Phase 1: Understand Lightning's Exact Flow (2 days)

1. **Extract Lightning's Core Loop**
   ```python
   # Study their exact implementation:
   - How they compute mel: shape, padding, normalization
   - How they prepare batches: tensor stacking, padding
   - How they call decode: exact parameters and shapes
   - How they handle failures: fallback mechanism
   ```

2. **Create Standalone Test**
   ```python
   # Reproduce Lightning's exact flow outside WhisperX
   - Test with same audio
   - Verify same performance
   - Document every tensor shape
   ```

### Phase 2: Bridge the Architecture Gap (3 days)

1. **Create New Pipeline Mode**
   ```python
   class LightningPipeline:
       def process_audio(self, audio):
           # Skip VAD completely
           # Compute mel once
           # Batch process everything
           # Return raw segments
   ```

2. **Modify WhisperX Flow**
   ```python
   # Add --lightning mode that:
   - Bypasses VAD when not needed
   - Uses batched processing
   - Applies alignment post-hoc
   ```

### Phase 3: Deep Backend Integration (4 days)

1. **Rewrite Batch Processor**
   ```python
   class TrueLightningBackend(WhisperBackend):
       def transcribe(self, audio):
           # Step 1: Compute mel spectrogram ONCE
           mel = self._compute_mel_entire_audio(audio)
           
           # Step 2: Create sliding windows
           windows = self._create_sliding_windows(mel)
           
           # Step 3: Batch process ALL windows
           results = self._batch_decode_all(windows)
           
           # Step 4: Merge results
           return self._merge_results(results)
   ```

2. **Fix Tensor Dimension Issues**
   ```python
   # Create compatibility layer:
   - Study MLX conv1d expectations
   - Create proper tensor permutations
   - Test with minimal examples
   ```

### Phase 4: Optimize VAD Integration (2 days)

1. **Parallel VAD + ASR**
   ```python
   # Run VAD and ASR in parallel:
   - Thread 1: VAD detection
   - Thread 2: Full ASR transcription
   - Merge results at the end
   ```

2. **Smart Batching with VAD**
   ```python
   # Group VAD segments into Lightning-style batches:
   - Concatenate small segments
   - Pad to optimal batch sizes
   - Process as unified batches
   ```

### Phase 5: Advanced Optimizations (3 days)

1. **Implement Speculative Decoding**
   ```python
   # Use small model to guide large model:
   - Run tiny model first
   - Use outputs to constrain large model
   - Potential 2x additional speedup
   ```

2. **Dynamic Batching**
   ```python
   # Adjust batch size based on:
   - Available memory
   - Audio length
   - Model size
   ```

3. **Quantization Integration**
   ```python
   # Properly integrate 4-bit models:
   - Test mlx-community quantized models
   - Benchmark speed/accuracy tradeoff
   - Auto-select based on hardware
   ```

## Implementation Checklist

### Week 1: Foundation
- [ ] Extract Lightning's exact tensor processing flow
- [ ] Create minimal reproduction of their performance
- [ ] Document all shape transformations
- [ ] Fix our tensor dimension issues

### Week 2: Integration  
- [ ] Implement true batch processing in new backend
- [ ] Add --lightning mode to bypass VAD
- [ ] Integrate single mel computation
- [ ] Test on various audio lengths

### Week 3: Optimization
- [ ] Add parallel VAD+ASR processing
- [ ] Implement dynamic batching
- [ ] Add quantization support
- [ ] Performance profiling and tuning

## Success Metrics

1. **Primary Goal**: Achieve 50x+ realtime on 30m.wav (currently 8.66x)
2. **Accuracy**: Maintain exact same WER as current implementation
3. **Compatibility**: Preserve WhisperX API and features
4. **Memory**: Reduce peak memory usage by 50%

## Technical Deep Dives Needed

1. **MLX Conv1d Input Format**
   - Why does it expect (batch, frames, features) not (batch, features, frames)?
   - How to properly prepare mel spectrograms?

2. **Batch Decode Internals**
   - What exactly does decode() do with batched inputs?
   - How does it handle attention masks for different lengths?

3. **Memory Layout Optimization**
   - How to minimize copies between CPU/GPU?
   - Optimal chunk sizes for different hardware?

## Risk Mitigation

1. **Compatibility Breaking**: Create new backend, don't modify existing
2. **Accuracy Loss**: Test WER at each step
3. **Memory Issues**: Profile and add adaptive batching
4. **API Changes**: Add new modes rather than changing defaults

## Next Immediate Steps

1. **Today**: Create test script that reproduces Lightning's exact mel processing
2. **Tomorrow**: Debug why our batch decode fails when Lightning's works
3. **Day 3**: Implement working prototype of true batch processing
4. **Day 4**: Integrate into WhisperX with --lightning flag
5. **Day 5**: Benchmark and iterate

This plan represents a complete architectural overhaul rather than incremental optimization. The key insight is that we need to fundamentally change how WhisperX processes audio to match Lightning's approach, not just tweak the existing pipeline.