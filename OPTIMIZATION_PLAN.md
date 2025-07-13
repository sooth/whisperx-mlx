# WhisperX-MLX Optimization Plan

## Executive Summary

Current benchmark results show Lightning-Whisper-MLX achieving 65.50x realtime performance while our WhisperX-MLX implementation only achieves 8.27x realtime - a 7.92x performance gap. This comprehensive plan outlines specific optimizations to close this gap.

## Current Performance Baseline

- **WhisperX-MLX**: 217.64s (8.27x realtime) for 30-minute audio
- **Lightning-Whisper-MLX**: 27.48s (65.50x realtime) for 30-minute audio
- **Performance Gap**: 7.92x

## Key Issues Identified

1. **No True Batch Processing**: Despite having "batch" in the name, `mlx_batch_optimized.py` processes segments individually
2. **Inefficient Audio Processing**: Each segment processed separately through `mlx_whisper.transcribe()`
3. **Missing Mel Spectrogram Batching**: Mel spectrograms computed individually
4. **Model Loading Overhead**: Model loaded multiple times in some code paths
5. **PyTorch Dependencies**: Audio processing uses PyTorch instead of MLX-native operations

## Optimization Strategy

### Phase 1: True Batch Processing Implementation (Target: 3-4x speedup)

#### 1.1 Implement Native MLX Batch Decoding
- Modify `mlx_batch_optimized.py` to process multiple segments in parallel
- Use MLX's native batch decode capabilities
- Group segments by length to minimize padding overhead

#### 1.2 Batch Mel Spectrogram Computation
- Replace individual mel spectrogram computation with batched processing
- Implement MLX-native mel spectrogram generation
- Cache computed spectrograms for reuse

#### 1.3 Optimize Segment Grouping
- Implement dynamic batching based on segment lengths
- Use bucket sorting to group similar-length segments
- Minimize padding waste

### Phase 2: MLX-Specific Optimizations (Target: 2x additional speedup)

#### 2.1 Kernel Fusion and Graph Optimization
- Use `mx.compile` for attention and feed-forward layers
- Implement fused kernels for common operations
- Leverage MLX's lazy evaluation for graph optimization

#### 2.2 Memory Management
- Implement unified memory model benefits
- Use MLX's zero-copy operations
- Optimize KV-cache reuse across batches

#### 2.3 Replace PyTorch Dependencies
- Convert audio processing to MLX operations
- Implement MLX-native FFT for spectrograms
- Remove unnecessary CPU-GPU transfers

### Phase 3: Advanced Optimizations (Target: 1.5-2x additional speedup)

#### 3.1 Quantization Implementation
- Add 4-bit and 8-bit quantization support
- Implement mixed-precision inference
- Use quantization-aware model loading

#### 3.2 Advanced Batching Strategies
- Implement dynamic batch sizing based on available memory
- Add streaming batch processing for long audio
- Optimize chunk overlap handling

#### 3.3 Model-Specific Optimizations
- Add distilled model support (distil-whisper)
- Implement model-specific kernels
- Optimize for different model sizes

## Implementation Roadmap

### Week 1: Core Batch Processing
1. Rewrite `_process_batch()` in `mlx_batch_optimized.py` for true parallel processing
2. Implement batched mel spectrogram computation
3. Add performance tracking and profiling

### Week 2: MLX-Native Operations
1. Replace PyTorch audio processing with MLX
2. Implement kernel fusion for key operations
3. Optimize memory management

### Week 3: Quantization and Advanced Features
1. Add quantization support
2. Implement dynamic batching
3. Add distilled model support

### Week 4: Testing and Optimization
1. Comprehensive accuracy testing
2. Performance profiling and bottleneck analysis
3. Fine-tuning and documentation

## Success Metrics

1. **Primary Goal**: Achieve at least 50x realtime performance (close to Lightning's 65.50x)
2. **Accuracy**: Maintain WER within 2% of baseline
3. **Memory Usage**: Reduce peak memory by 30%
4. **Latency**: First token latency under 500ms

## Risk Mitigation

1. **Accuracy Loss**: Test each optimization stage for WER impact
2. **Compatibility**: Maintain API compatibility with existing WhisperX
3. **Hardware Variability**: Test on M1, M2, and M3 chips
4. **Edge Cases**: Extensive testing with various audio formats and lengths

## Immediate Next Steps

1. Create feature branch for optimization work
2. Set up comprehensive benchmarking suite
3. Implement true batch processing in `mlx_batch_optimized.py`
4. Begin replacing PyTorch dependencies with MLX operations

## Expected Outcomes

With full implementation of this plan, we expect:
- 6-8x overall speedup (reaching 50-65x realtime)
- Reduced memory usage
- Better scalability for production use
- Maintained or improved accuracy