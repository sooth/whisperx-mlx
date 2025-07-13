# Word-Level Timestamps Implementation Plan

## Overview
Integrate word-level timestamp support into our Lightning backend while maintaining performance.

## Current Situation

### WhisperX Approach
- Uses separate wav2vec2 models for forced alignment
- Requires PyTorch dependencies
- Adds significant processing time
- Two-step process: transcribe → align

### MLX Whisper Native Support
- Built-in word timestamp extraction
- Uses cross-attention patterns from decoder
- Single-pass process
- No additional models needed

## Implementation Strategy

### Option 1: Direct MLX Integration (Recommended)
Use MLX Whisper's native `word_timestamps=True` parameter.

**Pros:**
- Native support, no extra dependencies
- Faster than WhisperX's two-step process
- Maintains our MLX-only approach

**Cons:**
- Need to modify our optimized decoding flow
- May impact performance slightly

### Option 2: Post-Processing Alignment
Keep current fast transcription, add alignment as optional step.

**Pros:**
- Maintains current speed for basic transcription
- Alignment only when needed

**Cons:**
- More complex implementation
- Still need alignment model

## Detailed Implementation (Option 1)

### 1. Modify Lightning Backend

```python
class WhisperMLXLightningSimple(WhisperBackend):
    def __init__(self, 
                 model_name: str,
                 compute_type: str = "float16",
                 word_timestamps: bool = False,  # Add parameter
                 **kwargs):
        self.word_timestamps = word_timestamps
        # ... rest of init
    
    def transcribe(self, audio, **kwargs):
        if self.word_timestamps:
            # Use MLX native transcribe with word timestamps
            return self._transcribe_with_words(audio, **kwargs)
        else:
            # Use our optimized segment-only path
            return self._transcribe_segments_only(audio, **kwargs)
```

### 2. Word Timestamp Implementation

```python
def _transcribe_with_words(self, audio, **kwargs):
    # Use mlx_whisper.transcribe directly for word timestamps
    import mlx_whisper
    
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=self.model_name,
        word_timestamps=True,
        temperature=self.temperature,
        fp16=(self.compute_type == "float16"),
        verbose=False,
        **kwargs
    )
    
    # Convert to WhisperX format
    segments = []
    for segment in result["segments"]:
        seg = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "words": []
        }
        
        # Add word-level data if available
        if "words" in segment:
            for word in segment["words"]:
                seg["words"].append({
                    "start": word["start"],
                    "end": word["end"],
                    "word": word["word"],
                    "probability": word.get("probability", 1.0)
                })
        
        segments.append(seg)
    
    return {
        "segments": segments,
        "text": result["text"],
        "language": result.get("language", "en")
    }
```

### 3. Performance Optimization

To minimize performance impact:

1. **Lazy Word Extraction**: Only compute word timestamps for segments that need them
2. **Batch Processing**: Process multiple segments' word timestamps together
3. **Caching**: Cache cross-attention patterns for reuse

### 4. API Compatibility

Ensure output format matches WhisperX:

```python
{
    "segments": [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Hello world",
            "words": [
                {"start": 0.0, "end": 0.5, "word": "Hello", "probability": 0.98},
                {"start": 0.6, "end": 1.2, "word": "world", "probability": 0.97}
            ]
        }
    ]
}
```

## Performance Considerations

### Expected Impact
- Without word timestamps: ~170x realtime (current)
- With word timestamps: ~100-120x realtime (estimated)
- Still faster than WhisperX with alignment: ~10-20x realtime

### Optimization Opportunities
1. **Selective Processing**: Only extract words for segments user requests
2. **Attention Caching**: Reuse attention patterns across segments
3. **Parallel Extraction**: Process word alignments in parallel

## Testing Plan

1. **Accuracy Tests**
   - Compare word boundaries with WhisperX alignment
   - Test on various audio types (fast speech, accents, noise)

2. **Performance Tests**
   - Benchmark with/without word timestamps
   - Compare against WhisperX+alignment pipeline

3. **Integration Tests**
   - Ensure API compatibility
   - Test with existing WhisperX workflows

## Next Steps

1. Create proof-of-concept implementation
2. Benchmark performance impact
3. Compare accuracy with WhisperX alignment
4. Optimize based on findings
5. Full integration with configuration options

## Alternative Approach

If performance impact is too high, implement hybrid approach:
- Fast path: Segment-only transcription (170x)
- Accurate path: With word timestamps (100x)
- User chooses based on needs