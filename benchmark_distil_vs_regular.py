#!/usr/bin/env python3
"""Comprehensive benchmark comparing distil-whisper vs regular whisper models"""

import time
import json
import whisperx
import torch
import numpy as np
from tabulate import tabulate

def benchmark_model(model_name, audio_files, backend="lightning", with_word_timestamps=False):
    """Benchmark a single model on multiple audio files"""
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    compute_type = "float16"
    
    results = []
    
    # Load model once
    print(f"\nLoading {model_name}...")
    load_start = time.time()
    model = whisperx.load_model(
        model_name, 
        device=device, 
        compute_type=compute_type,
        backend=backend
    )
    load_time = time.time() - load_start
    print(f"✓ Model loaded in {load_time:.2f}s")
    
    for audio_file in audio_files:
        print(f"\n  Testing on {audio_file}...")
        
        # Load audio
        audio = whisperx.load_audio(audio_file)
        duration = len(audio) / 16000
        
        # Transcribe without word timestamps
        start_time = time.time()
        result = model.transcribe(audio, batch_size=16)
        transcribe_time = time.time() - start_time
        
        # Transcribe with word timestamps if requested
        word_time = None
        if with_word_timestamps:
            start_time = time.time()
            result_with_words = model.transcribe(audio, batch_size=16, align_words=True)
            word_time = time.time() - start_time
        
        # Calculate metrics
        realtime_factor = duration / transcribe_time
        word_realtime_factor = duration / word_time if word_time else None
        
        # Extract text
        text = result.get('text', '')
        if not text and 'segments' in result:
            text = ' '.join([seg.get('text', '') for seg in result.get('segments', [])])
        
        # Count words
        num_words = len(text.split())
        num_segments = len(result.get('segments', []))
        
        results.append({
            'audio_file': audio_file,
            'duration': duration,
            'transcribe_time': transcribe_time,
            'realtime_factor': realtime_factor,
            'word_time': word_time,
            'word_realtime_factor': word_realtime_factor,
            'num_words': num_words,
            'num_segments': num_segments,
            'text_preview': text[:50] + '...' if len(text) > 50 else text
        })
        
        print(f"    Duration: {duration:.2f}s")
        print(f"    Transcribe time: {transcribe_time:.2f}s ({realtime_factor:.2f}x realtime)")
        if word_time:
            print(f"    With words time: {word_time:.2f}s ({word_realtime_factor:.2f}x realtime)")
        print(f"    Words: {num_words}, Segments: {num_segments}")
    
    return {
        'model': model_name,
        'load_time': load_time,
        'results': results
    }

def main():
    """Run comprehensive benchmark"""
    
    # Models to test
    models = [
        ("large-v3", "Regular Whisper Large v3"),
        ("distil-large-v3", "Distil-Whisper Large v3"),
        ("distil-whisper-large-v3", "Distil-Whisper Large v3 (alt name)"),
    ]
    
    # Audio files to test
    audio_files = ["short.wav"]  # Add "30m.wav" for longer test
    
    print("Distil-Whisper vs Regular Whisper Benchmark")
    print("=" * 80)
    
    all_results = []
    
    # Test each model
    for model_name, description in models:
        print(f"\n{'='*80}")
        print(f"Testing {description}")
        print(f"{'='*80}")
        
        try:
            # Test without word timestamps
            result = benchmark_model(model_name, audio_files, with_word_timestamps=False)
            result['description'] = description
            all_results.append(result)
            
            # Also test with word timestamps for key models
            if model_name in ["large-v3", "distil-large-v3"]:
                print(f"\n  Testing with word timestamps...")
                result_words = benchmark_model(model_name, audio_files, with_word_timestamps=True)
                result['word_results'] = result_words['results']
                
        except Exception as e:
            print(f"✗ Error testing {model_name}: {e}")
            all_results.append({
                'model': model_name,
                'description': description,
                'error': str(e)
            })
    
    # Generate summary report
    print(f"\n{'='*80}")
    print("SUMMARY REPORT")
    print(f"{'='*80}")
    
    # Model comparison table
    table_data = []
    regular_speed = None
    
    for result in all_results:
        if 'error' in result:
            continue
            
        avg_realtime = np.mean([r['realtime_factor'] for r in result['results']])
        
        if result['model'] == 'large-v3':
            regular_speed = avg_realtime
            
        speedup = avg_realtime / regular_speed if regular_speed else 1.0
        
        table_data.append([
            result['description'],
            f"{result['load_time']:.2f}s",
            f"{avg_realtime:.2f}x",
            f"{speedup:.2f}x"
        ])
    
    print("\nModel Performance Comparison:")
    print(tabulate(table_data, 
                   headers=['Model', 'Load Time', 'Avg Speed', 'Speedup vs Regular'],
                   tablefmt='grid'))
    
    # Word alignment comparison
    word_data = []
    for result in all_results:
        if 'word_results' in result:
            for i, r in enumerate(result['results']):
                wr = result['word_results'][i]
                if wr['word_time']:
                    overhead = (wr['word_time'] - r['transcribe_time']) / r['transcribe_time'] * 100
                    word_data.append([
                        result['description'],
                        r['audio_file'],
                        f"{r['realtime_factor']:.2f}x",
                        f"{wr['word_realtime_factor']:.2f}x",
                        f"{overhead:.1f}%"
                    ])
    
    if word_data:
        print("\nWord Alignment Performance:")
        print(tabulate(word_data,
                       headers=['Model', 'Audio', 'Base Speed', 'With Words', 'Overhead'],
                       tablefmt='grid'))
    
    # Save detailed results
    with open('distil_benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to distil_benchmark_results.json")
    
    # Key findings
    print("\nKey Findings:")
    print("-" * 40)
    
    distil_result = next((r for r in all_results if r['model'] == 'distil-large-v3'), None)
    regular_result = next((r for r in all_results if r['model'] == 'large-v3'), None)
    
    if distil_result and regular_result and 'error' not in distil_result and 'error' not in regular_result:
        distil_speed = np.mean([r['realtime_factor'] for r in distil_result['results']])
        regular_speed = np.mean([r['realtime_factor'] for r in regular_result['results']])
        speedup = distil_speed / regular_speed
        
        print(f"✓ Distil-Whisper is {speedup:.2f}x faster than regular Whisper")
        print(f"✓ Distil-Whisper achieves {distil_speed:.2f}x realtime speed")
        print(f"✓ Model loading time similar for both ({distil_result['load_time']:.2f}s vs {regular_result['load_time']:.2f}s)")
        
        if 'word_results' in distil_result:
            print(f"✓ Word alignment works with distil models")

if __name__ == "__main__":
    main()