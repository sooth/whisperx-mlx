#!/usr/bin/env python3
"""
Comprehensive VAD Benchmark Suite for WhisperX-MLX
Tests speed, accuracy, and memory usage of different VAD implementations
"""

import time
import json
import psutil
import os
import numpy as np
from typing import Dict, List, Tuple
import torch
import whisperx
from tabulate import tabulate
from whisperx.vads import Pyannote, Silero
from whisperx.audio import load_audio, SAMPLE_RATE

class VADBenchmark:
    """Benchmark suite for VAD implementations"""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process(os.getpid())
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_vad_speed(self, vad_model, audio: np.ndarray, name: str) -> Dict:
        """Benchmark VAD processing speed"""
        print(f"\nBenchmarking {name} speed...")
        
        # Prepare audio
        if hasattr(vad_model, 'preprocess_audio'):
            waveform = vad_model.preprocess_audio(audio)
        else:
            waveform = torch.from_numpy(audio).unsqueeze(0)
        
        # Measure memory before
        mem_before = self.get_memory_usage()
        
        # Time VAD processing
        start_time = time.time()
        segments = vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        vad_time = time.time() - start_time
        
        # Measure memory after
        mem_after = self.get_memory_usage()
        mem_used = mem_after - mem_before
        
        # Calculate metrics
        audio_duration = len(audio) / SAMPLE_RATE
        realtime_factor = audio_duration / vad_time
        
        result = {
            'name': name,
            'vad_time': vad_time,
            'audio_duration': audio_duration,
            'realtime_factor': realtime_factor,
            'memory_used_mb': mem_used,
            'num_segments': len(segments),
            'segments': segments if hasattr(segments, '__len__') else []
        }
        
        print(f"  Duration: {audio_duration:.2f}s")
        print(f"  VAD Time: {vad_time:.4f}s")
        print(f"  Realtime Factor: {realtime_factor:.2f}x")
        print(f"  Memory Used: {mem_used:.2f} MB")
        print(f"  Segments Found: {result['num_segments']}")
        
        return result
    
    def compare_vad_accuracy(self, segments1: List, segments2: List, name1: str, name2: str) -> Dict:
        """Compare accuracy between two VAD outputs"""
        print(f"\nComparing {name1} vs {name2} accuracy...")
        
        def segments_to_timeline(segments) -> List[Tuple[float, float]]:
            """Convert segments to timeline"""
            timeline = []
            for seg in segments:
                if hasattr(seg, 'start') and hasattr(seg, 'end'):
                    timeline.append((seg.start, seg.end))
                elif isinstance(seg, dict) and 'start' in seg and 'end' in seg:
                    timeline.append((seg['start'], seg['end']))
            return sorted(timeline)
        
        timeline1 = segments_to_timeline(segments1)
        timeline2 = segments_to_timeline(segments2)
        
        # Calculate overlap metrics
        total_speech1 = sum(end - start for start, end in timeline1)
        total_speech2 = sum(end - start for start, end in timeline2)
        
        # Simple overlap calculation
        overlap = 0
        for start1, end1 in timeline1:
            for start2, end2 in timeline2:
                overlap_start = max(start1, start2)
                overlap_end = min(end1, end2)
                if overlap_start < overlap_end:
                    overlap += overlap_end - overlap_start
        
        # Metrics
        precision = overlap / total_speech1 if total_speech1 > 0 else 0
        recall = overlap / total_speech2 if total_speech2 > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        result = {
            'comparison': f"{name1}_vs_{name2}",
            'total_speech_1': total_speech1,
            'total_speech_2': total_speech2,
            'overlap': overlap,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"  Total Speech {name1}: {total_speech1:.2f}s")
        print(f"  Total Speech {name2}: {total_speech2:.2f}s")
        print(f"  Overlap: {overlap:.2f}s")
        print(f"  F1 Score: {f1:.4f}")
        
        return result
    
    def test_vad_with_transcription(self, vad_name: str, audio_file: str) -> Dict:
        """Test VAD impact on transcription accuracy"""
        print(f"\nTesting {vad_name} with transcription...")
        
        # Load model with VAD
        model = whisperx.load_model(
            "tiny",
            device="cpu",
            compute_type="float32",
            backend="lightning",
            vad_method=vad_name.lower() if vad_name != "None" else None
        )
        
        # Load audio
        audio = load_audio(audio_file)
        
        # Transcribe
        start_time = time.time()
        result = model.transcribe(audio, batch_size=1)
        total_time = time.time() - start_time
        
        # Extract metrics
        text = ' '.join([seg.get('text', '') for seg in result.get('segments', [])])
        num_words = len(text.split())
        
        return {
            'vad_name': vad_name,
            'total_time': total_time,
            'num_segments': len(result.get('segments', [])),
            'num_words': num_words,
            'text_preview': text[:100] + '...' if len(text) > 100 else text
        }
    
    def run_comprehensive_benchmark(self, audio_files: List[str]):
        """Run comprehensive VAD benchmark"""
        print("="*80)
        print("WhisperX-MLX VAD Benchmark Suite")
        print("="*80)
        
        # Initialize VAD models
        print("\nInitializing VAD models...")
        device = "cpu"  # VADs run on CPU currently
        
        vad_models = {}
        
        # PyAnnote VAD
        try:
            vad_models['PyAnnote'] = Pyannote(
                device=device,
                vad_onset=0.5,
                vad_offset=0.363,
                chunk_size=30
            )
        except Exception as e:
            print(f"Failed to load PyAnnote: {e}")
        
        # Silero VAD
        try:
            vad_models['Silero'] = Silero(
                vad_onset=0.5,
                chunk_size=30
            )
        except Exception as e:
            print(f"Failed to load Silero: {e}")
        
        # TODO: Add MLX VAD when model conversion is available
        
        all_results = {
            'speed_benchmarks': [],
            'accuracy_comparisons': [],
            'transcription_tests': []
        }
        
        for audio_file in audio_files:
            print(f"\n{'='*60}")
            print(f"Testing on: {audio_file}")
            print(f"{'='*60}")
            
            # Load audio
            audio = load_audio(audio_file)
            duration = len(audio) / SAMPLE_RATE
            print(f"Audio duration: {duration:.2f}s")
            
            # Speed benchmarks
            speed_results = {}
            for name, vad in vad_models.items():
                result = self.benchmark_vad_speed(vad, audio, name)
                speed_results[name] = result
                all_results['speed_benchmarks'].append(result)
            
            # Accuracy comparisons
            if len(vad_models) >= 2:
                names = list(vad_models.keys())
                for i in range(len(names)):
                    for j in range(i+1, len(names)):
                        name1, name2 = names[i], names[j]
                        segments1 = speed_results[name1]['segments']
                        segments2 = speed_results[name2]['segments']
                        
                        comparison = self.compare_vad_accuracy(
                            segments1, segments2, name1, name2
                        )
                        all_results['accuracy_comparisons'].append(comparison)
            
            # Transcription tests
            for vad_name in list(vad_models.keys()) + ['None']:
                try:
                    trans_result = self.test_vad_with_transcription(vad_name, audio_file)
                    all_results['transcription_tests'].append(trans_result)
                except Exception as e:
                    print(f"  Failed transcription test for {vad_name}: {e}")
        
        # Generate report
        self.generate_report(all_results)
        
        # Save results
        with open('vad_benchmark_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print("\nDetailed results saved to vad_benchmark_results.json")
    
    def generate_report(self, results: Dict):
        """Generate summary report"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Speed comparison table
        if results['speed_benchmarks']:
            print("\nVAD Speed Comparison:")
            speed_data = []
            for r in results['speed_benchmarks']:
                speed_data.append([
                    r['name'],
                    f"{r['vad_time']:.4f}s",
                    f"{r['realtime_factor']:.2f}x",
                    f"{r['memory_used_mb']:.2f} MB",
                    r['num_segments']
                ])
            
            print(tabulate(speed_data,
                         headers=['VAD', 'Time', 'RT Factor', 'Memory', 'Segments'],
                         tablefmt='grid'))
        
        # Accuracy comparison table
        if results['accuracy_comparisons']:
            print("\nVAD Accuracy Comparison:")
            acc_data = []
            for r in results['accuracy_comparisons']:
                acc_data.append([
                    r['comparison'],
                    f"{r['precision']:.4f}",
                    f"{r['recall']:.4f}",
                    f"{r['f1_score']:.4f}"
                ])
            
            print(tabulate(acc_data,
                         headers=['Comparison', 'Precision', 'Recall', 'F1 Score'],
                         tablefmt='grid'))
        
        # Transcription impact table
        if results['transcription_tests']:
            print("\nVAD Impact on Transcription:")
            trans_data = []
            for r in results['transcription_tests']:
                trans_data.append([
                    r['vad_name'],
                    f"{r['total_time']:.2f}s",
                    r['num_segments'],
                    r['num_words']
                ])
            
            print(tabulate(trans_data,
                         headers=['VAD', 'Total Time', 'Segments', 'Words'],
                         tablefmt='grid'))
        
        # Key findings
        print("\nKey Findings:")
        print("-" * 40)
        
        if results['speed_benchmarks']:
            fastest = max(results['speed_benchmarks'], key=lambda x: x['realtime_factor'])
            print(f"✓ Fastest VAD: {fastest['name']} ({fastest['realtime_factor']:.2f}x realtime)")
            
            lightest = min(results['speed_benchmarks'], key=lambda x: x['memory_used_mb'])
            print(f"✓ Most memory efficient: {lightest['name']} ({lightest['memory_used_mb']:.2f} MB)")
        
        if results['accuracy_comparisons']:
            best_match = max(results['accuracy_comparisons'], key=lambda x: x['f1_score'])
            print(f"✓ Best agreement: {best_match['comparison']} (F1: {best_match['f1_score']:.4f})")

def main():
    """Run VAD benchmark"""
    
    # Test files
    audio_files = ["short.wav"]  # Add more files as needed
    
    # Run benchmark
    benchmark = VADBenchmark()
    benchmark.run_comprehensive_benchmark(audio_files)
    
    print("\nNext steps:")
    print("1. Convert Silero VAD to MLX format")
    print("2. Implement batch VAD processing")
    print("3. Test on longer audio files")
    print("4. Optimize VAD thresholds")

if __name__ == "__main__":
    main()