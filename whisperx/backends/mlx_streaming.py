"""
Real-Time Streaming Support for WhisperX-MLX
Low-latency streaming transcription with adaptive chunking
"""

import mlx.core as mx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Iterator
from dataclasses import dataclass
import time
import threading
from collections import deque
import queue

@dataclass 
class StreamingConfig:
    """Configuration for streaming transcription"""
    chunk_length: float = 1.0  # Chunk length in seconds
    overlap: float = 0.1       # Overlap between chunks in seconds
    min_silence_duration: float = 0.3  # Minimum silence to trigger processing
    max_latency: float = 2.0   # Maximum allowed latency
    lookahead_chunks: int = 2  # Number of chunks to look ahead
    enable_vad: bool = True    # Use VAD for smart chunking
    sample_rate: int = 16000   # Audio sample rate
    
    @property
    def chunk_samples(self) -> int:
        return int(self.chunk_length * self.sample_rate)
    
    @property
    def overlap_samples(self) -> int:
        return int(self.overlap * self.sample_rate)

class AudioBuffer:
    """Circular buffer for streaming audio"""
    
    def __init__(self, capacity_seconds: float = 30.0, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.capacity = int(capacity_seconds * sample_rate)
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        self.write_pos = 0
        self.read_pos = 0
        self.lock = threading.Lock()
        
    def write(self, audio: np.ndarray):
        """Write audio to buffer"""
        with self.lock:
            samples = len(audio)
            
            # Handle wrap-around
            if self.write_pos + samples <= self.capacity:
                self.buffer[self.write_pos:self.write_pos + samples] = audio
            else:
                # Split write at boundary
                first_part = self.capacity - self.write_pos
                self.buffer[self.write_pos:] = audio[:first_part]
                self.buffer[:samples - first_part] = audio[first_part:]
            
            self.write_pos = (self.write_pos + samples) % self.capacity
    
    def read(self, samples: int) -> Optional[np.ndarray]:
        """Read audio from buffer"""
        with self.lock:
            available = self.available_samples()
            
            if available < samples:
                return None
            
            # Read from buffer
            if self.read_pos + samples <= self.capacity:
                audio = self.buffer[self.read_pos:self.read_pos + samples].copy()
            else:
                # Handle wrap-around
                first_part = self.capacity - self.read_pos
                audio = np.concatenate([
                    self.buffer[self.read_pos:],
                    self.buffer[:samples - first_part]
                ])
            
            self.read_pos = (self.read_pos + samples) % self.capacity
            return audio
    
    def peek(self, samples: int, offset: int = 0) -> Optional[np.ndarray]:
        """Peek at audio without consuming"""
        with self.lock:
            available = self.available_samples()
            
            if available < samples + offset:
                return None
            
            peek_pos = (self.read_pos + offset) % self.capacity
            
            if peek_pos + samples <= self.capacity:
                audio = self.buffer[peek_pos:peek_pos + samples].copy()
            else:
                # Handle wrap-around
                first_part = self.capacity - peek_pos
                audio = np.concatenate([
                    self.buffer[peek_pos:],
                    self.buffer[:samples - first_part]
                ])
            
            return audio
    
    def available_samples(self) -> int:
        """Get number of available samples"""
        if self.write_pos >= self.read_pos:
            return self.write_pos - self.read_pos
        else:
            return self.capacity - self.read_pos + self.write_pos
    
    def clear(self):
        """Clear the buffer"""
        with self.lock:
            self.write_pos = 0
            self.read_pos = 0
            self.buffer.fill(0)

class StreamingChunker:
    """Intelligent chunking for streaming audio"""
    
    def __init__(self, config: StreamingConfig, vad_model=None):
        self.config = config
        self.vad_model = vad_model
        self.silence_samples = 0
        self.speech_buffer = []
        self.last_speech_end = 0
        
    def process_audio(self, audio_buffer: AudioBuffer) -> Iterator[Tuple[np.ndarray, float]]:
        """Process audio buffer and yield chunks for transcription"""
        
        while True:
            available = audio_buffer.available_samples()
            
            if available < self.config.chunk_samples:
                break
            
            # Peek at next chunk
            chunk = audio_buffer.peek(self.config.chunk_samples)
            if chunk is None:
                break
            
            if self.config.enable_vad and self.vad_model:
                # Use VAD to detect speech boundaries
                timestamps = self._get_speech_timestamps(chunk)
                
                if not timestamps:
                    # No speech - accumulate silence
                    self.silence_samples += self.config.chunk_samples
                    
                    # If we have accumulated speech and enough silence, yield it
                    if (self.speech_buffer and 
                        self.silence_samples >= self.config.min_silence_duration * self.config.sample_rate):
                        
                        # Yield accumulated speech
                        speech_audio = np.concatenate(self.speech_buffer)
                        timestamp = self.last_speech_end
                        self.speech_buffer = []
                        
                        yield speech_audio, timestamp
                    
                    # Consume the silent chunk
                    audio_buffer.read(self.config.chunk_samples - self.config.overlap_samples)
                    
                else:
                    # Speech detected
                    self.silence_samples = 0
                    self.speech_buffer.append(chunk)
                    self.last_speech_end = time.time()
                    
                    # Check if we should yield based on length
                    total_samples = sum(len(b) for b in self.speech_buffer)
                    if total_samples >= self.config.max_latency * self.config.sample_rate:
                        speech_audio = np.concatenate(self.speech_buffer)
                        timestamp = self.last_speech_end
                        self.speech_buffer = []
                        
                        yield speech_audio, timestamp
                    
                    # Consume chunk with overlap
                    audio_buffer.read(self.config.chunk_samples - self.config.overlap_samples)
            
            else:
                # No VAD - use fixed chunking
                chunk = audio_buffer.read(self.config.chunk_samples)
                if chunk is not None:
                    yield chunk, time.time()
    
    def _get_speech_timestamps(self, audio: np.ndarray) -> List[Dict[str, float]]:
        """Get speech timestamps using VAD"""
        if self.vad_model is None:
            return [{"start": 0, "end": len(audio) / self.config.sample_rate}]
        
        # Run VAD
        timestamps = self.vad_model(audio, self.config.sample_rate)
        return timestamps

class StreamingTranscriber:
    """Main streaming transcription engine"""
    
    def __init__(self, backend, config: Optional[StreamingConfig] = None):
        self.backend = backend
        self.config = config or StreamingConfig()
        self.audio_buffer = AudioBuffer()
        self.chunker = StreamingChunker(self.config)
        
        # Result management
        self.results_queue = queue.Queue()
        self.transcript_buffer = deque(maxlen=100)
        
        # State management
        self.is_running = False
        self.processing_thread = None
        
        # Context for better transcription
        self.context_tokens = []
        self.prev_text = ""
        
    def start(self):
        """Start streaming transcription"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        self.processing_thread.start()
    
    def stop(self):
        """Stop streaming transcription"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def write_audio(self, audio: np.ndarray):
        """Write audio to the stream"""
        self.audio_buffer.write(audio)
    
    def read_results(self, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """Read transcription results"""
        try:
            return self.results_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def _processing_loop(self):
        """Main processing loop for streaming transcription"""
        
        while self.is_running:
            try:
                # Process audio chunks
                for audio_chunk, timestamp in self.chunker.process_audio(self.audio_buffer):
                    if not self.is_running:
                        break
                    
                    # Transcribe chunk
                    result = self._transcribe_chunk(audio_chunk, timestamp)
                    
                    if result:
                        # Post-process and emit result
                        processed = self._post_process_result(result, timestamp)
                        self.results_queue.put(processed)
                
                # Small sleep to prevent busy waiting
                time.sleep(0.01)
                
            except Exception as e:
                # Emit error
                self.results_queue.put({
                    "error": str(e),
                    "timestamp": time.time()
                })
    
    def _transcribe_chunk(self, audio: np.ndarray, timestamp: float) -> Optional[Dict[str, Any]]:
        """Transcribe a single audio chunk"""
        
        try:
            # Add context from previous transcriptions
            options = {
                "initial_prompt": self.prev_text[-200:] if self.prev_text else None,
                "language": "en",  # Can be made configurable
                "temperature": 0.0,  # Greedy for consistency
                "no_speech_threshold": 0.6,
                "word_timestamps": True
            }
            
            # Transcribe
            result = self.backend.transcribe(audio, **options)
            
            # Update context
            if result and "text" in result:
                self.prev_text = result["text"]
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    def _post_process_result(self, result: Dict[str, Any], timestamp: float) -> Dict[str, Any]:
        """Post-process transcription result"""
        
        # Add metadata
        result["timestamp"] = timestamp
        result["latency"] = time.time() - timestamp
        
        # Stabilize output by comparing with previous results
        if "segments" in result:
            result["segments"] = self._stabilize_segments(result["segments"])
        
        # Add to transcript buffer
        self.transcript_buffer.append(result)
        
        # Generate incremental and final transcripts
        result["incremental"] = self._generate_incremental_transcript()
        result["is_final"] = self._is_final_result(result)
        
        return result
    
    def _stabilize_segments(self, segments: List[Dict]) -> List[Dict]:
        """Stabilize segments by comparing with previous results"""
        
        # Simple stabilization - can be made more sophisticated
        stabilized = []
        
        for segment in segments:
            # Check confidence
            if segment.get("no_speech_prob", 0) < 0.9:
                stabilized.append(segment)
        
        return stabilized
    
    def _generate_incremental_transcript(self) -> str:
        """Generate incremental transcript from buffer"""
        
        texts = []
        for result in self.transcript_buffer:
            if "text" in result and result["text"].strip():
                texts.append(result["text"].strip())
        
        return " ".join(texts)
    
    def _is_final_result(self, result: Dict[str, Any]) -> bool:
        """Determine if result is final or incremental"""
        
        # Simple heuristic - can be improved
        if "segments" not in result:
            return False
        
        # Check if we have silence after speech
        last_segment = result["segments"][-1] if result["segments"] else None
        if last_segment and "end" in last_segment:
            silence_duration = time.time() - result["timestamp"] - last_segment["end"]
            return silence_duration > self.config.min_silence_duration
        
        return False

class RealtimeDemo:
    """Demo application for real-time streaming"""
    
    def __init__(self, model_name: str = "tiny"):
        # Initialize backend
        import whisperx
        self.model = whisperx.load_model(
            model_name,
            device="cpu",
            backend="lightning"
        )
        
        # Initialize streaming
        config = StreamingConfig(
            chunk_length=1.0,
            overlap=0.1,
            min_silence_duration=0.5,
            enable_vad=True
        )
        self.streamer = StreamingTranscriber(self.model, config)
        
    def simulate_microphone(self, audio_file: str):
        """Simulate microphone input from file"""
        
        import whisperx
        
        # Load audio
        audio = whisperx.load_audio(audio_file)
        sample_rate = 16000
        
        # Start streaming
        self.streamer.start()
        
        print("Streaming transcription started...")
        print("-" * 60)
        
        # Simulate real-time audio chunks
        chunk_size = sample_rate  # 1 second chunks
        
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            
            # Write to stream
            self.streamer.write_audio(chunk)
            
            # Read results
            while True:
                result = self.streamer.read_results(timeout=0.1)
                if result is None:
                    break
                
                if "error" in result:
                    print(f"Error: {result['error']}")
                elif "text" in result:
                    latency = result.get("latency", 0) * 1000
                    is_final = result.get("is_final", False)
                    
                    marker = "[FINAL]" if is_final else "[PARTIAL]"
                    print(f"{marker} ({latency:.0f}ms): {result['text']}")
            
            # Simulate real-time delay
            time.sleep(1.0)
        
        # Stop streaming
        self.streamer.stop()
        print("\nStreaming stopped.")

def benchmark_streaming():
    """Benchmark streaming performance"""
    
    print("Streaming Transcription Benchmark")
    print("=" * 60)
    
    configs = [
        {
            "name": "Low Latency",
            "chunk_length": 0.5,
            "overlap": 0.1,
            "max_latency": 1.0
        },
        {
            "name": "Balanced",
            "chunk_length": 1.0,
            "overlap": 0.2,
            "max_latency": 2.0
        },
        {
            "name": "High Accuracy",
            "chunk_length": 2.0,
            "overlap": 0.5,
            "max_latency": 3.0
        }
    ]
    
    for config in configs:
        print(f"\n{config['name']} Configuration:")
        print(f"  Chunk length: {config['chunk_length']}s")
        print(f"  Overlap: {config['overlap']}s")
        print(f"  Max latency: {config['max_latency']}s")
        
        # Calculate theoretical latency
        min_latency = config['chunk_length'] * 0.1  # Processing time
        avg_latency = config['chunk_length'] / 2 + min_latency
        max_latency = config['max_latency']
        
        print(f"  Expected latency: {avg_latency:.1f}s (min: {min_latency:.1f}s, max: {max_latency:.1f}s)")
        
        # Memory usage
        buffer_seconds = 30
        memory_mb = buffer_seconds * 16000 * 4 / (1024 * 1024)
        print(f"  Buffer memory: {memory_mb:.1f} MB")

if __name__ == "__main__":
    benchmark_streaming()
    
    # Demo streaming (commented out - requires audio file)
    # demo = RealtimeDemo("tiny")
    # demo.simulate_microphone("test_audio.wav")