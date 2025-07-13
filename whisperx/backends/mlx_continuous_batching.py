"""
Continuous Batching System for WhisperX-MLX
Dynamic request scheduling and efficient batch processing
"""

import mlx.core as mx
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import time
import threading
import queue
from collections import deque
import heapq

@dataclass
class TranscriptionRequest:
    """Single transcription request"""
    request_id: str
    audio: np.ndarray
    priority: int = 0  # Higher priority = processed first
    timestamp: float = field(default_factory=time.time)
    options: Dict[str, Any] = field(default_factory=dict)
    callback: Optional[callable] = None
    
    def __lt__(self, other):
        # For priority queue - higher priority first, then earlier timestamp
        return (-self.priority, self.timestamp) < (-other.priority, other.timestamp)

@dataclass
class BatchConfig:
    """Configuration for continuous batching"""
    max_batch_size: int = 16
    max_wait_time: float = 0.1  # Max time to wait for batch to fill
    min_batch_size: int = 1     # Min batch size to process
    memory_limit_mb: int = 4096  # Memory limit for batching
    enable_padding_opt: bool = True  # Enable padding optimization
    enable_bucketing: bool = True    # Enable length-based bucketing
    bucket_boundaries: List[float] = field(default_factory=lambda: [5, 10, 20, 30, 60])  # Audio length buckets in seconds

class RequestQueue:
    """Priority queue for managing transcription requests"""
    
    def __init__(self):
        self._queue = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._request_map = {}  # For request tracking
        
    def put(self, request: TranscriptionRequest):
        """Add request to queue"""
        with self._lock:
            heapq.heappush(self._queue, request)
            self._request_map[request.request_id] = request
            self._condition.notify()
    
    def get(self, timeout: Optional[float] = None) -> Optional[TranscriptionRequest]:
        """Get highest priority request"""
        with self._condition:
            if not self._queue and timeout is not None:
                self._condition.wait(timeout)
            
            if self._queue:
                request = heapq.heappop(self._queue)
                self._request_map.pop(request.request_id, None)
                return request
            return None
    
    def get_batch(self, max_size: int, max_wait: float) -> List[TranscriptionRequest]:
        """Get a batch of requests"""
        batch = []
        deadline = time.time() + max_wait
        
        with self._condition:
            while len(batch) < max_size:
                remaining_time = deadline - time.time()
                if remaining_time <= 0:
                    break
                
                if self._queue:
                    request = heapq.heappop(self._queue)
                    self._request_map.pop(request.request_id, None)
                    batch.append(request)
                else:
                    self._condition.wait(remaining_time)
            
        return batch
    
    def size(self) -> int:
        """Get queue size"""
        with self._lock:
            return len(self._queue)

class BatchOptimizer:
    """Optimizes batch composition for efficient processing"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        
    def create_buckets(self, requests: List[TranscriptionRequest]) -> Dict[int, List[TranscriptionRequest]]:
        """Bucket requests by audio length"""
        buckets = {}
        
        for request in requests:
            duration = len(request.audio) / 16000  # Assuming 16kHz
            bucket_idx = np.searchsorted(self.config.bucket_boundaries, duration)
            
            if bucket_idx not in buckets:
                buckets[bucket_idx] = []
            buckets[bucket_idx].append(request)
        
        return buckets
    
    def optimize_batch(self, requests: List[TranscriptionRequest]) -> List[List[TranscriptionRequest]]:
        """Optimize requests into efficient batches"""
        
        if not self.config.enable_bucketing:
            # Simple batching without optimization
            batches = []
            for i in range(0, len(requests), self.config.max_batch_size):
                batches.append(requests[i:i + self.config.max_batch_size])
            return batches
        
        # Bucket by length for efficient padding
        buckets = self.create_buckets(requests)
        batches = []
        
        # Create batches from each bucket
        for bucket_requests in buckets.values():
            # Sort by length within bucket for better padding
            bucket_requests.sort(key=lambda r: len(r.audio))
            
            # Create batches
            for i in range(0, len(bucket_requests), self.config.max_batch_size):
                batch = bucket_requests[i:i + self.config.max_batch_size]
                batches.append(batch)
        
        return batches
    
    def estimate_memory(self, batch: List[TranscriptionRequest]) -> float:
        """Estimate memory usage for a batch in MB"""
        if not batch:
            return 0.0
        
        # Find max length for padding
        max_length = max(len(r.audio) for r in batch)
        batch_size = len(batch)
        
        # Estimate memory: batch_size * max_length * 4 bytes (float32)
        memory_mb = (batch_size * max_length * 4) / (1024 * 1024)
        
        # Add overhead for features and intermediate tensors (3x multiplier)
        return memory_mb * 3

class ContinuousBatcher:
    """Main continuous batching engine"""
    
    def __init__(self, backend, config: Optional[BatchConfig] = None):
        self.backend = backend
        self.config = config or BatchConfig()
        self.request_queue = RequestQueue()
        self.optimizer = BatchOptimizer(self.config)
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "total_batches": 0,
            "total_audio_seconds": 0.0,
            "total_processing_time": 0.0,
            "average_batch_size": 0.0,
            "queue_wait_times": deque(maxlen=1000)
        }
        
        # Processing thread
        self._running = False
        self._thread = None
        
    def start(self):
        """Start the batching engine"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the batching engine"""
        self._running = False
        if self._thread:
            self._thread.join()
    
    def submit(self, request: TranscriptionRequest) -> str:
        """Submit a request for processing"""
        self.request_queue.put(request)
        self.stats["total_requests"] += 1
        return request.request_id
    
    def _process_loop(self):
        """Main processing loop"""
        while self._running:
            # Collect requests for batching
            requests = self.request_queue.get_batch(
                self.config.max_batch_size,
                self.config.max_wait_time
            )
            
            if not requests:
                continue
            
            # Update wait time statistics
            current_time = time.time()
            for req in requests:
                wait_time = current_time - req.timestamp
                self.stats["queue_wait_times"].append(wait_time)
            
            # Optimize into batches
            batches = self.optimizer.optimize_batch(requests)
            
            # Process each batch
            for batch in batches:
                self._process_batch(batch)
    
    def _process_batch(self, batch: List[TranscriptionRequest]):
        """Process a single batch of requests"""
        
        start_time = time.time()
        self.stats["total_batches"] += 1
        
        # Check memory constraints
        estimated_memory = self.optimizer.estimate_memory(batch)
        if estimated_memory > self.config.memory_limit_mb:
            # Split batch if too large
            mid = len(batch) // 2
            self._process_batch(batch[:mid])
            self._process_batch(batch[mid:])
            return
        
        # Prepare batch data
        batch_audio, batch_lengths, padding_mask = self._prepare_batch(batch)
        
        try:
            # Process with backend
            results = self.backend.transcribe_batch(
                batch_audio,
                batch_lengths=batch_lengths,
                padding_mask=padding_mask
            )
            
            # Distribute results
            for i, request in enumerate(batch):
                result = self._extract_result(results, i, batch_lengths[i])
                
                # Execute callback if provided
                if request.callback:
                    request.callback(request.request_id, result)
                    
                # Update statistics
                audio_duration = len(request.audio) / 16000
                self.stats["total_audio_seconds"] += audio_duration
                
        except Exception as e:
            # Handle errors
            error_result = {"error": str(e)}
            for request in batch:
                if request.callback:
                    request.callback(request.request_id, error_result)
        
        # Update processing time
        processing_time = time.time() - start_time
        self.stats["total_processing_time"] += processing_time
        
        # Update average batch size
        total_batches = self.stats["total_batches"]
        self.stats["average_batch_size"] = (
            (self.stats["average_batch_size"] * (total_batches - 1) + len(batch)) / total_batches
        )
    
    def _prepare_batch(self, batch: List[TranscriptionRequest]) -> Tuple[mx.array, List[int], mx.array]:
        """Prepare batch data with padding"""
        
        # Get audio lengths
        lengths = [len(r.audio) for r in batch]
        max_length = max(lengths)
        
        # Pad audio to same length
        padded_audio = []
        for i, request in enumerate(batch):
            audio = request.audio
            if len(audio) < max_length:
                # Pad with zeros
                pad_length = max_length - len(audio)
                audio = np.concatenate([audio, np.zeros(pad_length)])
            padded_audio.append(audio)
        
        # Convert to MLX arrays
        batch_audio = mx.array(np.stack(padded_audio))
        
        # Create padding mask (1 for valid, 0 for padding)
        padding_mask = mx.zeros((len(batch), max_length))
        for i, length in enumerate(lengths):
            padding_mask[i, :length] = 1
        
        return batch_audio, lengths, padding_mask
    
    def _extract_result(self, batch_results: Dict[str, Any], idx: int, original_length: int) -> Dict[str, Any]:
        """Extract individual result from batch results"""
        
        result = {}
        
        # Extract segments for this index
        if "segments" in batch_results:
            all_segments = batch_results["segments"]
            result["segments"] = all_segments[idx] if idx < len(all_segments) else []
        
        # Extract text
        if "text" in batch_results:
            all_text = batch_results["text"]
            result["text"] = all_text[idx] if isinstance(all_text, list) else all_text
        
        # Add metadata
        result["audio_length"] = original_length / 16000
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics"""
        stats = self.stats.copy()
        
        # Calculate derived metrics
        if stats["total_processing_time"] > 0:
            stats["throughput"] = stats["total_audio_seconds"] / stats["total_processing_time"]
        else:
            stats["throughput"] = 0.0
        
        # Average queue wait time
        if stats["queue_wait_times"]:
            stats["avg_queue_wait"] = np.mean(list(stats["queue_wait_times"]))
        else:
            stats["avg_queue_wait"] = 0.0
        
        return stats

def benchmark_continuous_batching():
    """Benchmark continuous batching performance"""
    
    print("Continuous Batching Benchmark")
    print("=" * 60)
    
    # Simulate different load patterns
    patterns = [
        {
            "name": "Steady Load",
            "requests": 100,
            "arrival_rate": 10,  # requests per second
            "audio_lengths": [10, 10, 10]  # Uniform 10s audio
        },
        {
            "name": "Bursty Load",
            "requests": 100,
            "arrival_rate": 50,  # High burst rate
            "audio_lengths": [5, 10, 20, 30]  # Mixed lengths
        },
        {
            "name": "Variable Load",
            "requests": 100,
            "arrival_rate": 20,
            "audio_lengths": [2, 5, 10, 20, 30, 60]  # Wide range
        }
    ]
    
    for pattern in patterns:
        print(f"\n{pattern['name']}:")
        print(f"  Requests: {pattern['requests']}")
        print(f"  Arrival rate: {pattern['arrival_rate']} req/s")
        
        # Calculate theoretical metrics
        avg_audio_length = np.mean(pattern['audio_lengths'])
        total_audio = pattern['requests'] * avg_audio_length
        
        # Without batching (sequential)
        sequential_time = pattern['requests'] * 0.5  # Assume 0.5s per request
        
        # With batching (assuming 4x speedup from batching)
        batch_speedup = 4.0
        batched_time = sequential_time / batch_speedup
        
        print(f"  Total audio: {total_audio:.1f}s")
        print(f"  Sequential processing: {sequential_time:.1f}s")
        print(f"  Batched processing: {batched_time:.1f}s")
        print(f"  Speedup: {sequential_time/batched_time:.2f}x")
        print(f"  Throughput: {total_audio/batched_time:.1f}x realtime")

class DynamicBatchScheduler:
    """Advanced scheduler with dynamic batch size adjustment"""
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.history = deque(maxlen=100)
        self.current_batch_size = config.max_batch_size // 2
        
    def adjust_batch_size(self, queue_size: int, avg_wait_time: float):
        """Dynamically adjust batch size based on queue metrics"""
        
        # If queue is growing and wait times are high, increase batch size
        if queue_size > 20 and avg_wait_time > 0.2:
            self.current_batch_size = min(
                self.current_batch_size + 2,
                self.config.max_batch_size
            )
        # If queue is small and wait times are low, decrease batch size
        elif queue_size < 5 and avg_wait_time < 0.05:
            self.current_batch_size = max(
                self.current_batch_size - 1,
                self.config.min_batch_size
            )
        
        return self.current_batch_size

if __name__ == "__main__":
    benchmark_continuous_batching()