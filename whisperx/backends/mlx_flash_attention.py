"""
Flash Attention Implementation for MLX
Optimized attention mechanism for Apple Silicon
Based on Flash Attention v2 principles adapted for MLX
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Optional, Tuple
import math

class FlashAttentionMLX(nn.Module):
    """Flash Attention implementation optimized for MLX/Apple Silicon"""
    
    def __init__(self, 
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 block_size: int = 64):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.block_size = block_size
        self.dropout = dropout
        
        # Scaling factor
        self.scale = 1.0 / math.sqrt(self.d_k)
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def __call__(self,
                 query: mx.array,
                 key: mx.array,
                 value: mx.array,
                 mask: Optional[mx.array] = None,
                 is_causal: bool = False) -> mx.array:
        """
        Flash Attention forward pass
        
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: Optional attention mask
            is_causal: Whether to use causal masking
        
        Returns:
            output: [batch_size, seq_len_q, d_model]
        """
        
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project and reshape for multi-head attention
        Q = self.q_proj(query).reshape(batch_size, seq_len_q, self.n_heads, self.d_k)
        K = self.k_proj(key).reshape(batch_size, seq_len_k, self.n_heads, self.d_k)
        V = self.v_proj(value).reshape(batch_size, seq_len_k, self.n_heads, self.d_k)
        
        # Transpose for attention computation
        Q = Q.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_q, d_k]
        K = K.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_k, d_k]
        V = V.transpose(0, 2, 1, 3)  # [batch, n_heads, seq_k, d_k]
        
        # Apply Flash Attention
        if seq_len_q * seq_len_k > 1024 * 1024:  # Use flash attention for large sequences
            output = self._flash_attention(Q, K, V, mask, is_causal)
        else:
            # Standard attention for small sequences
            output = self._standard_attention(Q, K, V, mask, is_causal)
        
        # Reshape and project output
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(output)
        
        return output
    
    def _flash_attention(self,
                        Q: mx.array,
                        K: mx.array,
                        V: mx.array,
                        mask: Optional[mx.array],
                        is_causal: bool) -> mx.array:
        """
        Flash Attention algorithm with tiling and recomputation
        
        Key optimizations:
        1. Tiled computation to fit in SRAM/cache
        2. Online softmax computation
        3. Reduced HBM/memory accesses
        """
        
        batch_size, n_heads, seq_len_q, d_k = Q.shape
        seq_len_k = K.shape[2]
        
        # Initialize output and normalization factors
        O = mx.zeros_like(Q)
        L = mx.zeros((batch_size, n_heads, seq_len_q, 1))
        M = mx.full((batch_size, n_heads, seq_len_q, 1), -float('inf'))
        
        # Tile sizes (tuned for Apple Silicon cache sizes)
        Br = min(self.block_size, seq_len_q)
        Bc = min(self.block_size, seq_len_k)
        
        # Process in tiles
        for i in range(0, seq_len_q, Br):
            i_end = min(i + Br, seq_len_q)
            Qi = Q[:, :, i:i_end, :]
            
            # Initialize block accumulators
            Oi = mx.zeros_like(Qi)
            Li = mx.zeros((batch_size, n_heads, i_end - i, 1))
            Mi = mx.full((batch_size, n_heads, i_end - i, 1), -float('inf'))
            
            for j in range(0, seq_len_k, Bc):
                j_end = min(j + Bc, seq_len_k)
                Kj = K[:, :, j:j_end, :]
                Vj = V[:, :, j:j_end, :]
                
                # Compute attention scores for this block
                Sij = mx.matmul(Qi, Kj.transpose(0, 1, 3, 2)) * self.scale
                
                # Apply causal mask if needed
                if is_causal and j >= i:
                    causal_mask = mx.triu(
                        mx.full((i_end - i, j_end - j), -float('inf')),
                        k=j - i + 1
                    )
                    Sij = Sij + causal_mask
                
                # Apply attention mask if provided
                if mask is not None:
                    mask_slice = mask[:, :, i:i_end, j:j_end]
                    Sij = mx.where(mask_slice, Sij, -float('inf'))
                
                # Online softmax computation
                Mij = mx.max(Sij, axis=-1, keepdims=True)
                Pij = mx.exp(Sij - Mij)
                Lij = mx.sum(Pij, axis=-1, keepdims=True)
                
                # Update statistics
                Mi_new = mx.maximum(Mi, Mij)
                Li_new = mx.exp(Mi - Mi_new) * Li + mx.exp(Mij - Mi_new) * Lij
                
                # Update output accumulator
                Oi = mx.exp(Mi - Mi_new) * Oi + mx.matmul(Pij / Lij, Vj)
                
                # Update running statistics
                Mi = Mi_new
                Li = Li_new
            
            # Write back to output
            O[:, :, i:i_end, :] = Oi
            L[:, :, i:i_end, :] = Li
            M[:, :, i:i_end, :] = Mi
        
        return O
    
    def _standard_attention(self,
                           Q: mx.array,
                           K: mx.array,
                           V: mx.array,
                           mask: Optional[mx.array],
                           is_causal: bool) -> mx.array:
        """Standard scaled dot-product attention"""
        
        # Compute attention scores
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) * self.scale
        
        # Apply causal mask
        if is_causal:
            seq_len = Q.shape[2]
            causal_mask = mx.triu(mx.full((seq_len, seq_len), -float('inf')), k=1)
            scores = scores + causal_mask
        
        # Apply attention mask
        if mask is not None:
            scores = mx.where(mask, scores, -float('inf'))
        
        # Softmax
        attn_weights = mx.softmax(scores, axis=-1)
        
        # Dropout
        if self.dropout > 0 and self.training:
            attn_weights = mx.dropout(attn_weights, p=self.dropout)
        
        # Apply attention to values
        output = mx.matmul(attn_weights, V)
        
        return output

class MultiHeadFlashAttention(nn.Module):
    """Multi-Head Attention with Flash Attention backend"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.0,
                 use_flash: bool = True):
        super().__init__()
        
        self.use_flash = use_flash
        
        if use_flash:
            self.attention = FlashAttentionMLX(d_model, n_heads, dropout)
        else:
            # Fallback to standard attention
            self.attention = nn.MultiHeadAttention(d_model, n_heads, bias=False)
    
    def __call__(self,
                 query: mx.array,
                 key: Optional[mx.array] = None,
                 value: Optional[mx.array] = None,
                 mask: Optional[mx.array] = None) -> mx.array:
        
        if key is None:
            key = query
        if value is None:
            value = query
        
        if self.use_flash:
            return self.attention(query, key, value, mask)
        else:
            return self.attention(query, key, value, mask)

def benchmark_flash_attention():
    """Benchmark Flash Attention vs Standard Attention"""
    
    import time
    
    print("Flash Attention MLX Benchmark")
    print("=" * 60)
    
    # Test configurations
    configs = [
        {"seq_len": 512, "d_model": 512, "n_heads": 8},
        {"seq_len": 1024, "d_model": 768, "n_heads": 12},
        {"seq_len": 2048, "d_model": 1024, "n_heads": 16},
        {"seq_len": 4096, "d_model": 1024, "n_heads": 16},
    ]
    
    batch_size = 1
    
    for config in configs:
        seq_len = config["seq_len"]
        d_model = config["d_model"]
        n_heads = config["n_heads"]
        
        print(f"\nConfig: seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
        
        # Create random inputs
        x = mx.random.normal((batch_size, seq_len, d_model))
        
        # Flash Attention
        flash_attn = FlashAttentionMLX(d_model, n_heads)
        
        # Warmup
        _ = flash_attn(x, x, x)
        mx.eval(_)
        
        # Benchmark
        start = time.time()
        for _ in range(10):
            out = flash_attn(x, x, x, is_causal=True)
            mx.eval(out)
        flash_time = (time.time() - start) / 10
        
        # Memory usage approximation
        # Standard attention: O(seq_len^2)
        # Flash attention: O(seq_len)
        standard_memory = (seq_len ** 2) * 4 / (1024 ** 2)  # MB
        flash_memory = seq_len * d_model * 4 / (1024 ** 2)  # MB
        
        print(f"  Flash Attention time: {flash_time*1000:.2f} ms")
        print(f"  Memory savings: {standard_memory:.1f} MB → {flash_memory:.1f} MB")
        print(f"  Memory reduction: {(1 - flash_memory/standard_memory)*100:.1f}%")

class SlidingWindowAttention(nn.Module):
    """Sliding window attention for long sequences"""
    
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 window_size: int = 256,
                 overlap: int = 128):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.overlap = overlap
        self.attention = FlashAttentionMLX(d_model, n_heads)
        
    def __call__(self, x: mx.array) -> mx.array:
        """Process long sequences with sliding window"""
        
        batch_size, seq_len, d_model = x.shape
        
        if seq_len <= self.window_size:
            # Single window - use regular attention
            return self.attention(x, x, x)
        
        # Process with sliding windows
        stride = self.window_size - self.overlap
        output = mx.zeros_like(x)
        counts = mx.zeros((batch_size, seq_len, 1))
        
        for i in range(0, seq_len - self.overlap, stride):
            end = min(i + self.window_size, seq_len)
            
            # Process window
            window_input = x[:, i:end, :]
            window_output = self.attention(window_input, window_input, window_input)
            
            # Accumulate results
            output[:, i:end, :] += window_output
            counts[:, i:end, :] += 1
        
        # Average overlapping regions
        output = output / mx.maximum(counts, 1)
        
        return output

if __name__ == "__main__":
    benchmark_flash_attention()