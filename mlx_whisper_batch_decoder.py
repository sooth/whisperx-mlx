"""
MLX Whisper Batch Decoder - True parallel decoding implementation
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

from mlx_whisper.decoding import (
    DecodingOptions, DecodingResult, DecodingTask,
    GreedyDecoder, TokenDecoder, LogitFilter
)
from mlx_whisper.tokenizer import Tokenizer


class BatchInference:
    """
    Batch-aware inference that properly handles KV cache for multiple sequences.
    
    Key innovation: Instead of sharing KV cache across sequences, we maintain
    proper separation using masking and index tracking.
    """
    
    def __init__(self, model: "Whisper", batch_size: int):
        self.model = model
        self.batch_size = batch_size
        self.reset()
        
    def reset(self):
        """Reset state for new batch."""
        self.kv_cache = None
        self.sequence_lengths = mx.zeros(self.batch_size, dtype=mx.int32)
        self.cache_positions = mx.zeros(self.batch_size, dtype=mx.int32)
        
    def logits(self, tokens: mx.array, audio_features: mx.array,
               sequence_mask: Optional[mx.array] = None) -> mx.array:
        """
        Compute logits for batch with proper KV cache handling.
        
        Args:
            tokens: (batch_size, seq_len) or (batch_size, 1) for single token
            audio_features: (batch_size, n_audio_ctx, n_audio_state)
            sequence_mask: (batch_size,) boolean mask of active sequences
            
        Returns:
            logits: (batch_size, vocab_size)
        """
        # If no mask provided, assume all sequences are active
        if sequence_mask is None:
            sequence_mask = mx.ones(self.batch_size, dtype=mx.bool_)
            
        # Get active indices
        # Convert to numpy for boolean indexing, then back to mx
        mask_np = np.array(sequence_mask)
        active_indices_list = np.where(mask_np)[0].tolist()
        
        if len(active_indices_list) == 0:
            # No active sequences
            return mx.zeros((self.batch_size, self.model.dims.n_vocab))
            
        # Extract active sequences using indices
        active_tokens = mx.take(tokens, mx.array(active_indices_list), axis=0)
        active_audio = mx.take(audio_features, mx.array(active_indices_list), axis=0)
        
        # Forward through decoder
        if self.kv_cache is None:
            # First iteration - no KV cache yet
            logits, new_kv_cache, _ = self.model.decoder(
                active_tokens, active_audio
            )
            # Take only the last token's logits
            logits = logits[:, -1]
            
            # Initialize full-size KV cache
            self.kv_cache = self._expand_kv_cache(new_kv_cache, mx.array(active_indices_list))
        else:
            # Subsequent iterations - use and update KV cache
            # Extract KV cache for active sequences
            active_kv_cache = self._extract_active_kv_cache(mx.array(active_indices_list))
            
            # Forward pass
            logits, new_kv_cache, _ = self.model.decoder(
                active_tokens, active_audio, kv_cache=active_kv_cache
            )
            # Take only the last token's logits
            logits = logits[:, -1]
            
            # Update KV cache for active sequences
            self._update_kv_cache(new_kv_cache, mx.array(active_indices_list))
            
        # Expand logits back to full batch size
        full_logits = mx.zeros((self.batch_size, logits.shape[-1]))
        
        # Scatter the active logits back to their positions
        for i, idx in enumerate(active_indices_list):
            full_logits[idx] = logits[i]
            
        return full_logits.astype(mx.float32)
        
    def _expand_kv_cache(self, active_kv_cache: List, active_indices: mx.array) -> List:
        """Expand KV cache from active sequences to full batch size."""
        full_kv_cache = []
        
        for block_cache in active_kv_cache:
            if block_cache is None:
                full_kv_cache.append(None)
                continue
            
            # Block cache is (self_attn_cache, cross_attn_cache)
            if isinstance(block_cache, (list, tuple)) and len(block_cache) == 2:
                self_attn_cache, cross_attn_cache = block_cache
                
                # Process self-attention cache
                if self_attn_cache is not None and isinstance(self_attn_cache, (list, tuple)) and len(self_attn_cache) == 2:
                    key, value = self_attn_cache
                else:
                    full_kv_cache.append(block_cache)
                    continue
            else:
                # Skip if format is unexpected
                full_kv_cache.append(None)
                continue
            
            # Create full-size cache filled with zeros
            full_key = mx.zeros(
                (self.batch_size,) + key.shape[1:],
                dtype=key.dtype
            )
            full_value = mx.zeros(
                (self.batch_size,) + value.shape[1:],
                dtype=value.dtype
            )
            
            # Copy active caches to their positions
            for i, idx in enumerate(active_indices):
                idx_val = idx.item()
                full_key[idx_val] = key[i]
                full_value[idx_val] = value[i]
            
            # Reconstruct the block cache with expanded self-attention
            full_self_attn = (full_key, full_value)
            
            # Handle cross-attention cache
            if cross_attn_cache is not None and isinstance(cross_attn_cache, (list, tuple)) and len(cross_attn_cache) == 2:
                cross_key, cross_value = cross_attn_cache
                
                # Expand cross-attention cache
                full_cross_key = mx.zeros(
                    (self.batch_size,) + cross_key.shape[1:],
                    dtype=cross_key.dtype
                )
                full_cross_value = mx.zeros(
                    (self.batch_size,) + cross_value.shape[1:],
                    dtype=cross_value.dtype
                )
                
                for i, idx in enumerate(active_indices):
                    idx_val = idx.item()
                    full_cross_key[idx_val] = cross_key[i]
                    full_cross_value[idx_val] = cross_value[i]
                
                full_cross_attn = (full_cross_key, full_cross_value)
            else:
                full_cross_attn = cross_attn_cache
                
            full_kv_cache.append((full_self_attn, full_cross_attn))
            
        return full_kv_cache
        
    def _extract_active_kv_cache(self, active_indices: mx.array) -> List:
        """Extract KV cache for active sequences."""
        active_kv_cache = []
        
        for block_cache in self.kv_cache:
            if block_cache is None:
                active_kv_cache.append(None)
                continue
            
            # Block cache is (self_attn_cache, cross_attn_cache)
            if isinstance(block_cache, (list, tuple)) and len(block_cache) == 2:
                self_attn_cache, cross_attn_cache = block_cache
                
                # Extract self-attention cache
                if self_attn_cache is not None and isinstance(self_attn_cache, (list, tuple)) and len(self_attn_cache) == 2:
                    key, value = self_attn_cache
                    active_key = mx.take(key, active_indices, axis=0)
                    active_value = mx.take(value, active_indices, axis=0)
                    active_self_attn = (active_key, active_value)
                else:
                    active_self_attn = self_attn_cache
                
                # Extract cross-attention cache
                if cross_attn_cache is not None and isinstance(cross_attn_cache, (list, tuple)) and len(cross_attn_cache) == 2:
                    cross_key, cross_value = cross_attn_cache
                    active_cross_key = mx.take(cross_key, active_indices, axis=0)
                    active_cross_value = mx.take(cross_value, active_indices, axis=0)
                    active_cross_attn = (active_cross_key, active_cross_value)
                else:
                    active_cross_attn = cross_attn_cache
                    
                active_kv_cache.append((active_self_attn, active_cross_attn))
            else:
                active_kv_cache.append(None)
            
        return active_kv_cache
        
    def _update_kv_cache(self, new_kv_cache: List, active_indices: mx.array):
        """Update KV cache for active sequences only."""
        indices_list = active_indices.tolist()
        
        for i, new_block_cache in enumerate(new_kv_cache):
            if new_block_cache is None:
                continue
            
            old_block_cache = self.kv_cache[i]
            if old_block_cache is None:
                continue
                
            # Block cache is (self_attn_cache, cross_attn_cache)
            if isinstance(new_block_cache, (list, tuple)) and len(new_block_cache) == 2:
                new_self_attn, new_cross_attn = new_block_cache
                old_self_attn, old_cross_attn = old_block_cache
                
                # Update self-attention cache
                if new_self_attn is not None and old_self_attn is not None:
                    new_key, new_value = new_self_attn
                    old_key, old_value = old_self_attn
                    
                    # The decoder returns the FULL updated cache
                    # Since shapes don't match, we need to handle variable-length caches
                    # For now, let's pad all caches to the same length
                    max_len = max(old_key.shape[1], new_key.shape[1])
                    
                    # Create padded arrays
                    padded_key = mx.zeros((self.batch_size, max_len, old_key.shape[-1]), dtype=old_key.dtype)
                    padded_value = mx.zeros((self.batch_size, max_len, old_value.shape[-1]), dtype=old_value.dtype)
                    
                    # Copy existing caches
                    for k in range(self.batch_size):
                        if k in indices_list:
                            # Active sequence - use new cache
                            j = indices_list.index(k)
                            seq_len = new_key[j].shape[0]
                            padded_key[k, :seq_len] = new_key[j]
                            padded_value[k, :seq_len] = new_value[j]
                        else:
                            # Inactive sequence - keep old cache
                            seq_len = old_key[k].shape[0]
                            padded_key[k, :seq_len] = old_key[k]
                            padded_value[k, :seq_len] = old_value[k]
                    
                    # Replace the cache
                    self.kv_cache[i] = ((padded_key, padded_value), old_cross_attn)
                
                # Cross-attention cache doesn't change after initialization
                # It only depends on audio features which are fixed
                # So we don't need to update it


class BatchGreedyDecoder(GreedyDecoder):
    """
    Fixed version of GreedyDecoder that properly handles batches.
    """
    
    def update(self, tokens: mx.array, logits: mx.array, sum_logprobs: mx.array,
               active_mask: Optional[mx.array] = None) -> Tuple[mx.array, mx.array, mx.array]:
        """
        Update tokens with proper batch handling.
        
        Returns:
            tokens: Updated token sequences
            completed_mask: Boolean mask of completed sequences
            sum_logprobs: Updated sum of log probabilities
        """
        if self.temperature == 0:
            next_tokens = logits.argmax(axis=-1)
        else:
            next_tokens = mx.random.categorical(logits / self.temperature)
            
        # Fix: Add keepdims=True
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        
        # Get current logprobs for selected tokens
        batch_indices = mx.arange(logprobs.shape[0])
        current_logprobs = logprobs[batch_indices, next_tokens]
        
        # Update sum_logprobs only for non-EOT tokens
        not_eot_mask = tokens[:, -1] != self.eot
        sum_logprobs = sum_logprobs + current_logprobs * not_eot_mask
        
        # For sequences that already hit EOT, keep outputting EOT
        eot_mask = tokens[:, -1] == self.eot
        next_tokens = mx.where(eot_mask, self.eot, next_tokens)
        
        # Append next tokens
        tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=-1)
        
        # Mark sequences as completed if they output EOT
        completed_mask = tokens[:, -1] == self.eot
        
        return tokens, completed_mask, sum_logprobs


class BatchDecodingTask(DecodingTask):
    """
    Batch-aware decoding task that properly handles multiple sequences.
    """
    
    def __init__(self, model: "Whisper", options: DecodingOptions):
        super().__init__(model, options)
        # Replace with batch-aware versions
        self.inference = None  # Will be created per batch
        self.decoder = BatchGreedyDecoder(options.temperature, self.tokenizer.eot)
        
    def _main_loop_batch(self, audio_features: mx.array, tokens: mx.array):
        """
        Main decoding loop with proper batch handling.
        
        Key improvements:
        1. Tracks completed sequences individually
        2. Only processes active sequences
        3. Properly manages KV cache per sequence
        4. Early stops completed sequences
        """
        batch_size = tokens.shape[0]
        sum_logprobs = mx.zeros(batch_size)
        completed_mask = mx.zeros(batch_size, dtype=mx.bool_)
        
        # Create batch inference
        self.inference = BatchInference(self.model, batch_size)
        
        # First token generation (process all initial tokens)
        logits = self.inference.logits(tokens, audio_features)
        
        # Apply logit filters
        for logit_filter in self.logit_filters:
            logits = logit_filter.apply(logits, tokens)
            
        # Update tokens
        tokens, completed_mask, sum_logprobs = self.decoder.update(
            tokens, logits, sum_logprobs
        )
        
        # Calculate no_speech_probs if needed
        if self.tokenizer.no_speech is not None:
            # Use the logits we already computed
            probs_at_sot = mx.softmax(logits, axis=-1)
            no_speech_probs = probs_at_sot[:, self.tokenizer.no_speech]
        else:
            no_speech_probs = mx.full(batch_size, mx.nan)
            
        # Main generation loop
        for i in range(1, self.sample_len):
            # Check if all sequences are completed
            if mx.all(completed_mask):
                break
                
            # Only process active (non-completed) sequences
            active_mask = ~completed_mask
            
            # Get last tokens for active sequences
            last_tokens = tokens[:, -1:]
            
            # Check context length
            if tokens.shape[-1] > self.n_ctx:
                break
                
            # Get logits for active sequences
            logits = self.inference.logits(
                last_tokens, audio_features, sequence_mask=active_mask
            )
            
            # Apply logit filters
            for logit_filter in self.logit_filters:
                logits = logit_filter.apply(logits, tokens)
                
            # Update tokens
            tokens, completed_mask, sum_logprobs = self.decoder.update(
                tokens, logits, sum_logprobs, active_mask
            )
            
        return tokens, sum_logprobs, no_speech_probs
        
    def run(self, mel: mx.array) -> List[DecodingResult]:
        """
        Run batch decoding on mel spectrograms.
        
        Args:
            mel: (batch_size, n_mels, n_frames) or (n_mels, n_frames)
            
        Returns:
            List of DecodingResult objects
        """
        single = mel.ndim == 2
        if single:
            mel = mel[None]
            
        n_audio = mel.shape[0]
        
        # Get audio features
        audio_features = self._get_audio_features(mel)
        
        # Initialize tokens
        tokens = mx.array(self.initial_tokens)
        tokens = mx.broadcast_to(tokens, (n_audio, len(self.initial_tokens)))
        
        # Detect language if needed
        languages, language_probs = self._detect_language(audio_features, tokens)
        
        if self.options.task == "lang_id":
            return [
                DecodingResult(
                    audio_features=features, 
                    language=language, 
                    language_probs=probs
                )
                for features, language, probs in zip(
                    audio_features, languages, language_probs
                )
            ]
            
        # Main decoding loop
        tokens, sum_logprobs, no_speech_probs = self._main_loop_batch(
            audio_features, tokens
        )
        
        # Process results
        tokens = tokens[:, self.sample_begin:]
        
        # Convert to lists
        mx.eval(tokens, sum_logprobs, no_speech_probs)
        tokens_list = tokens.tolist()
        sum_logprobs_list = sum_logprobs.tolist()
        no_speech_probs_list = no_speech_probs.tolist()
        
        # Trim tokens at EOT
        for i in range(n_audio):
            if self.tokenizer.eot in tokens_list[i]:
                tokens_list[i] = tokens_list[i][:tokens_list[i].index(self.tokenizer.eot)]
                
        # Decode to text
        texts = [self.tokenizer.decode(t).strip() for t in tokens_list]
        
        # Calculate average logprobs
        avg_logprobs = [
            lp / (len(t) + 1) for t, lp in zip(tokens_list, sum_logprobs_list)
        ]
        
        # Create results
        results = [
            DecodingResult(
                audio_features=features,
                language=language,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=self.options.temperature,
                compression_ratio=self._calculate_compression_ratio(text),
            )
            for features, language, tokens, text, avg_logprob, no_speech_prob in zip(
                audio_features, languages, tokens_list, texts, avg_logprobs, no_speech_probs_list
            )
        ]
        
        return results[0] if single else results
        
    def _calculate_compression_ratio(self, text: str) -> float:
        """Calculate compression ratio for text."""
        import zlib
        text_bytes = text.encode("utf-8")
        try:
            return len(text_bytes) / len(zlib.compress(text_bytes))
        except:
            return 1.0


def batch_decode(
    model: "Whisper",
    mel: mx.array,
    options: DecodingOptions = DecodingOptions(),
) -> List[DecodingResult]:
    """
    Decode audio with proper batch support.
    
    This is the main entry point for batch decoding.
    """
    task = BatchDecodingTask(model, options)
    return task.run(mel)