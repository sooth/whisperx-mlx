"""WER-neutral MLX decode speedups for the default Whisper path.

Monkeypatches mlx-whisper's greedy decoder, timestamp logit filter,
decoder KV cache, and encoder self-attention. Token identity is
preserved: argmax is unchanged, the timestamp-vs-text gate is
algebraically identical, preallocated KV prefixes match concat, and
encoder SDPA uses the same dual-scale qk as mlx-whisper.
"""

from __future__ import annotations

_APPLIED = False


def _tune_mlx_memory() -> None:
    """Retain compiled Metal buffers. Without this, post-warmup runs were ~4s."""
    import mlx.core as mx

    eight_gib = 8 * 1024 * 1024 * 1024
    for fn in (
        getattr(mx, "set_cache_limit", None),
        getattr(getattr(mx, "metal", None), "set_cache_limit", None),
    ):
        if fn is None:
            continue
        try:
            fn(eight_gib)
            break
        except Exception:
            continue


def apply_mlx_decode_speedups() -> None:
    """Patch mlx-whisper decode helpers. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return

    import mlx.core as mx
    from mlx_whisper.decoding import ApplyTimestampRules, GreedyDecoder

    _tune_mlx_memory()

    _NEG = mx.array(-mx.inf, dtype=mx.float32)
    _vocab_cache: dict = {}
    _ts_hot = {}

    def _vocab_and_nots(n_vocab: int, no_timestamps):
        key = (n_vocab, no_timestamps)
        cached = _vocab_cache.get(key)
        if cached is not None:
            return cached
        vocab = mx.arange(n_vocab)
        nt = mx.zeros((n_vocab,), dtype=mx.float32)
        if no_timestamps is not None:
            nt = mx.where(vocab == no_timestamps, _NEG, nt)
        _vocab_cache[key] = (vocab, nt)
        return vocab, nt

    def _hot_filter(n_vocab: int, ts_begin: int, eot: int):
        key = (n_vocab, ts_begin, eot)
        fn = _ts_hot.get(key)
        if fn is not None:
            return fn
        vocab = mx.arange(n_vocab)

        def core(logits, last_was_ts, penult_was_ts, nt):
            mask = mx.zeros(logits.shape, dtype=mx.float32) + nt
            force_text = mx.logical_and(last_was_ts, penult_was_ts)
            force_ts = mx.logical_and(last_was_ts, mx.logical_not(penult_was_ts))
            mask = mask + mx.where(
                mx.logical_and(force_text[:, None], vocab[None, :] >= ts_begin),
                _NEG,
                0.0,
            )
            mask = mask + mx.where(
                mx.logical_and(force_ts[:, None], vocab[None, :] < eot),
                _NEG,
                0.0,
            )
            timestamp_score = logits[:, ts_begin:].logsumexp(axis=-1, keepdims=True)
            max_text = logits[:, :ts_begin].max(axis=-1, keepdims=True)
            mask_text = mx.where(timestamp_score > max_text, _NEG, mask[:, :ts_begin])
            mask = mx.concatenate([mask_text, mask[:, ts_begin:]], axis=-1)
            return logits + mask

        fn = mx.compile(core)
        _ts_hot[key] = fn
        return fn

    def timestamp_apply(self, logits, tokens):
        """Same token decisions as mlx-whisper ApplyTimestampRules, no host sync."""
        ts_begin = self.tokenizer.timestamp_begin
        eot = self.tokenizer.eot
        n_batch, n_vocab = logits.shape
        tlen = tokens.shape[1]
        vocab, nt = _vocab_and_nots(n_vocab, self.tokenizer.no_timestamps)
        sampled = tlen - self.sample_begin

        if tlen == self.sample_begin:
            mask = mx.zeros((n_batch, n_vocab), dtype=mx.float32) + nt
            mask = mask + mx.where(vocab[None, :] < ts_begin, _NEG, 0.0)
            if self.max_initial_timestamp_index is not None:
                last_allowed = ts_begin + self.max_initial_timestamp_index
                mask = mask + mx.where(vocab[None, :] > last_allowed, _NEG, 0.0)
            timestamp_score = logits[:, ts_begin:].logsumexp(axis=-1, keepdims=True)
            max_text = logits[:, :ts_begin].max(axis=-1, keepdims=True)
            mask_text = mx.where(timestamp_score > max_text, _NEG, mask[:, :ts_begin])
            mask = mx.concatenate([mask_text, mask[:, ts_begin:]], axis=-1)
            return logits + mask

        last_was_ts = tokens[:, -1] >= ts_begin
        if sampled >= 2:
            penult_was_ts = tokens[:, -2] >= ts_begin
        else:
            penult_was_ts = mx.ones((n_batch,), dtype=mx.bool_)
        return _hot_filter(n_vocab, ts_begin, eot)(logits, last_was_ts, penult_was_ts, nt)

    def greedy_update(self, tokens, logits, sum_logprobs):
        if self.temperature == 0:
            next_tokens = logits.argmax(axis=-1)
        else:
            from mlx_whisper.decoding import categorical

            next_tokens = categorical(logits, self.temperature)
        eot_mask = tokens[:, -1] == self.eot
        next_tokens = next_tokens * (1 - eot_mask) + self.eot * eot_mask
        tokens = mx.concatenate([tokens, next_tokens[:, None]], axis=-1)
        completed = mx.all(tokens[:, -1] == self.eot)
        return tokens, completed, sum_logprobs

    ApplyTimestampRules.apply = timestamp_apply
    GreedyDecoder.update = greedy_update
    _patch_preallocated_kv()
    _patch_encoder_sdpa()
    _APPLIED = True


class _PreallocKV:
    """Fixed-size self-attn cache. Prefix `[:, :offset]` matches concat."""

    def __init__(self, n_layers: int, batch: int, max_len: int, n_state, dtype):
        import mlx.core as mx

        self.offset = 0
        self.self_k = [mx.zeros((batch, max_len, n_state), dtype=dtype) for _ in range(n_layers)]
        self.self_v = [mx.zeros((batch, max_len, n_state), dtype=dtype) for _ in range(n_layers)]
        self.cross = [None] * n_layers

    def write_self(self, layer: int, k, v):
        t = k.shape[1]
        off = self.offset
        self.self_k[layer][:, off : off + t, :] = k
        self.self_v[layer][:, off : off + t, :] = v
        return self.self_k[layer][:, : off + t, :], self.self_v[layer][:, : off + t, :]


def _patch_preallocated_kv() -> None:
    """Replace concat KV with indexed writes. Same prefixes → same attention."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_whisper.decoding import Inference

    orig_reset = Inference.reset

    def reset(self):
        orig_reset(self)
        store = getattr(self, "_prealloc", None)
        if store is not None:
            store.offset = 0
            store.cross = [None] * len(store.cross)

    def logits(self, tokens, audio_features):
        decoder = self.model.decoder
        batch, t_now = int(tokens.shape[0]), int(tokens.shape[1])
        n_state = int(self.model.dims.n_text_state)
        max_len = int(self.model.dims.n_text_ctx)
        n_layers = int(self.model.dims.n_text_layer)
        dtype = decoder.positional_embedding.dtype

        store = getattr(self, "_prealloc", None)
        if store is None or store.self_k[0].shape[0] != batch:
            store = _PreallocKV(n_layers, batch, max_len, n_state, dtype)
            mx.eval(store.self_k, store.self_v)
            self._prealloc = store

        offset = store.offset
        if offset + t_now > max_len:
            raise RuntimeError("preallocated KV cache overflow")

        x = decoder.token_embedding(tokens) + decoder.positional_embedding[offset : offset + t_now]
        mask = decoder._mask
        for i, block in enumerate(decoder.blocks):
            h = block.attn_ln(x)
            q = block.attn.query(h)
            k = block.attn.key(h)
            v = block.attn.value(h)
            k_all, v_all = store.write_self(i, k, v)
            wv, _ = block.attn.qkv_attention(q, k_all, v_all, mask)
            x = x + block.attn.out(wv)

            h = block.cross_attn_ln(x)
            q = block.cross_attn.query(h)
            if store.cross[i] is None:
                ck = block.cross_attn.key(audio_features)
                cv = block.cross_attn.value(audio_features)
                store.cross[i] = (ck, cv)
            else:
                ck, cv = store.cross[i]
            wv, _ = block.cross_attn.qkv_attention(q, ck, cv, mask=None)
            x = x + block.cross_attn.out(wv)
            x = x + block.mlp2(nn.gelu(block.mlp1(block.mlp_ln(x))))

        x = decoder.ln(x)
        out = decoder.token_embedding.as_linear(x).astype(mx.float32)
        store.offset = offset + t_now
        self.kv_cache = store
        return out

    Inference.reset = reset
    Inference.logits = logits


def _patch_encoder_sdpa() -> None:
    """Encoder self-attn via SDPA. Dual-scale matches mlx-whisper qk (WER 0).

    Decoder stays on the original matmul path (Tq < 32 or causal mask).
    """
    import mlx.core as mx
    from mlx_whisper.whisper import MultiHeadAttention

    orig = MultiHeadAttention.qkv_attention

    def qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        if mask is not None or n_ctx < 32:
            return orig(self, q, k, v, mask)
        n_head = self.n_head
        head_dim = n_state // n_head
        scale = head_dim ** -0.25
        qh = q.reshape(n_batch, n_ctx, n_head, head_dim).transpose(0, 2, 1, 3) * scale
        kh = k.reshape(k.shape[0], k.shape[1], n_head, head_dim).transpose(0, 2, 1, 3) * scale
        vh = v.reshape(v.shape[0], v.shape[1], n_head, head_dim).transpose(0, 2, 1, 3)
        out = mx.fast.scaled_dot_product_attention(qh, kh, vh, scale=1.0)
        out = out.transpose(0, 2, 1, 3).reshape(n_batch, n_ctx, n_state)
        return out, None

    MultiHeadAttention.qkv_attention = qkv_attention


def preheat_metal() -> None:
    """Raise GPU clocks after idle. Cheap GEMMs; does not change ASR tokens."""
    import mlx.core as mx

    x = mx.ones((2048, 2048), dtype=mx.float16)
    for _ in range(12):
        x = mx.matmul(x, x)
    mx.eval(x)
