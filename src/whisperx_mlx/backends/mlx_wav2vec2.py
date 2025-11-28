"""
MLX implementation of Wav2Vec2 for forced alignment.

This is a port of the torchaudio Wav2Vec2Model to Apple's MLX framework
for GPU/ANE acceleration on Apple Silicon.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

import mlx.core as mx
import mlx.nn as nn


@dataclass
class Wav2Vec2Config:
    """Configuration for Wav2Vec2 model."""
    # Feature extractor
    conv_layers: List[Tuple[int, int, int]] = None  # (out_channels, kernel, stride)
    conv_bias: bool = False

    # Transformer encoder
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout: float = 0.1
    attention_dropout: float = 0.1

    # Positional embedding
    num_conv_pos_embeddings: int = 128
    num_conv_pos_embedding_groups: int = 16

    # CTC head
    vocab_size: int = 29

    def __post_init__(self):
        if self.conv_layers is None:
            # Default WAV2VEC2_ASR_BASE_960H config
            self.conv_layers = [
                (512, 10, 5),  # Layer 0: in=1, out=512, k=10, s=5
                (512, 3, 2),   # Layer 1-4: k=3, s=2
                (512, 3, 2),
                (512, 3, 2),
                (512, 3, 2),
                (512, 2, 2),   # Layer 5-6: k=2, s=2
                (512, 2, 2),
            ]


class ConvLayerBlock(nn.Module):
    """Convolutional layer block for feature extraction."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        bias: bool = False,
        use_group_norm: bool = False,
    ):
        super().__init__()
        self.use_group_norm = use_group_norm

        if use_group_norm:
            # GroupNorm before conv (like layer 0)
            self.layer_norm = nn.GroupNorm(
                num_groups=out_channels,
                dims=out_channels,
                pytorch_compatible=True,
            )

        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, channels) - MLX uses channels-last
        x = self.conv(x)
        if self.use_group_norm:
            x = self.layer_norm(x)
        x = nn.gelu(x)
        return x


class FeatureExtractor(nn.Module):
    """CNN feature extractor that converts raw audio to feature sequences."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.conv_layers = []

        in_channels = 1
        for i, (out_channels, kernel_size, stride) in enumerate(config.conv_layers):
            # Only first layer has GroupNorm
            use_group_norm = (i == 0)
            layer = ConvLayerBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                bias=config.conv_bias,
                use_group_norm=use_group_norm,
            )
            self.conv_layers.append(layer)
            in_channels = out_channels

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time) raw audio
        # Add channel dimension: (batch, time, 1)
        x = mx.expand_dims(x, axis=-1)

        for layer in self.conv_layers:
            x = layer(x)

        return x  # (batch, time', 512)


class FeatureProjection(nn.Module):
    """Projects CNN features to transformer hidden size."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        conv_out_dim = config.conv_layers[-1][0]  # 512

        self.layer_norm = nn.LayerNorm(conv_out_dim)
        self.projection = nn.Linear(conv_out_dim, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class ConvPositionalEmbedding(nn.Module):
    """Convolutional positional embedding (replaces sinusoidal)."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        # Uses grouped convolution with groups=16
        self.num_groups = config.num_conv_pos_embedding_groups  # 16
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            stride=1,
            padding=config.num_conv_pos_embeddings // 2,
            groups=self.num_groups,
        )
        # Note: Weight norm is pre-applied during weight conversion
        # so we don't need to implement it for inference

    def __call__(self, x: mx.array) -> mx.array:
        # x: (batch, time, hidden)
        pos_embed = self.conv(x)
        # Trim to match input length (due to padding)
        pos_embed = pos_embed[:, :x.shape[1], :]
        return nn.gelu(pos_embed)


class SelfAttention(nn.Module):
    """Multi-head self attention."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.attention_dropout)

    def __call__(self, x: mx.array) -> mx.array:
        batch, seq_len, _ = x.shape

        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        # (batch, seq, heads, head_dim) -> (batch, heads, seq, head_dim)
        q = q.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Attention scores
        attn_weights = (q @ k.transpose(0, 1, 3, 2)) * self.scale
        attn_weights = mx.softmax(attn_weights, axis=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = attn_weights @ v

        # Reshape back: (batch, heads, seq, head_dim) -> (batch, seq, hidden)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch, seq_len, -1)

        return self.out_proj(attn_output)


class FeedForward(nn.Module):
    """Feed-forward network in transformer."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_dropout = nn.Dropout(0.0)  # torchaudio uses 0.0 here
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.intermediate_dense(x)
        x = nn.gelu(x)
        x = self.intermediate_dropout(x)
        x = self.output_dense(x)
        x = self.output_dropout(x)
        return x


class EncoderLayer(nn.Module):
    """Single transformer encoder layer."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.feed_forward = FeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size)

    def __call__(self, x: mx.array) -> mx.array:
        # Self-attention with residual
        residual = x
        x = self.attention(x)
        x = self.dropout(x)
        x = residual + x
        x = self.layer_norm(x)

        # Feed-forward with residual
        residual = x
        x = self.feed_forward(x)
        x = residual + x
        x = self.final_layer_norm(x)

        return x


class Transformer(nn.Module):
    """Transformer encoder stack."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.pos_conv_embed = ConvPositionalEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layers = [EncoderLayer(config) for _ in range(config.num_hidden_layers)]

    def __call__(self, x: mx.array) -> mx.array:
        # Add positional embeddings
        pos_embed = self.pos_conv_embed(x)
        x = x + pos_embed
        x = self.layer_norm(x)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        return x


class Encoder(nn.Module):
    """Full encoder: feature projection + transformer."""

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.feature_projection = FeatureProjection(config)
        self.transformer = Transformer(config)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.feature_projection(x)
        x = self.transformer(x)
        return x


class Wav2Vec2Model(nn.Module):
    """
    MLX Wav2Vec2 model for CTC-based forced alignment.

    Architecture:
    - FeatureExtractor: 7 Conv1d layers (downsamples 320x)
    - Encoder: Feature projection + 12-layer Transformer
    - aux (CTC head): Linear projection to vocabulary
    """

    def __init__(self, config: Optional[Wav2Vec2Config] = None):
        super().__init__()
        if config is None:
            config = Wav2Vec2Config()
        self.config = config

        self.feature_extractor = FeatureExtractor(config)
        self.encoder = Encoder(config)
        self.aux = nn.Linear(config.hidden_size, config.vocab_size)

    def __call__(self, x: mx.array) -> mx.array:
        """
        Forward pass.

        Args:
            x: Raw audio waveform (batch, time) at 16kHz

        Returns:
            Log probabilities (batch, time', vocab_size)
        """
        # Feature extraction
        x = self.feature_extractor(x)

        # Encoder
        x = self.encoder(x)

        # CTC head
        logits = self.aux(x)

        # Log softmax for CTC
        log_probs = mx.log(mx.softmax(logits, axis=-1) + 1e-10)

        return log_probs


def convert_torch_weights_to_mlx(torch_model) -> dict:
    """
    Convert PyTorch Wav2Vec2 weights to MLX format.

    Handles:
    - Conv1d weight transpose: PyTorch (out, in, k) -> MLX (out, k, in)
    - Weight norm de-parametrization for positional conv
    - Key name mapping
    """
    import numpy as np

    mlx_weights = {}

    # Get the positional conv weight directly (already de-parametrized via property)
    pos_conv_weight = torch_model.encoder.transformer.pos_conv_embed.conv.weight.detach().cpu().numpy()
    pos_conv_bias = torch_model.encoder.transformer.pos_conv_embed.conv.bias.detach().cpu().numpy()

    # Transpose: PyTorch (out=768, in/groups=48, k=128) -> MLX (out=768, k=128, in/groups=48)
    pos_conv_weight = pos_conv_weight.transpose(0, 2, 1)
    mlx_weights["encoder.transformer.pos_conv_embed.conv.weight"] = mx.array(pos_conv_weight)
    mlx_weights["encoder.transformer.pos_conv_embed.conv.bias"] = mx.array(pos_conv_bias)

    # Get state dict for other weights
    torch_state = torch_model.state_dict()

    for key, value in torch_state.items():
        # Skip the parametrization keys - we handled pos_conv above
        if "parametrizations" in key:
            continue
        # Skip pos_conv bias since we already added it
        if "pos_conv_embed.conv.bias" in key:
            continue

        # Convert tensor to numpy
        np_value = value.detach().cpu().numpy()

        # Handle Conv1d weight transpose (PyTorch: out, in, k -> MLX: out, k, in)
        if "conv" in key and "weight" in key and len(np_value.shape) == 3:
            np_value = np_value.transpose(0, 2, 1)

        mlx_weights[key] = mx.array(np_value)

    return mlx_weights


def load_wav2vec2_from_torchaudio(
    model_name: str = "WAV2VEC2_ASR_BASE_960H",
    cache_dir: Optional[str] = None,
) -> Tuple[Wav2Vec2Model, dict]:
    """
    Load a Wav2Vec2 model by converting from torchaudio.

    Args:
        model_name: Name of the torchaudio pipeline
        cache_dir: Directory to cache converted weights

    Returns:
        Tuple of (MLX model, metadata dict with labels)
    """
    import os
    import torchaudio

    # Check for cached MLX weights
    if cache_dir is None:
        cache_dir = os.path.expanduser("~/.cache/whisperx_mlx/wav2vec2")
    os.makedirs(cache_dir, exist_ok=True)

    cache_path = os.path.join(cache_dir, f"{model_name}.safetensors")
    labels_path = os.path.join(cache_dir, f"{model_name}_labels.txt")

    # Get bundle info
    bundle = getattr(torchaudio.pipelines, model_name)
    labels = bundle.get_labels()

    # Create MLX model
    config = Wav2Vec2Config(vocab_size=len(labels))
    model = Wav2Vec2Model(config)

    # Load or convert weights
    if os.path.exists(cache_path):
        # Load cached weights
        weights = mx.load(cache_path)
    else:
        # Convert from PyTorch
        print(f"Converting {model_name} weights to MLX format...")
        torch_model = bundle.get_model()
        weights = convert_torch_weights_to_mlx(torch_model)

        # Save for next time
        mx.save_safetensors(cache_path, weights)
        with open(labels_path, "w") as f:
            f.write("\n".join(labels))
        print(f"Cached MLX weights to {cache_path}")

    # Load weights into model
    model.load_weights(list(weights.items()))

    metadata = {
        "labels": labels,
        "sample_rate": bundle.sample_rate,
    }

    return model, metadata
