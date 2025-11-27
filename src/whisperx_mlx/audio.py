"""
Audio loading and processing utilities for WhisperX-MLX.

This module provides functions for loading audio files and computing
mel spectrograms compatible with Whisper models.
"""

import os
import subprocess
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F


# Audio hyperparameters (matching Whisper's requirements)
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions have stride 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load an audio file and convert to mono waveform at the target sample rate.

    Uses FFmpeg for robust audio decoding and resampling.

    Args:
        file: Path to the audio file
        sr: Target sample rate (default: 16000 Hz for Whisper)

    Returns:
        NumPy array of audio samples as float32, normalized to [-1, 1]

    Raises:
        RuntimeError: If FFmpeg fails to load the audio
        FileNotFoundError: If the file doesn't exist
    """
    if not os.path.exists(file):
        raise FileNotFoundError(f"Audio file not found: {file}")

    try:
        # Use FFmpeg to decode, down-mix to mono, and resample
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads", "0",
            "-i", file,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg to load audio files.\n"
            "On macOS: brew install ffmpeg\n"
            "On Ubuntu: sudo apt install ffmpeg"
        )

    # Convert to float32 and normalize
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """Pad or trim the audio array to the specified length.

    Args:
        array: NumPy array or PyTorch tensor
        length: Target length (default: N_SAMPLES for 30 seconds)
        axis: Axis along which to pad/trim

    Returns:
        Array with the exact specified length
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.

    Args:
        device: PyTorch device
        n_mels: Number of mel bands (80 or 128)

    Returns:
        Mel filterbank matrix as a PyTorch tensor
    """
    assert n_mels in [80, 128], f"Unsupported n_mels: {n_mels}"

    # Try to load from assets
    assets_dir = os.path.join(os.path.dirname(__file__), "assets")
    mel_file = os.path.join(assets_dir, "mel_filters.npz")

    if os.path.exists(mel_file):
        with np.load(mel_file) as f:
            return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)

    # Generate mel filters using librosa if available
    try:
        import librosa
        filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)
        return torch.from_numpy(filters).to(device)
    except ImportError:
        raise RuntimeError(
            "Mel filter file not found and librosa not available. "
            "Either copy mel_filters.npz to the assets directory or install librosa."
        )


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int = 80,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Compute the log-Mel spectrogram of an audio signal.

    Args:
        audio: Path to audio file, NumPy array, or PyTorch tensor (16kHz)
        n_mels: Number of mel bands (default: 80)
        padding: Number of zero samples to pad to the right
        device: Device to move the tensor to before STFT

    Returns:
        Log-mel spectrogram tensor of shape (n_mels, n_frames)
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))

    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    return log_spec


def get_audio_duration(audio: Union[str, np.ndarray]) -> float:
    """Get the duration of an audio file or array in seconds.

    Args:
        audio: Path to audio file or NumPy array of samples

    Returns:
        Duration in seconds
    """
    if isinstance(audio, str):
        audio = load_audio(audio)
    return len(audio) / SAMPLE_RATE
