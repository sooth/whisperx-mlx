"""Regression test for the torchcodec-free diarization audio loader.

Guards against the silent failure seen 2026-09-10, where pyannote could not
decode a file path (torchcodec/FFmpeg mismatch) and transcripts came out
with a single UNKNOWN speaker. load_waveform() must always produce the
in-memory dict pyannote 4.x accepts, regardless of input format.

Run with pytest, or directly:  python tests/test_load_waveform.py
"""
import tempfile

import numpy as np
import torch
from scipy.io import wavfile

from whisperx_mlx.diarization.pyannote_backend import PYANNOTE_SAMPLE_RATE, load_waveform


def _write_wav(sr, data):
    path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    wavfile.write(path, sr, data)
    return path


def _check(w, expected_seconds):
    assert set(w) == {"waveform", "sample_rate"}
    assert w["sample_rate"] == PYANNOTE_SAMPLE_RATE
    assert isinstance(w["waveform"], torch.Tensor)
    assert w["waveform"].dtype == torch.float32
    assert w["waveform"].ndim == 2 and w["waveform"].shape[0] == 1, "expected (1, time) mono"
    n = w["waveform"].shape[1]
    assert abs(n - expected_seconds * PYANNOTE_SAMPLE_RATE) <= 2
    assert float(w["waveform"].abs().max()) <= 1.0, "PCM must be normalised to [-1, 1]"


def test_stereo_44k_int16_becomes_mono_16k():
    sr, secs = 44100, 2
    t = np.arange(sr * secs) / sr
    x = (np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], 1) * 32000).astype(np.int16)
    _check(load_waveform(_write_wav(sr, x)), secs)


def test_mono_16k_int16_passthrough():
    sr, secs = 16000, 3
    x = (np.sin(2 * np.pi * 300 * np.arange(sr * secs) / sr) * 20000).astype(np.int16)
    w = load_waveform(_write_wav(sr, x))
    _check(w, secs)
    assert w["waveform"].shape[1] == sr * secs  # no resampling, exact length


def test_float32_wav():
    sr, secs = 16000, 1
    x = (np.sin(2 * np.pi * 200 * np.arange(sr * secs) / sr) * 0.5).astype(np.float32)
    _check(load_waveform(_write_wav(sr, x)), secs)


def test_ndarray_input_is_treated_as_16k_mono():
    _check(load_waveform(np.zeros(16000 * 2, dtype=np.float32)), 2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
