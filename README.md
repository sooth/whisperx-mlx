# WhisperX-MLX

WhisperX with an MLX backend for Apple Silicon: transcription, word-level alignment, and speaker diarization.

The **default** MLX path (`large-v3`, no extra flags) is the fast path. It batches VAD chunks through the encoder and decoder, uses an 8-bit decoder with an fp16 encoder, and keeps greedy token identity versus the previous fp16 large-v3 decode.

## Features

- GPU transcription on Apple Silicon via MLX (`mlx-community/whisper-large-v3-mlx` by default)
- Automatic backend: MLX on Apple Silicon, faster-whisper elsewhere
- Default batch size 16 for VAD chunks (CLI `--batch-size` and `transcribe(batch_size=...)`)
- 8-bit decoder (`bits=8`, `group_size=64`); encoder stays fp16
- Encoder self-attention via `mx.fast.scaled_dot_product_attention` (dual-scale qk, same greedy tokens)
- GPU timestamp logit filter, preallocated decoder KV, Metal cache retain + GEMM preheat at model load
- Word-level timestamps (`--align`) and speaker diarization (`--diarize`)
- Silero VAD on the default transcribe path

## Performance

Measured on an M4 Max, `short.wav` (~90s), default `large-v3` MLX, Silero VAD, English, greedy temp-0:

| Path | Wall time | Notes |
|---|---|---|
| Previous default MLX (fp16 decoder, batch 16) | 3.34s median | pre-change baseline |
| Current default MLX | **2.97–2.99s** overall | two 90s-idle sessions; every pair under 3.17s |
| Sequential per-VAD-chunk decode | 6.17s median | same machine, same audio |

Normalized WER versus the previous default transcript is **0.0** on that clip. Spoken markers `famous` / `gordon` / `ramsay` still appear. Alignment and diarization are opt-in and are not in these timings.

## Installation

```bash
# Apple Silicon (recommended)
uv tool install --python 3.12 "whisperx-mlx[mlx]"

# or
pip install "whisperx-mlx[mlx]"

# CUDA/CPU fallback
pip install "whisperx-mlx[faster-whisper]"
```

Needs **ffmpeg** on `PATH`.

## Quick Start

### Command Line

```bash
# Default: large-v3, MLX on Apple Silicon, Silero VAD
whisperx-mlx audio.mp3 --model large-v3 --backend mlx --language en

# Word alignment
whisperx-mlx audio.mp3 --model large-v3 --align

# Speaker diarization
whisperx-mlx audio.mp3 --model large-v3 --diarize --hf-token YOUR_TOKEN
```

There is no `--fast` flag. The defaults above are the optimized path.

### Python API

```python
from whisperx_mlx import transcribe

result = transcribe("audio.mp3", model="large-v3", backend="mlx", language="en")

for segment in result["segments"]:
    print(f"{segment['start']:.2f} - {segment['end']:.2f}: {segment['text']}")
```

## Models

| Name | Hugging Face repo |
|---|---|
| `large-v3` (default), `large` | `mlx-community/whisper-large-v3-mlx` |
| `tiny` / `tiny.en`, `base` / `base.en`, `small` / `small.en`, `medium` / `medium.en` | `mlx-community/whisper-<name>-mlx` |
| `large-v2` | `mlx-community/whisper-large-v2-mlx` |
| `turbo`, `large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` |
| `distil-large-v3` | `mlx-community/distil-whisper-large-v3` |

A full `org/repo` path is accepted as `--model`. Switching the default to turbo/distil is not done here: those checkpoints change tokens versus large-v3.

## Requirements

- Python 3.10+
- FFmpeg
- MLX backend: macOS on Apple Silicon (M1–M4)
- Diarization: Hugging Face token

## License

BSD-2-Clause (same as WhisperX)
