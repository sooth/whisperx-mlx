# WhisperX-MLX

WhisperX with MLX acceleration for Apple Silicon - speech recognition with word-level alignment and speaker diarization.

## Features

- GPU-accelerated transcription on Apple Silicon using MLX
- Automatic backend selection (MLX on Apple Silicon, faster-whisper elsewhere)
- Word-level timestamps via forced alignment
- Speaker diarization (who said what)
- Voice Activity Detection preprocessing

## Installation

```bash
# For Apple Silicon (MLX backend)
pip install whisperx-mlx[mlx]

# For CUDA/CPU (faster-whisper backend)
pip install whisperx-mlx[faster-whisper]

# For all backends
pip install whisperx-mlx[all]
```

## Quick Start

### Command Line

```bash
# Basic transcription
whisperx-mlx audio.mp3 --model large-v3

# With word alignment
whisperx-mlx audio.mp3 --model large-v3 --align

# With speaker diarization
whisperx-mlx audio.mp3 --model large-v3 --diarize --hf-token YOUR_TOKEN

# Force specific backend
whisperx-mlx audio.mp3 --model large-v3 --backend mlx
```

### Python API

```python
from whisperx_mlx import transcribe

# Basic transcription
result = transcribe("audio.mp3", model="large-v3")

for segment in result["segments"]:
    print(f"{segment['start']:.2f} - {segment['end']:.2f}: {segment['text']}")
```

## Models

Supported models:
- `tiny`, `tiny.en`
- `base`, `base.en`
- `small`, `small.en`
- `medium`, `medium.en`
- `large-v2`, `large-v3`
- `turbo` (large-v3-turbo)
- `distil-large-v3`

## Requirements

- Python 3.10+
- FFmpeg (for audio loading)
- For MLX backend: macOS with Apple Silicon (M1/M2/M3)
- For diarization: HuggingFace token

## License

BSD-2-Clause (same as WhisperX)
