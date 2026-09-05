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
- Word-level timestamps (`--align` / `transcribe_with_alignment`); wav2vec2 runs in eval mode (dropout off)
- Process-level cache of Whisper, Silero, and the aligner (warmup pays load; later calls in the same process reuse weights)
- Speaker diarization (`--diarize`)
- Silero VAD on the default transcribe path

## Performance

Measured on an M4 Max, `short.wav` (~90s), default `large-v3` MLX, Silero VAD, English, greedy temp-0. Timed runs are post-warmup in one process.

**ASR only** (`transcribe()`, no `--align`):

| Path | Wall time | Notes |
|---|---|---|
| Previous default MLX (fp16 decoder, batch 16) | 3.34s median | ASR-only pre-change |
| Current default MLX | **2.97–2.99s** overall | two 90s-idle sessions; every pair under 3.17s |
| Sequential per-VAD-chunk decode | 6.17s median | same machine, same audio |

Normalized WER versus the previous default ASR transcript is **0.0**. Spoken markers `famous` / `gordon` / `ramsay` still appear.

**ASR + word timestamps** (`transcribe_with_alignment` / CLI `--align`):

| Path | Wall time | Notes |
|---|---|---|
| Previous aligned MLX (reload Whisper+Silero+wav2vec2 every call) | 4.81s median | 4.81 / 4.75 / 5.03s |
| Current aligned MLX | **3.49s** median | 3.46 / 3.52s; **~1.38×** (~27% less); need &lt; 3.85s |

WER versus the previous aligned transcript is **0.0**. Same 268 words; markers appear in the **word** list. Word `start`/`end` match torchaudio wav2vec2 **eval** (dropout off) at 1 ms rounding; two timed runs are identical. Diarization is still opt-in and is not in these timings. A one-shot CLI process still pays model load; the 3.49s bar is reused weights after warmup.

## Changelog

Newest first. Numbers are the same M4 Max / `short.wav` / `large-v3` / Silero / greedy setup as above unless noted.

### 2026-09-05 — aligned path (`--align`)

Word timestamps on, no extra fast flag. Default model still `large-v3`.

| Step | What landed | `short.wav` wall time |
|---|---|---|
| Pre-change `transcribe_with_alignment` | reload Whisper + Silero + wav2vec2 every call, then `del` | **4.81s** median |
| Process-level ASR + align cache | warmup pays load; timed runs still run Silero + decode + CTC | ASR portion ~2.88s once cached |
| Load NLTK `punkt` **once** per `align()` | was once per segment (49× on this clip) | align 1.02s → **~0.49s** |
| wav2vec2 `model.eval()` | MLX Dropout defaulted to train (10% drop); eval matches torchaudio | correctness, not the bulk of the 1.3s |
| **Shipped aligned path** | cache + punkt-once + eval; wav2vec2 still **per-segment** (transformer batch moved 8/268 boxes) | **3.49s** median (~1.38× vs 4.81s) |

Tried and **not** shipped for `--align`: padding raw waveforms into a wav2vec2 batch (conv0 GroupNorm changes every frame), transformer batch with pad mask (8 words shifted, one by 264ms), loading MLX wav2vec2 on a worker thread (`no Stream in current thread`).

Unittest: `tests/test_aligned_path_speed_quality.py` (WER 0.0, markers in words, word-time stability, median &lt; 0.80 × 4.81s).

### 2026-09-04 — `1c40a24`

Default MLX ASR, no extra flag. Greedy tokens match the previous large-v3 transcript (WER 0.0).

| Step | What landed | `short.wav` wall time |
|---|---|---|
| Sequential per-VAD-chunk decode | quality/speed baseline (`transcribe_sequential`) | 6.17s median |
| Batched VAD encode/decode | already default from `1a4b975`; this commit’s pre-change measurement | 3.34s median |
| GPU timestamp filter, skip greedy logZ, preallocated decoder KV, encoder SDPA (dual-scale), pipeline `batch_size` 16, Silero `inference_mode` + 1 torch thread, reused `DecodingTask` | still fp16 decoder | ~3.15–3.17s cooled overall (not every pair under 3.17s) |
| **8-bit decoder** (`bits=8`, `group_size=64`); encoder stays fp16 | current default | **2.97s / 2.99s** overall on two independent 90s-idle sessions; every pair under 3.17s; unittest 2.93s vs sequential 6.17s (~2.1×) |

Tried and **not** shipped (WER rose or wall time got worse): `without_timestamps`, decoder SDPA, 8-bit encoder, `mx.compile` encoder wrapper, turbo/distil as default, transcribe-time Metal preheat.

### `c5f2c06` — alignment

MLX Wav2Vec2 encoder + NumPy CTC: **2.6×** faster `--align` than the previous alignment path. Not on the default transcribe path (still `--align`).

### `1a4b975` — batched MLX ASR

True batched encode/decode of VAD chunks (stack mels, one `DecodingTask.run`). About **6–7×** on long files versus sequential per-chunk `mlx_whisper.transcribe`. This is the batched path the later ASR numbers build on.

### `0a0713f`

Faster diarization startup (load diarization in parallel). Not ASR decode.

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
from whisperx_mlx import transcribe, transcribe_with_alignment

result = transcribe("audio.mp3", model="large-v3", backend="mlx", language="en")

for segment in result["segments"]:
    print(f"{segment['start']:.2f} - {segment['end']:.2f}: {segment['text']}")

aligned = transcribe_with_alignment(
    "audio.mp3", model="large-v3", backend="mlx", language="en"
)
for w in aligned["word_segments"]:
    print(f"{w['start']:.2f} - {w['end']:.2f}: {w['word']}")
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
