"""
Voice Activity Detection (VAD) modules for WhisperX-MLX.

Provides VAD models for preprocessing audio before transcription,
improving accuracy by segmenting speech regions.
"""

from whisperx_mlx.vads.vad import Vad

__all__ = ["Vad"]

# Lazy imports to avoid loading large models unnecessarily
def get_pyannote():
    from whisperx_mlx.vads.pyannote import Pyannote
    return Pyannote

def get_silero():
    from whisperx_mlx.vads.silero import Silero
    return Silero
