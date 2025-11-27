"""
Senko (CoreML) diarization backend for WhisperX-MLX.

Senko provides CoreML-accelerated speaker diarization on Apple Silicon,
achieving ~11 seconds for 2 hours of audio on M3.

Note: Senko is run in a subprocess to avoid OMP runtime conflicts with torch.
This is necessary because both torch and senko's CoreML backend use OpenMP,
and having both in the same process causes SIGSEGV crashes on long audio.

See: https://github.com/narcotic-sh/senko
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from typing import Optional, Union, List, Dict, Tuple

import numpy as np

from whisperx_mlx.diarization.base import (
    DiarizationBackend,
    DiarizationSegment,
    normalize_speaker_ids,
)

logger = logging.getLogger(__name__)


# Subprocess script to run senko in isolation
SENKO_SUBPROCESS_SCRIPT = '''
import json
import sys
import os

def main():
    # Set OMP environment before importing senko
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

    import senko

    audio_path = sys.argv[1]
    min_speakers = int(sys.argv[2]) if sys.argv[2] != "None" else None
    max_speakers = int(sys.argv[3]) if sys.argv[3] != "None" else None
    device = sys.argv[4]

    diarizer = senko.Diarizer(device=device, warmup=False)

    kwargs = {}
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    result = diarizer.diarize(audio_path, **kwargs)

    # Output JSON to stdout
    output = {
        "segments": result.get("merged_segments", result.get("segments", [])),
        "speakers_detected": result.get("merged_speakers_detected", 0),
    }
    print(json.dumps(output))

if __name__ == "__main__":
    main()
'''


class SenkoDiarizationPipeline(DiarizationBackend):
    """CoreML-accelerated diarization using Senko.

    Senko uses Apple's CoreML and Neural Engine for fast speaker diarization
    on macOS with Apple Silicon.

    Note: Senko is run in a subprocess to avoid OMP runtime conflicts with
    torch, which can cause SIGSEGV crashes on long audio files.
    """

    def __init__(self, device: str = "auto"):
        """Initialize the Senko diarization pipeline.

        Args:
            device: Device for computation. Senko supports 'auto', 'cpu', 'mps'.
                    'auto' uses CoreML with ANE when available.
        """
        # Verify senko is installed (but don't import it - that happens in subprocess)
        try:
            import importlib.util
            if importlib.util.find_spec("senko") is None:
                raise ImportError("senko not found")
        except ImportError:
            raise ImportError(
                "Senko is required for CoreML diarization. "
                "Install with: pip install git+https://github.com/narcotic-sh/senko.git"
            )

        # Map device names
        if device == "mlx":
            device = "auto"  # Senko uses 'auto' for best device selection
        elif device == "cuda":
            logger.warning("CUDA not supported by Senko, falling back to 'auto'")
            device = "auto"

        logger.info(f"Initializing Senko diarizer with device={device}")
        self._device = device
        self._temp_audio_file = None

    @property
    def backend_name(self) -> str:
        return "senko"

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
    ) -> Union[List[DiarizationSegment], Tuple[List[DiarizationSegment], Optional[Dict[str, np.ndarray]]]]:
        """Run speaker diarization on audio using Senko.

        Args:
            audio: Path to audio file or numpy array (16kHz mono)
            min_speakers: Minimum number of speakers (passed to Senko)
            max_speakers: Maximum number of speakers (passed to Senko)
            return_embeddings: Whether to return speaker embeddings

        Returns:
            List of DiarizationSegment objects with speaker labels.
            If return_embeddings is True, also returns speaker embeddings dict.
        """
        # Senko requires 16kHz mono WAV - convert if needed
        temp_file = None
        if isinstance(audio, np.ndarray):
            import soundfile as sf
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            # Ensure mono
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            sf.write(temp_file.name, audio, 16000, subtype='PCM_16')
            audio_path = temp_file.name
        else:
            # Convert file to 16kHz mono if needed
            audio_path = self._convert_audio_for_senko(audio)

        try:
            # Run diarization in subprocess to avoid OMP conflicts with torch
            logger.debug(f"Running Senko diarization on {audio_path} (subprocess)")
            result = self._run_senko_subprocess(
                audio_path, min_speakers, max_speakers
            )

            # Convert Senko output to our segment format
            segments = self._convert_senko_output(result)

            # Normalize speaker IDs to SPEAKER_00 format
            segments = normalize_speaker_ids(segments)

            if return_embeddings:
                embeddings = self._extract_embeddings(segments)
                return segments, embeddings

            return segments

        except Exception as e:
            logger.error(f"Senko diarization failed: {e}")
            if return_embeddings:
                return [], None
            return []

        finally:
            # Clean up temp files
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass
            if self._temp_audio_file:
                try:
                    os.unlink(self._temp_audio_file)
                except OSError:
                    pass
                self._temp_audio_file = None

    def _run_senko_subprocess(
        self,
        audio_path: str,
        min_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> dict:
        """Run Senko in a subprocess to avoid OMP conflicts with torch.

        This is necessary because both torch and senko's CoreML use OpenMP,
        and having both in the same process causes SIGSEGV on long audio.
        """
        # Write the subprocess script to a temp file
        script_file = tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        )
        script_file.write(SENKO_SUBPROCESS_SCRIPT)
        script_file.close()

        try:
            # Run subprocess with clean OMP environment to avoid conflicts
            env = os.environ.copy()
            # These settings prevent OMP conflicts between torch and CoreML
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            env["OMP_NUM_THREADS"] = "1"

            # Try to find a Python that doesn't have torch loaded
            # The isolated senko venv works, falling back to sys.executable
            python_candidates = [
                "/tmp/senko_isolated/.venv/bin/python",  # Our isolated senko venv
                "/usr/bin/python3",  # System Python (no torch)
                sys.executable,  # Fallback to current Python
            ]

            python_exe = sys.executable
            for candidate in python_candidates:
                if os.path.exists(candidate):
                    python_exe = candidate
                    break

            logger.debug(f"Using Python interpreter: {python_exe}")

            result = subprocess.run(
                [
                    python_exe,
                    script_file.name,
                    audio_path,
                    str(min_speakers),
                    str(max_speakers),
                    self._device,
                ],
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
                env=env,
            )

            if result.returncode != 0:
                logger.error(f"Senko subprocess failed (exit {result.returncode}): {result.stderr}")
                raise RuntimeError(f"Senko subprocess failed: {result.stderr}")

            # Log any warnings from stderr (like OMP deprecation warnings)
            if result.stderr:
                logger.debug(f"Senko subprocess warnings: {result.stderr.strip()}")

            # Parse JSON output from subprocess
            try:
                output = json.loads(result.stdout.strip())
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Senko output: {result.stdout}")
                raise RuntimeError(f"Failed to parse Senko output: {e}")
            return output

        finally:
            try:
                os.unlink(script_file.name)
            except OSError:
                pass

    def _convert_audio_for_senko(self, audio_path: str) -> str:
        """Convert audio file to 16kHz mono WAV format required by Senko.

        Args:
            audio_path: Path to input audio file

        Returns:
            Path to converted audio file (temp file if conversion needed)
        """
        import soundfile as sf

        # Read audio file
        try:
            audio_data, sample_rate = sf.read(audio_path)
        except Exception as e:
            logger.warning(f"Could not read audio with soundfile: {e}, trying scipy")
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(audio_path)
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Check if conversion is needed
        needs_conversion = False

        # Convert to mono if stereo
        if audio_data.ndim > 1 and audio_data.shape[1] > 1:
            audio_data = audio_data.mean(axis=1)
            needs_conversion = True
        elif audio_data.ndim > 1:
            audio_data = audio_data.flatten()
            needs_conversion = True

        # Resample to 16kHz if needed
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio_data) * 16000 / sample_rate)
            audio_data = signal.resample(audio_data, num_samples)
            needs_conversion = True

        if needs_conversion:
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(temp_file.name, audio_data.astype(np.float32), 16000, subtype='PCM_16')
            self._temp_audio_file = temp_file.name
            return temp_file.name

        return audio_path

    def _convert_senko_output(self, result: dict) -> List[DiarizationSegment]:
        """Convert Senko output format to DiarizationSegment list.

        Senko returns segments with 'speaker' key in SPEAKER_XX format.
        """
        segments = []

        raw_segments = result.get("segments", [])

        for seg in raw_segments:
            # Senko uses 'speaker' key with SPEAKER_XX format
            speaker_id = seg.get("speaker", seg.get("speaker_id", "0"))
            segments.append(DiarizationSegment(
                start=float(seg["start"]),
                end=float(seg["end"]),
                speaker=str(speaker_id),
            ))

        return segments

    def _extract_embeddings(
        self,
        segments: List[DiarizationSegment],
    ) -> Optional[Dict[str, np.ndarray]]:
        """Extract speaker embeddings.

        Note: Senko doesn't expose speaker embeddings directly.
        This returns placeholder embeddings for API compatibility.
        """
        if not segments:
            return None

        # Return placeholder embeddings (Senko doesn't expose these)
        unique_speakers = set(seg.speaker for seg in segments)
        return {
            speaker: np.zeros(256)  # Placeholder embedding
            for speaker in unique_speakers
        }
