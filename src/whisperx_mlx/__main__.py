"""
Command-line interface for WhisperX-MLX.

Usage:
    whisperx-mlx audio.mp3 --model large-v3
    whisperx-mlx audio.mp3 --model turbo --backend mlx
    whisperx-mlx audio.mp3 --model large-v3 --diarize --hf-token YOUR_TOKEN
"""

import argparse
import json
import logging
import os
import sys
import warnings
from typing import Optional

import torch

from whisperx_mlx.backends import (
    is_apple_silicon,
    is_mlx_available,
    get_default_backend,
    get_default_device,
)


def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="WhisperX-MLX: Speech recognition with MLX acceleration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "audio",
        nargs="+",
        help="Audio file(s) to transcribe",
    )

    # Model selection
    parser.add_argument(
        "--model", "-m",
        default="large-v3",
        help="Whisper model name (e.g., large-v3, turbo, small.en)",
    )
    parser.add_argument(
        "--backend", "-b",
        default="auto",
        choices=["auto", "mlx", "faster-whisper"],
        help="ASR backend to use. 'auto' selects MLX on Apple Silicon.",
    )

    # Device options
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use (auto-detected if not specified)",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="GPU device index for CUDA",
    )
    parser.add_argument(
        "--compute-type",
        default="float16",
        choices=["float16", "float32", "int8"],
        help="Compute precision type",
    )

    # Transcription options
    parser.add_argument(
        "--language", "-l",
        default=None,
        help="Language code (e.g., 'en', 'fr'). Auto-detected if not specified.",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to perform. 'translate' converts to English.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=30,
        help="Maximum audio chunk duration in seconds",
    )

    # VAD options
    parser.add_argument(
        "--vad-method",
        default="silero",
        choices=["silero", "pyannote"],
        help="Voice Activity Detection method (silero recommended due to pyannote compatibility issues)",
    )
    parser.add_argument(
        "--vad-onset",
        type=float,
        default=0.5,
        help="VAD onset threshold",
    )
    parser.add_argument(
        "--vad-offset",
        type=float,
        default=0.363,
        help="VAD offset threshold",
    )

    # Alignment options
    parser.add_argument(
        "--align",
        action="store_true",
        help="Perform word-level alignment",
    )
    parser.add_argument(
        "--align-model",
        default=None,
        help="Alignment model name (auto-selected based on language if not specified)",
    )

    # Diarization options
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Perform speaker diarization",
    )
    parser.add_argument(
        "--diarization-backend",
        default="auto",
        choices=["auto", "senko", "pyannote"],
        help="Diarization backend: 'auto' uses Senko on macOS Apple Silicon, pyannote elsewhere",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="HuggingFace token for diarization model (required for pyannote)",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum number of speakers (hint for diarization)",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum number of speakers (hint for diarization)",
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        default=".",
        help="Output directory for results",
    )
    parser.add_argument(
        "--output-format", "-f",
        default="json",
        choices=["json", "txt", "srt", "vtt", "tsv", "all"],
        help="Output format",
    )

    # ASR options
    parser.add_argument(
        "--initial-prompt",
        default=None,
        help="Initial prompt for the model",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam size for decoding",
    )
    parser.add_argument(
        "--no-speech-threshold",
        type=float,
        default=0.6,
        help="Threshold for no speech probability",
    )

    # Other options
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Number of CPU threads (0 for auto)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--print-progress",
        action="store_true",
        help="Print progress during transcription",
    )

    return parser


def write_output(result: dict, audio_path: str, output_dir: str, output_format: str):
    """Write transcription results to file."""
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(audio_path))[0]

    formats = [output_format] if output_format != "all" else ["json", "txt", "srt", "vtt", "tsv"]

    for fmt in formats:
        output_path = os.path.join(output_dir, f"{base_name}.{fmt}")

        if fmt == "json":
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)

        elif fmt == "txt":
            with open(output_path, "w", encoding="utf-8") as f:
                for segment in result.get("segments", []):
                    f.write(segment.get("text", "").strip() + "\n")

        elif fmt == "srt":
            with open(output_path, "w", encoding="utf-8") as f:
                for i, segment in enumerate(result.get("segments", []), 1):
                    start = format_timestamp(segment.get("start", 0), fmt="srt")
                    end = format_timestamp(segment.get("end", 0), fmt="srt")
                    text = segment.get("text", "").strip()
                    f.write(f"{i}\n{start} --> {end}\n{text}\n\n")

        elif fmt == "vtt":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for segment in result.get("segments", []):
                    start = format_timestamp(segment.get("start", 0), fmt="vtt")
                    end = format_timestamp(segment.get("end", 0), fmt="vtt")
                    text = segment.get("text", "").strip()
                    f.write(f"{start} --> {end}\n{text}\n\n")

        elif fmt == "tsv":
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("start\tend\ttext\n")
                for segment in result.get("segments", []):
                    start = segment.get("start", 0)
                    end = segment.get("end", 0)
                    text = segment.get("text", "").strip().replace("\t", " ")
                    f.write(f"{start:.3f}\t{end:.3f}\t{text}\n")

        print(f"Saved: {output_path}")


def format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """Format seconds as SRT or VTT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    if fmt == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}".replace(".", ",")
    else:  # vtt
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def cli():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Print system info
    logger.info("WhisperX-MLX")
    logger.info(f"Platform: {'Apple Silicon' if is_apple_silicon() else 'Other'}")
    logger.info(f"MLX available: {is_mlx_available()}")
    logger.info(f"Default backend: {get_default_backend()}")

    # Set threads if specified
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        logger.info(f"Using {args.threads} CPU threads")

    # Prepare ASR options
    asr_options = {
        "initial_prompt": args.initial_prompt,
        "temperatures": (args.temperature,) if args.temperature > 0 else (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        "beam_size": args.beam_size,
        "no_speech_threshold": args.no_speech_threshold,
    }

    # Import transcription functions
    from whisperx_mlx.transcribe import transcribe, transcribe_with_alignment, transcribe_with_diarization

    # Process each audio file
    for audio_path in args.audio:
        logger.info(f"Processing: {audio_path}")

        try:
            if args.diarize:
                result = transcribe_with_diarization(
                    audio_path,
                    model=args.model,
                    hf_token=args.hf_token,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                    diarization_backend=args.diarization_backend,
                    backend=args.backend,
                    device=args.device,
                    device_index=args.device_index,
                    compute_type=args.compute_type,
                    batch_size=args.batch_size,
                    chunk_size=args.chunk_size,
                    language=args.language,
                    task=args.task,
                    vad_method=args.vad_method,
                    vad_onset=args.vad_onset,
                    vad_offset=args.vad_offset,
                    asr_options=asr_options,
                    print_progress=args.print_progress,
                    verbose=args.verbose,
                    align=args.align,
                )
            elif args.align:
                result = transcribe_with_alignment(
                    audio_path,
                    model=args.model,
                    align_model=args.align_model,
                    backend=args.backend,
                    device=args.device,
                    device_index=args.device_index,
                    compute_type=args.compute_type,
                    batch_size=args.batch_size,
                    chunk_size=args.chunk_size,
                    language=args.language,
                    task=args.task,
                    vad_method=args.vad_method,
                    vad_onset=args.vad_onset,
                    vad_offset=args.vad_offset,
                    asr_options=asr_options,
                    print_progress=args.print_progress,
                    verbose=args.verbose,
                )
            else:
                result = transcribe(
                    audio_path,
                    model=args.model,
                    backend=args.backend,
                    device=args.device,
                    device_index=args.device_index,
                    compute_type=args.compute_type,
                    batch_size=args.batch_size,
                    chunk_size=args.chunk_size,
                    language=args.language,
                    task=args.task,
                    vad_method=args.vad_method,
                    vad_onset=args.vad_onset,
                    vad_offset=args.vad_offset,
                    asr_options=asr_options,
                    print_progress=args.print_progress,
                    verbose=args.verbose,
                )

            # Write output
            write_output(result, audio_path, args.output_dir, args.output_format)

            # Print summary
            num_segments = len(result.get("segments", []))
            language = result.get("language", "unknown")
            logger.info(f"Completed: {num_segments} segments, language: {language}")

        except Exception as e:
            logger.error(f"Error processing {audio_path}: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            continue

    logger.info("Done!")


if __name__ == "__main__":
    cli()
