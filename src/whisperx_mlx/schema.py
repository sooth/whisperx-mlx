"""
Data schema definitions for WhisperX-MLX.

These TypedDicts define the structure of transcription results,
segments, and aligned outputs used throughout the pipeline.
"""

from typing import TypedDict, Optional, List, Tuple


class SingleWordSegment(TypedDict):
    """A single word of a speech with timing and confidence."""
    word: str
    start: float
    end: float
    score: float


class SingleCharSegment(TypedDict):
    """A single character of a speech with timing and confidence."""
    char: str
    start: float
    end: float
    score: float


class SingleSegment(TypedDict):
    """A single segment (up to multiple sentences) of a speech."""
    start: float
    end: float
    text: str


class SegmentData(TypedDict):
    """
    Temporary processing data used during alignment.
    Contains cleaned and preprocessed data for each segment.
    """
    clean_char: List[str]  # Cleaned characters that exist in model dictionary
    clean_cdx: List[int]   # Original indices of cleaned characters
    clean_wdx: List[int]   # Indices of words containing valid characters
    sentence_spans: List[Tuple[int, int]]  # Start and end indices of sentences


class SingleAlignedSegment(TypedDict):
    """A single segment of speech with word-level alignment."""
    start: float
    end: float
    text: str
    words: List[SingleWordSegment]
    chars: Optional[List[SingleCharSegment]]


class TranscriptionResult(TypedDict):
    """Result of transcription: segments and detected language."""
    segments: List[SingleSegment]
    language: str


class AlignedTranscriptionResult(TypedDict):
    """Result of alignment: segments with word-level timing."""
    segments: List[SingleAlignedSegment]
    word_segments: List[SingleWordSegment]
