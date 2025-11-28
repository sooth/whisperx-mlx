"""
NumPy-based CTC forced alignment functions.

These replace the PyTorch-based CTC functions for use with MLX,
eliminating the PyTorch dependency in the alignment pipeline.

Optimized for vectorization where possible.
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Point:
    """A point in the CTC alignment path."""
    token_index: int
    time_index: int
    score: float


@dataclass
class Segment:
    """An aligned character segment."""
    label: str
    start: int
    end: int
    score: float

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def get_wildcard_emission_np(frame_emission: np.ndarray, tokens: np.ndarray, blank_id: int) -> np.ndarray:
    """
    Get emission scores for tokens, handling wildcards (-1).

    Fully vectorized NumPy implementation.

    Args:
        frame_emission: Log probabilities for one frame, shape (vocab_size,)
        tokens: Token indices, shape (num_tokens,). -1 indicates wildcard.
        blank_id: Index of the blank token

    Returns:
        Emission scores for each token, shape (num_tokens,)
    """
    tokens = np.asarray(tokens)

    # Create mask for wildcards
    wildcard_mask = (tokens == -1)

    # Get scores for regular tokens (use 0 for wildcards temporarily)
    safe_tokens = np.clip(tokens, 0, None)
    regular_scores = frame_emission[safe_tokens]

    # For wildcards, use max score excluding blank
    if wildcard_mask.any():
        mask = np.ones(len(frame_emission), dtype=bool)
        mask[blank_id] = False
        max_valid_score = frame_emission[mask].max()
        regular_scores = np.where(wildcard_mask, max_valid_score, regular_scores)

    return regular_scores


def get_trellis_np(emission: np.ndarray, tokens: List[int], blank_id: int = 0) -> np.ndarray:
    """
    Build CTC trellis using dynamic programming.

    Optimized NumPy implementation with vectorized operations.

    Args:
        emission: Log probabilities, shape (num_frames, vocab_size)
        tokens: List of token indices
        blank_id: Index of the blank token

    Returns:
        Trellis matrix, shape (num_frames, num_tokens)
    """
    num_frames = emission.shape[0]
    num_tokens = len(tokens)
    tokens_arr = np.array(tokens)

    # Initialize trellis
    trellis = np.zeros((num_frames, num_tokens), dtype=np.float32)

    # First column: cumulative blank probabilities
    trellis[1:, 0] = np.cumsum(emission[1:, blank_id])

    # First row (except first element): impossible
    trellis[0, 1:] = -np.inf

    # Last rows in first column that would require impossible alignment
    if num_tokens > 1:
        trellis[-num_tokens + 1:, 0] = np.inf

    # Fill trellis using vectorized operations where possible
    # Pre-compute all wildcard emissions for tokens[1:]
    tokens_shifted = tokens_arr[1:]  # tokens for "change" transitions

    for t in range(num_frames - 1):
        # Score for staying at same token (emit blank)
        stay_scores = trellis[t, 1:] + emission[t, blank_id]

        # Score for transitioning to next token
        change_emissions = get_wildcard_emission_np(emission[t], tokens_shifted, blank_id)
        change_scores = trellis[t, :-1] + change_emissions

        # Take maximum of stay vs change
        trellis[t + 1, 1:] = np.maximum(stay_scores, change_scores)

    return trellis


def backtrack_beam_np(
    trellis: np.ndarray,
    emission: np.ndarray,
    tokens: List[int],
    blank_id: int = 0,
    beam_width: int = 5
) -> Optional[List[Point]]:
    """
    Beam search backtracking through CTC trellis.

    NumPy implementation with optimizations.

    Args:
        trellis: CTC trellis from get_trellis_np
        emission: Log probabilities
        tokens: Token indices
        blank_id: Blank token index
        beam_width: Number of beams to maintain

    Returns:
        List of Points representing the alignment path, or None if failed
    """
    T = trellis.shape[0] - 1
    J = trellis.shape[1] - 1
    tokens_arr = np.array(tokens)

    # Initialize beam with final position
    init_score = float(trellis[T, J])
    init_prob = float(np.exp(emission[T, blank_id]))

    # Use lists of tuples for beams: (token_idx, time_idx, score, path)
    beams = [(J, T, init_score, [Point(J, T, init_prob)])]

    while beams and beams[0][0] > 0:  # token_index > 0
        next_beams = []

        for token_idx, time_idx, score, path in beams:
            if time_idx <= 0:
                continue

            t = time_idx
            j = token_idx

            # Probabilities
            p_stay = float(np.exp(emission[t - 1, blank_id]))

            # Get change probability
            token_val = tokens_arr[j]
            if token_val == -1:
                # Wildcard: use max non-blank
                mask = np.ones(emission.shape[1], dtype=bool)
                mask[blank_id] = False
                p_change = float(np.exp(emission[t - 1, mask].max()))
            else:
                p_change = float(np.exp(emission[t - 1, token_val]))

            # Scores from trellis
            stay_score = float(trellis[t - 1, j])
            change_score = float(trellis[t - 1, j - 1]) if j > 0 else -np.inf

            # Stay transition
            if not math.isinf(stay_score):
                new_path = path.copy()
                new_path.append(Point(j, t - 1, p_stay))
                next_beams.append((j, t - 1, stay_score, new_path))

            # Change transition
            if j > 0 and not math.isinf(change_score):
                new_path = path.copy()
                new_path.append(Point(j - 1, t - 1, p_change))
                next_beams.append((j - 1, t - 1, change_score, new_path))

        # Sort by score and keep top beam_width
        next_beams.sort(key=lambda x: x[2], reverse=True)
        beams = next_beams[:beam_width]

        if not beams:
            break

    if not beams:
        return None

    # Get best beam
    token_idx, time_idx, score, path = beams[0]

    # Complete path back to time 0
    t = time_idx
    j = token_idx
    while t > 0:
        prob = float(np.exp(emission[t - 1, blank_id]))
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]


def merge_repeats_np(path: List[Point], transcript: str) -> List[Segment]:
    """
    Merge consecutive points with same token into segments.

    Pure Python, already optimal.

    Args:
        path: Alignment path from backtrack
        transcript: Character transcript

    Returns:
        List of Segment objects
    """
    i1, i2 = 0, 0
    segments = []

    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1

        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
            )
        )
        i1 = i2

    return segments


# Optimized batch processing for multiple segments
def align_segment_np(
    emission: np.ndarray,
    tokens: List[int],
    blank_id: int = 0,
    beam_width: int = 2
) -> Optional[List[Segment]]:
    """
    Align a single segment using NumPy CTC.

    Convenience function combining trellis + backtrack + merge.

    Args:
        emission: Log probabilities for segment
        tokens: Token indices
        blank_id: Blank token index
        beam_width: Beam search width

    Returns:
        List of aligned Segments or None if alignment failed
    """
    trellis = get_trellis_np(emission, tokens, blank_id)
    path = backtrack_beam_np(trellis, emission, tokens, blank_id, beam_width)

    if path is None:
        return None

    # Need transcript for merge_repeats
    # This function should be called with pre-processed text
    return path, trellis
