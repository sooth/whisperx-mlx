"""
Pyannote-based Voice Activity Detection for WhisperX-MLX.

Uses pyannote-audio for robust speech detection with support for
WhisperX's min-cut operation for handling long speech segments.
"""

import os
import logging
from typing import Optional, Union, Callable, List, Dict, Any

import numpy as np
import torch
from pyannote.audio import Model
from pyannote.audio.core.io import AudioFile
from pyannote.audio.pipelines import VoiceActivityDetection
from pyannote.audio.pipelines.utils import PipelineModel
from pyannote.core import Annotation, SlidingWindowFeature, Segment

from whisperx_mlx.vads.vad import Vad, Segment as SegmentX

logger = logging.getLogger(__name__)


class Binarize:
    """Binarize detection scores using hysteresis thresholding.

    Includes min-cut operation to ensure segments don't exceed max_duration,
    as described in the WhisperX paper.

    Reference:
        Gregory Gelly and Jean-Luc Gauvain. "Minimum Word Error Training of
        RNN-based Voice Activity Detection", InterSpeech 2015.

        Modified by Max Bain for WhisperX's min-cut operation:
        https://arxiv.org/abs/2303.00747
    """

    def __init__(
        self,
        onset: float = 0.5,
        offset: Optional[float] = None,
        min_duration_on: float = 0.0,
        min_duration_off: float = 0.0,
        pad_onset: float = 0.0,
        pad_offset: float = 0.0,
        max_duration: float = float('inf'),
    ):
        self.onset = onset
        self.offset = offset or onset
        self.pad_onset = pad_onset
        self.pad_offset = pad_offset
        self.min_duration_on = min_duration_on
        self.min_duration_off = min_duration_off
        self.max_duration = max_duration

    def __call__(self, scores: SlidingWindowFeature) -> Annotation:
        """Binarize detection scores.

        Args:
            scores: Detection scores from the segmentation model

        Returns:
            Annotation with binarized speech regions
        """
        num_frames, num_classes = scores.data.shape
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(num_frames)]

        active = Annotation()

        for k, k_scores in enumerate(scores.data.T):
            label = k if scores.labels is None else scores.labels[k]

            start = timestamps[0]
            is_active = k_scores[0] > self.onset
            curr_scores = [k_scores[0]]
            curr_timestamps = [start]
            t = start

            for t, y in zip(timestamps[1:], k_scores[1:]):
                if is_active:
                    curr_duration = t - start
                    if curr_duration > self.max_duration:
                        # Min-cut: divide segment at lowest score
                        search_after = len(curr_scores) // 2
                        min_score_div_idx = search_after + np.argmin(curr_scores[search_after:])
                        min_score_t = curr_timestamps[min_score_div_idx]
                        region = Segment(start - self.pad_onset, min_score_t + self.pad_offset)
                        active[region, k] = label
                        start = curr_timestamps[min_score_div_idx]
                        curr_scores = curr_scores[min_score_div_idx + 1:]
                        curr_timestamps = curr_timestamps[min_score_div_idx + 1:]
                    elif y < self.offset:
                        region = Segment(start - self.pad_onset, t + self.pad_offset)
                        active[region, k] = label
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                    curr_scores.append(y)
                    curr_timestamps.append(t)
                else:
                    if y > self.onset:
                        start = t
                        is_active = True

            if is_active:
                region = Segment(start - self.pad_onset, t + self.pad_offset)
                active[region, k] = label

        # Merge overlapping regions and fill short gaps
        if self.pad_offset > 0.0 or self.pad_onset > 0.0 or self.min_duration_off > 0.0:
            if self.max_duration < float("inf"):
                raise NotImplementedError("Padding not supported with max_duration")
            active = active.support(collar=self.min_duration_off)

        # Remove short segments
        if self.min_duration_on > 0:
            for segment, track in list(active.itertracks()):
                if segment.duration < self.min_duration_on:
                    del active[segment, track]

        return active


class VoiceActivitySegmentation(VoiceActivityDetection):
    """Voice Activity Segmentation pipeline for WhisperX-MLX."""

    def __init__(
        self,
        segmentation: PipelineModel = "pyannote/segmentation",
        fscore: bool = False,
        use_auth_token: Optional[str] = None,
        **inference_kwargs,
    ):
        super().__init__(
            segmentation=segmentation,
            fscore=fscore,
            use_auth_token=use_auth_token,
            **inference_kwargs
        )

    def apply(self, file: AudioFile, hook: Optional[Callable] = None) -> SlidingWindowFeature:
        """Apply voice activity detection.

        Args:
            file: Audio file to process
            hook: Optional callback for debugging

        Returns:
            Segmentation scores as SlidingWindowFeature
        """
        hook = self.setup_hook(file, hook=hook)

        if self.training:
            if self.CACHED_SEGMENTATION in file:
                segmentations = file[self.CACHED_SEGMENTATION]
            else:
                segmentations = self._segmentation(file)
                file[self.CACHED_SEGMENTATION] = segmentations
        else:
            segmentations = self._segmentation(file)

        return segmentations


def load_vad_model(
    device: torch.device,
    vad_onset: float = 0.5,
    vad_offset: float = 0.363,
    use_auth_token: Optional[str] = None,
    model_fp: Optional[str] = None,
) -> VoiceActivitySegmentation:
    """Load the Pyannote VAD model.

    Args:
        device: PyTorch device for the model
        vad_onset: Onset threshold
        vad_offset: Offset threshold
        use_auth_token: HuggingFace auth token (optional)
        model_fp: Path to model file (uses bundled model if None)

    Returns:
        Configured VoiceActivitySegmentation pipeline
    """
    if model_fp is None:
        # Use bundled model
        assets_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_fp = os.path.join(assets_dir, "assets", "pytorch_model.bin")

    model_fp = os.path.abspath(model_fp)

    if not os.path.exists(model_fp):
        raise FileNotFoundError(f"VAD model file not found: {model_fp}")

    vad_model = Model.from_pretrained(model_fp, use_auth_token=use_auth_token)

    hyperparameters = {
        "onset": vad_onset,
        "offset": vad_offset,
        "min_duration_on": 0.1,
        "min_duration_off": 0.1,
    }

    vad_pipeline = VoiceActivitySegmentation(
        segmentation=vad_model,
        device=device,
    )
    vad_pipeline.instantiate(hyperparameters)

    return vad_pipeline


class Pyannote(Vad):
    """Pyannote-based Voice Activity Detection.

    Uses pyannote-audio's segmentation model with WhisperX's
    min-cut operation for optimal segment lengths.
    """

    def __init__(
        self,
        device: torch.device,
        use_auth_token: Optional[str] = None,
        model_fp: Optional[str] = None,
        vad_onset: float = 0.5,
        vad_offset: float = 0.363,
        **kwargs,
    ):
        """Initialize Pyannote VAD.

        Args:
            device: PyTorch device for the model
            use_auth_token: HuggingFace auth token
            model_fp: Path to model file (optional)
            vad_onset: Onset threshold
            vad_offset: Offset threshold
        """
        super().__init__(vad_onset=vad_onset, **kwargs)
        logger.info("Initializing Pyannote voice activity detection...")

        self.vad_offset = vad_offset
        self.vad_pipeline = load_vad_model(
            device=device,
            vad_onset=vad_onset,
            vad_offset=vad_offset,
            use_auth_token=use_auth_token,
            model_fp=model_fp,
        )

    def __call__(self, audio: AudioFile, **kwargs) -> SlidingWindowFeature:
        """Run VAD on audio.

        Args:
            audio: Audio data dictionary with 'waveform' and 'sample_rate'

        Returns:
            Segmentation scores
        """
        return self.vad_pipeline(audio)

    @staticmethod
    def preprocess_audio(audio: np.ndarray) -> torch.Tensor:
        """Preprocess audio for Pyannote VAD.

        Args:
            audio: NumPy array of audio samples

        Returns:
            PyTorch tensor with batch dimension
        """
        return torch.from_numpy(audio).unsqueeze(0)

    @staticmethod
    def merge_chunks(
        segments: SlidingWindowFeature,
        chunk_size: float,
        onset: float = 0.5,
        offset: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Merge VAD segments into chunks for transcription.

        Args:
            segments: Segmentation scores from VAD
            chunk_size: Maximum chunk duration
            onset: Onset threshold
            offset: Offset threshold

        Returns:
            List of merged segment dictionaries
        """
        assert chunk_size > 0

        binarize = Binarize(max_duration=chunk_size, onset=onset, offset=offset)
        segments_annotation = binarize(segments)

        segments_list = []
        for speech_turn in segments_annotation.get_timeline():
            segments_list.append(SegmentX(speech_turn.start, speech_turn.end, "UNKNOWN"))

        if not segments_list:
            logger.warning("No active speech found in audio")
            return []

        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)
