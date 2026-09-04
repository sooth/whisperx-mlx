#!/usr/bin/env python3
"""Gating test for the GitHub whisperx-mlx default MLX path.

Drives shipped load_model → transcribe vs sequential per-VAD-segment
mlx_whisper.transcribe on real speech (short.wav).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SHORT_WAV = ROOT / "short.wav"
MODEL = os.environ.get("WHISPERX_TEST_MODEL", "large-v3")
SPOKEN_MARKERS = ("famous", "gordon", "ramsay")


def _ensure_hf_home() -> None:
    x10 = "/Volumes/Crucial X10/huggingface"
    if os.path.isdir(x10):
        os.environ.setdefault("HF_HOME", x10)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(x10, "hub"))


def _join_text(result: dict) -> str:
    return " ".join(s.get("text", "") for s in result.get("segments", []) if s.get("text"))


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _wer(hypothesis: str, reference: str) -> float:
    hyp = _normalize(hypothesis).split()
    ref = _normalize(reference).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            dp[j] = prev if ref[i - 1] == hyp[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m] / n


class TestDefaultPathSpeedQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_hf_home()
        if not SHORT_WAV.is_file():
            raise unittest.SkipTest(f"missing {SHORT_WAV}")
        if not os.path.isdir("/Volumes/Crucial X10"):
            raise unittest.SkipTest("Crucial X10 unmounted; refusing model load")

        from whisperx_mlx.backends import load_model
        from whisperx_mlx.audio import load_audio
        from whisperx_mlx.backends.mlx_speedups import apply_mlx_decode_speedups
        from mlx_whisper.decoding import ApplyTimestampRules, GreedyDecoder

        apply_mlx_decode_speedups()
        assert ApplyTimestampRules.apply.__name__ == "timestamp_apply"
        assert GreedyDecoder.update.__name__ == "greedy_update"

        cls.load_audio = load_audio
        cls.audio = load_audio(str(SHORT_WAV))
        cls.pipe = load_model(
            MODEL,
            backend="mlx",
            language="en",
            vad_method="silero",
            asr_options={
                "temperatures": (0.0,),
                "condition_on_previous_text": False,
            },
        )

    def test_decode_speedups_are_the_default_path(self):
        from mlx_whisper.decoding import ApplyTimestampRules, GreedyDecoder, Inference
        from mlx_whisper.whisper import MultiHeadAttention
        from whisperx_mlx.backends.mlx_speedups import apply_mlx_decode_speedups

        apply_mlx_decode_speedups()
        apply_mlx_decode_speedups()
        self.assertEqual(ApplyTimestampRules.apply.__name__, "timestamp_apply")
        self.assertEqual(GreedyDecoder.update.__name__, "greedy_update")
        self.assertIn("mlx_speedups", Inference.logits.__code__.co_filename)
        self.assertIn("mlx_speedups", MultiHeadAttention.qkv_attention.__code__.co_filename)
        q = self.pipe._model.decoder.blocks[0].attn.query
        self.assertEqual(
            type(q).__name__,
            "QuantizedLinear",
            "default MLX decoder must be 8-bit quantized",
        )

    def test_batched_matches_sequential_and_is_faster(self):
        pipe = self.pipe
        audio = self.audio

        _ = pipe.transcribe(audio, language="en", verbose=False)

        seq_times = []
        opt_times = []
        seq_text = None
        opt_text = None
        opt_result = None

        for _ in range(2):
            t0 = time.perf_counter()
            seq = pipe.transcribe_sequential(audio, language="en", verbose=False)
            seq_times.append(time.perf_counter() - t0)
            seq_text = _join_text(seq)

            t0 = time.perf_counter()
            opt = pipe.transcribe(audio, language="en", verbose=False)
            opt_times.append(time.perf_counter() - t0)
            opt_text = _join_text(opt)
            opt_result = opt

        seq_med = statistics.median(seq_times)
        opt_med = statistics.median(opt_times)
        spread = max(
            statistics.pstdev(seq_times) if len(seq_times) > 1 else 0.0,
            statistics.pstdev(opt_times) if len(opt_times) > 1 else 0.0,
            0.05 * seq_med,
        )
        wer = _wer(opt_text, seq_text)

        payload = {
            "model": MODEL,
            "seq_times": seq_times,
            "opt_times": opt_times,
            "seq_median": seq_med,
            "opt_median": opt_med,
            "spread": spread,
            "speedup": (seq_med / opt_med) if opt_med else None,
            "wer": wer,
            "n_opt_segments": len(opt_result.get("segments", [])),
        }
        print("SPEED_JSON " + json.dumps(payload))
        scratch = os.environ.get("WHISPERX_TEST_SCRATCH")
        if scratch:
            Path(scratch).mkdir(parents=True, exist_ok=True)
            (Path(scratch) / "speed.json").write_text(json.dumps(payload, indent=2) + "\n")

        self.assertTrue(opt_result["segments"])
        for seg in opt_result["segments"]:
            self.assertIn("start", seg)
            self.assertIn("end", seg)
            self.assertIn("text", seg)
            self.assertGreaterEqual(seg["end"], seg["start"])

        joined = _normalize(opt_text)
        for marker in SPOKEN_MARKERS:
            self.assertIn(marker, joined)

        self.assertEqual(wer, 0.0, f"WER vs sequential is {wer:.4f}")
        self.assertLess(
            opt_med,
            seq_med - spread,
            f"batched median {opt_med:.3f}s not faster than sequential "
            f"{seq_med:.3f}s by more than spread {spread:.3f}s",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
