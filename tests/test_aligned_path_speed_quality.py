#!/usr/bin/env python3
"""Gating test for the shipped MLX path with word-level timestamps.

Drives public transcribe_with_alignment (same as CLI --align) on short.wav.
Speed bar is the pre-change aligned median in PRE_CHANGE_ALIGNED_*, not ASR-only.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SHORT_WAV = ROOT / "short.wav"
MODEL = os.environ.get("WHISPERX_TEST_MODEL", "large-v3")
SPOKEN_MARKERS = ("famous", "gordon", "ramsay")

# Captured on 1c40a24 current tree before aligned-path speed edits
# (warmup + 3 timed transcribe_with_alignment runs). See {SCRATCH}/pre_change_aligned.json.
PRE_CHANGE_ALIGNED_TIMES = (4.812753624981269, 4.748302457970567, 5.025801999960095)
PRE_CHANGE_ALIGNED_MEDIAN = 4.812753624981269
PRE_CHANGE_ALIGNED_PSTDEV = 0.11857934934256623
PRE_CHANGE_ALIGNED_TRANSCRIPT = (
    "That's why he's so fucking famous, bro. That's why Gordon Ramsay's so famous. "
    "The only one there. Because he's the only one over there who knows how to make food. "
    "They're like, we got one. He goes to the UK. He meets with King Charles. I mean, this is "
    "right after he got kicked out of the White House. I don't know. I've had to verify that. "
    "He was kicked out of the fucking White House. After that fucking meeting, they were "
    "supposed to have a joint. Yeah, he went on a Brett Baer or somebody and like half "
    "apologized. Half apologized. Listen, and dude, real talk, the motherfucker has never "
    "said thank you. Ever. He's come here and threatened us multiple times. Dude, straight "
    "up demanding that we do this. Demanding it. Demanding it. Dude, who do you think you "
    "are, little man? You know what I'm saying? Like, bro, I'll beat your fucking ass. So "
    "fucking, oh, just about, the guy's fucking 5'5\". 5'8\". He's over, no, he ain't. If they "
    "claim 5'8\", he's 5'5\". Yes. You know how the internet works. Okay? So, it's true. It's "
    "true. Yeah. He's 5'5\". So, you know, I don't know, dude. I don't. I'm thinking shit's "
    "bigger than me. Did you see that fucking video of that dude crying? Oh, the, the, the, "
    "the, the, the bat. Oh, dude. Dude. I thought that was like a fake. I thought it was a "
    "first time. Dude, you know what? That guy has to be a fucking influencer that they pay "
    "to do that. Yes. There's no fucking way. I'm not crying on TV. I'm not crying."
)


def _ensure_hf_home() -> None:
    x10 = "/Volumes/Crucial X10/huggingface"
    if os.path.isdir(x10):
        os.environ.setdefault("HF_HOME", x10)
        os.environ.setdefault("HF_HUB_CACHE", os.path.join(x10, "hub"))


def _join_text(result: dict) -> str:
    return " ".join(s.get("text", "") for s in result.get("segments", []) if s.get("text"))


def _word_list(result: dict) -> list:
    words = result.get("word_segments") or []
    if words:
        return words
    nested = []
    for seg in result.get("segments", []):
        nested.extend(seg.get("words") or [])
    return nested


def _join_words(words: list) -> str:
    return " ".join(w.get("word", "") for w in words if w.get("word"))


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


class TestAlignedPathSpeedQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_hf_home()
        if not SHORT_WAV.is_file():
            raise unittest.SkipTest(f"missing {SHORT_WAV}")
        if not os.path.isdir("/Volumes/Crucial X10"):
            raise unittest.SkipTest("Crucial X10 unmounted; refusing model load")

        from whisperx_mlx.audio import load_audio

        cls.audio = load_audio(str(SHORT_WAV))

    def test_aligned_path_matches_prechange_and_is_faster(self):
        import time
        from whisperx_mlx.transcribe import transcribe_with_alignment

        audio = self.audio
        twa = transcribe_with_alignment

        # Warmup: pays Whisper + wav2vec2 load. Timed runs must still run
        # Silero + decode + CTC (models cached; VAD results not cached).
        _ = twa(audio, model=MODEL, backend="mlx", language="en")

        times = []
        runs = []
        for _ in range(2):
            t0 = time.perf_counter()
            runs.append(twa(audio, model=MODEL, backend="mlx", language="en"))
            times.append(time.perf_counter() - t0)

        last = runs[-1]
        self.assertIsNotNone(last)
        words = _word_list(last)
        words_a = _word_list(runs[0])
        self.assertEqual(len(words_a), len(words), "word count changed between timed runs")
        for wa, wb in zip(words_a, words):
            self.assertEqual(wa.get("word"), wb.get("word"))
            self.assertEqual(
                wa.get("start"),
                wb.get("start"),
                f"word start jitter {wa.get('word')!r}: {wa.get('start')} vs {wb.get('start')}",
            )
            self.assertEqual(
                wa.get("end"),
                wb.get("end"),
                f"word end jitter {wa.get('word')!r}: {wa.get('end')} vs {wb.get('end')}",
            )
        self.assertTrue(words, "alignment skipped: no word_segments / segments[].words")
        for w in words:
            self.assertIn("word", w)
            self.assertIn("start", w)
            self.assertIn("end", w)
            self.assertGreaterEqual(w["end"], w["start"])

        opt_text = _join_text(last)
        joined_words = _join_words(words)
        self.assertTrue(opt_text.strip(), "empty transcript")
        for marker in SPOKEN_MARKERS:
            self.assertIn(marker, _normalize(joined_words), f"marker {marker!r} missing from words")

        wer = _wer(opt_text, PRE_CHANGE_ALIGNED_TRANSCRIPT)
        opt_med = statistics.median(times)
        base_med = PRE_CHANGE_ALIGNED_MEDIAN
        spread = max(
            PRE_CHANGE_ALIGNED_PSTDEV,
            statistics.pstdev(times) if len(times) > 1 else 0.0,
            0.05 * base_med,
        )
        bar_20 = 0.80 * base_med
        bar_spread = base_med - spread

        payload = {
            "model": MODEL,
            "align": True,
            "path": "transcribe_with_alignment",
            "opt_times": times,
            "opt_median": opt_med,
            "pre_change_times": list(PRE_CHANGE_ALIGNED_TIMES),
            "pre_change_median": base_med,
            "spread": spread,
            "bar_20": bar_20,
            "bar_spread": bar_spread,
            "speedup": (base_med / opt_med) if opt_med else None,
            "wer": wer,
            "n_segments": len(last.get("segments") or []),
            "n_words": len(words),
        }
        print("SPEED_ALIGNED_JSON " + json.dumps(payload))
        scratch = os.environ.get("WHISPERX_TEST_SCRATCH")
        if scratch:
            Path(scratch).mkdir(parents=True, exist_ok=True)
            (Path(scratch) / "speed_aligned.json").write_text(
                json.dumps(payload, indent=2) + "\n"
            )

        self.assertEqual(wer, 0.0, f"WER vs pre-change aligned transcript is {wer:.4f}")
        self.assertLess(
            opt_med,
            bar_20,
            f"aligned median {opt_med:.3f}s not < 0.80 × pre-change {base_med:.3f}s ({bar_20:.3f}s)",
        )
        self.assertLess(
            opt_med,
            bar_spread,
            f"aligned median {opt_med:.3f}s not faster than pre-change "
            f"{base_med:.3f}s by more than spread {spread:.3f}s",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
