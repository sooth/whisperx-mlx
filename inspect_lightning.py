#!/usr/bin/env python3
"""Inspect lightning-whisper-mlx mel processing"""
import sys
sys.path.insert(0, '/Users/dmalson/whisperx-mlx/lightning-whisper-mlx/lightning-whisper-mlx-env/lib/python3.11/site-packages')

import inspect
from lightning_whisper_mlx import transcribe

# Get the source of the transcribe function
source = inspect.getsource(transcribe.transcribe)

# Find the mel processing part
lines = source.split('\n')
for i, line in enumerate(lines):
    if 'pad_or_trim' in line:
        print(f"Line {i}: {line}")
        # Print context
        for j in range(max(0, i-5), min(len(lines), i+6)):
            print(f"  {j}: {lines[j]}")