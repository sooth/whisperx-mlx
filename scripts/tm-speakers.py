#!/usr/bin/env python3
"""Rewrite .txt and .srt from whisperx-mlx JSON with speaker labels."""
import json, sys, pathlib

jp = pathlib.Path(sys.argv[1])
d = json.load(open(jp))
segs = d["segments"] if isinstance(d, dict) else d

# Fail loudly if diarization silently produced nothing — otherwise the .txt
# comes out as one [UNKNOWN] block and this script still reports success.
speakers = sorted({s["speaker"] for s in segs if s.get("speaker")})
if not speakers:
    print(f"    ERROR: no speaker labels in {jp.name} — diarization did not run "
          f"(check the log above for 'Pyannote diarization failed')", file=sys.stderr)
    sys.exit(2)

def ts(t):
    h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")

# .txt — grouped by speaker turn, readable transcript
with open(jp.with_suffix(".txt"), "w") as f:
    last = None
    for s in segs:
        sp = s.get("speaker", "UNKNOWN")
        txt = s.get("text", "").strip()
        if not txt:
            continue
        if sp != last:
            f.write(f"\n\n[{sp}]: {txt}")
            last = sp
        else:
            f.write(" " + txt)
    f.write("\n")

# .srt — per-segment with speaker prefix (stock WhisperX format)
with open(jp.with_suffix(".srt"), "w") as f:
    for i, s in enumerate(segs, 1):
        sp = s.get("speaker")
        pre = f"[{sp}]: " if sp else ""
        f.write(f"{i}\n{ts(s['start'])} --> {ts(s['end'])}\n{pre}{s.get('text','').strip()}\n\n")

print(f"    {len(speakers)} speakers ({', '.join(speakers)}) — labels written to "
      f"{jp.with_suffix('.txt').name} and {jp.with_suffix('.srt').name}")
