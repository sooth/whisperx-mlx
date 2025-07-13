
import whisperx
# Test parameters
audio_file = "short.wav"
model_size = "large-v3"
import torch
device = "mps" if torch.backends.mps.is_available() else "cpu"
compute_type = "float32"

import time

# Load audio once
audio = whisperx.load_audio(audio_file)

# Get Lightning results without VAD (to get word timestamps)
print("Processing Lightning WhisperX...")
model_lightning = whisperx.load_model(model_size, device, compute_type=compute_type, 
                                    backend="mlx_lightning", word_timestamps=True, vad_method=None)

start_time = time.time()
result_lightning = model_lightning.transcribe(audio, align_words=True)
end_time = time.time()
print(f"Lightning WhisperX time: {end_time - start_time} seconds")

