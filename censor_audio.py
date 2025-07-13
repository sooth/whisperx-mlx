#!/usr/bin/env python3
"""Use Lightning word-level timestamps to censor bad words from audio"""

import numpy as np
import whisperx
import torch
import soundfile as sf
from scipy.io import wavfile

# Define bad words to censor
BAD_WORDS = {
    'fuck', 'fucking', 'motherfucking', 'shit', 'shits', 'damn', 'hell', 'ass', 'bitch',
    'fucks', 'fucked', 'fucker', 'fuckers', 'motherfucker', 'motherfuckers'
}

def is_bad_word(word):
    """Check if a word should be censored"""
    return word.lower().strip('.,!?') in BAD_WORDS

def apply_beep(audio, start_sample, end_sample, sample_rate):
    """Replace audio segment with a beep sound"""
    duration = (end_sample - start_sample) / sample_rate
    t = np.linspace(0, duration, end_sample - start_sample)
    # 1000 Hz beep
    beep = 0.3 * np.sin(2 * np.pi * 1000 * t)
    audio[start_sample:end_sample] = beep
    return audio

def apply_silence(audio, start_sample, end_sample):
    """Replace audio segment with silence"""
    audio[start_sample:end_sample] = 0
    return audio

def main():
    # Parameters
    audio_file = "short.wav"
    output_file = "censored.wav"
    model_size = "distil-large-v3"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    compute_type = "float32"
    censor_method = "remove"  # "beep", "silence", or "remove"
    
    print(f"Censoring bad words from {audio_file}")
    print("="*60)
    
    # Load audio for processing
    audio_data, sample_rate = sf.read(audio_file)
    if len(audio_data.shape) > 1:
        # Convert to mono if stereo
        audio_data = audio_data.mean(axis=1)
    
    # Load audio for WhisperX
    audio = whisperx.load_audio(audio_file)
    
    # Get transcription with word timestamps using optimized Lightning
    print("\n1. Transcribing with optimized Lightning (batch word alignment)...")
    
    # Use the new optimized Lightning backend with batch word alignment
    model = whisperx.load_model(model_size, device, compute_type=compute_type, 
                               backend="mlx_lightning", word_timestamps=True)
    
    # Transcribe with batch processing and word alignment in one pass
    result = model.transcribe(audio, batch_size=16, word_timestamps=True)
    
    # Collect all words
    all_words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []):
            all_words.append(word)
    
    # Sort words by start time (should already be sorted, but just in case)
    all_words.sort(key=lambda w: w.get("start", 0))
    
    print(f"\n2. Transcription complete. Found {len(all_words)} total words.")
    
    # Find bad words
    bad_words = []
    for word in all_words:
        if is_bad_word(word.get("word", "")):
            bad_words.append(word)
    
    print(f"\n3. Found {len(bad_words)} bad words to censor:")
    for w in bad_words:
        print(f"   - '{w['word']}' at {w['start']:.2f}s - {w['end']:.2f}s")
    
    # Apply censoring
    print(f"\n4. Applying {censor_method} censoring...")
    
    if censor_method == "remove":
        # Create list of segments to keep
        segments_to_keep = []
        current_pos = 0
        
        # Sort bad words by start time
        bad_words_sorted = sorted(bad_words, key=lambda w: w['start'])
        
        for word in bad_words_sorted:
            start_sample = int(word['start'] * sample_rate)
            end_sample = int(word['end'] * sample_rate)
            
            # Add padding
            padding_samples = int(0.05 * sample_rate)  # 50ms padding
            start_sample = max(0, start_sample - padding_samples)
            end_sample = min(len(audio_data), end_sample + padding_samples)
            
            # Keep audio before bad word
            if start_sample > current_pos:
                segments_to_keep.append(audio_data[current_pos:start_sample])
            
            # Skip bad word
            current_pos = end_sample
        
        # Keep remaining audio
        if current_pos < len(audio_data):
            segments_to_keep.append(audio_data[current_pos:])
        
        # Concatenate segments
        censored_audio = np.concatenate(segments_to_keep) if segments_to_keep else np.array([])
        
    else:
        # Original beep/silence method
        censored_audio = audio_data.copy()
        
        for word in bad_words:
            start_sample = int(word['start'] * sample_rate)
            end_sample = int(word['end'] * sample_rate)
            
            # Add small padding to ensure complete censoring
            padding_samples = int(0.05 * sample_rate)  # 50ms padding
            start_sample = max(0, start_sample - padding_samples)
            end_sample = min(len(censored_audio), end_sample + padding_samples)
            
            if censor_method == "beep":
                censored_audio = apply_beep(censored_audio, start_sample, end_sample, sample_rate)
            else:
                censored_audio = apply_silence(censored_audio, start_sample, end_sample)
    
    # Save censored audio
    sf.write(output_file, censored_audio, sample_rate)
    print(f"\n5. Saved censored audio to {output_file}")
    
    # Show duration change
    original_duration = len(audio_data) / sample_rate
    censored_duration = len(censored_audio) / sample_rate
    print(f"   Original duration: {original_duration:.2f}s")
    print(f"   Censored duration: {censored_duration:.2f}s")
    print(f"   Removed: {original_duration - censored_duration:.2f}s")
    
    # Verify censoring by transcribing the censored file
    print("\n6. Verifying censored audio...")
    censored_whisper = whisperx.load_audio(output_file)
    result_censored = model.transcribe(censored_whisper, batch_size=16)
    
    censored_text = ' '.join([seg['text'].strip() for seg in result_censored.get('segments', [])])
    print(f"\nOriginal text sample: {' '.join([w['word'] for w in all_words[:30]])}...")
    print(f"Censored text sample: {censored_text[:150]}...")
    
    # Count remaining bad words
    remaining_bad = 0
    for word in censored_text.split():
        if is_bad_word(word):
            remaining_bad += 1
    
    if remaining_bad == 0:
        print(f"\n✓ Success! All bad words have been censored.")
    else:
        print(f"\n⚠ Warning: {remaining_bad} bad words may still remain.")
    
    print(f"\nCensoring complete! Output saved to {output_file}")

if __name__ == "__main__":
    main()