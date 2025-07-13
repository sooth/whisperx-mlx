#!/usr/bin/env python3
"""
Script to transcribe audio using lightning-whisper-mlx
"""

import sys
import os
from lightning_whisper_mlx import LightningWhisperMLX
from pathlib import Path

def transcribe_audio(audio_path, model_name="base", output_path=None):
    """
    Transcribe audio file using lightning-whisper-mlx
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model to use (tiny, base, small, medium, large, large-v2, large-v3)
        output_path: Path to save the transcription (optional)
    """
    
    # Check if audio file exists
    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' not found!")
        return False
    
    print(f"Loading lightning-whisper-mlx with model: {model_name}")
    
    # Initialize the model
    whisper = LightningWhisperMLX(model=model_name)
    
    print(f"Transcribing: {audio_path}")
    print("This may take a while for long audio files...")
    
    try:
        # Transcribe the audio
        result = whisper.transcribe(audio_path)
        
        # Print the result
        print("\n" + "="*50)
        print("TRANSCRIPTION RESULT:")
        print("="*50)
        print(result['text'])
        print("="*50)
        
        # Save to file if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(result['text'])
            print(f"\nTranscription saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python transcribe_audio.py <audio_file> [model_name] [output_file]")
        print("Example: python transcribe_audio.py 30m.wav base transcription.txt")
        print("Available models: tiny, base, small, medium, large, large-v2, large-v3")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "base"
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    success = transcribe_audio(audio_file, model_name, output_file)
    
    if success:
        print("\nTranscription completed successfully!")
    else:
        print("\nTranscription failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
