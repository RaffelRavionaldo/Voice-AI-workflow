from transformers import pipeline, AutoProcessor, AutoModel
import torch
import io
import os
import soundfile as sf  # Untuk menyimpan output TTS
from pydub import AudioSegment  # Untuk memainkan audio
from pydub.playback import play
import time
from speech import HuggingFaceTTS, WhisperSTT

def test_tts():
    print("Testing Text-to-Speech (TTS)...")
    tts = HuggingFaceTTS()
    
    test_text = "Hello, this is a test of the text to speech system. How are you today?"
    
    print(f"Generating speech for text: '{test_text}'")
    audio_array, sample_rate = tts.generate_speech(test_text)
    
    output_file = "tts_output.wav"
    sf.write(output_file, audio_array, sample_rate)
    print(f"Audio saved to {output_file}")
    
    try:
        audio = AudioSegment.from_wav(output_file)
        print("Playing the generated audio...")
        play(audio)
    except Exception as e:
        print(f"Could not play audio: {e}. Please check if ffmpeg is installed.")

def test_stt():
    print("\nTesting Speech-to-Text (STT)...")
    stt = WhisperSTT()
    
    test_audio = "recordings/recording_20250426_213610.wav"

    if not os.path.exists(test_audio):
        print(f"Test audio file not found at {test_audio}")
        print("Please provide a WAV file for testing.")
        return
    
    print(f"Transcribing audio file: {test_audio}")
    transcription = stt.transcribe(test_audio)
    print(f"Transcription result: {transcription}")

if __name__ == "__main__":
    print("Testing STT and TTS systems...")
    
    test_tts()
    
    time.sleep(2)
    
    test_stt()