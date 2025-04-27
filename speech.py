from transformers import pipeline, AutoProcessor, AutoModel
import torch
import io
import os

# STT with Whisper
class WhisperSTT:
    def __init__(self, model_size="small"):
        self.model = pipeline(
            "automatic-speech-recognition",
            model=f"openai/whisper-{model_size}",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def transcribe(self, audio_file_path):
        if not os.path.exists(audio_file_path):
            raise FileNotFoundError(f"File {audio_file_path} not found")
        result = self.model(
                    audio_file_path,
                    generate_kwargs={
                            "task": "transcribe",
                            "language": "english"
                        })
        return result["text"]

# TTS with HuggingFace (English)
class HuggingFaceTTS:
    def __init__(self):
        self.processor = AutoProcessor.from_pretrained("suno/bark-small")
        self.model = AutoModel.from_pretrained("suno/bark-small").to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def generate_speech(self, text):
        inputs = self.processor(
            text=text,
            return_tensors="pt",
        ).to(self.model.device)
        
        with torch.no_grad():
            speech = self.model.generate(**inputs)
        
        audio_array = speech.cpu().numpy().squeeze()
        return audio_array, 24000  # sample rate
