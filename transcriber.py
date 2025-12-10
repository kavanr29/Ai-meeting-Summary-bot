"""
Simple transcription using faster-whisper or Groq
"""

from faster_whisper import WhisperModel
import os

class Transcriber:
    def __init__(self, mode="local", api_key=None):
        """
        mode: "local" (faster-whisper) or "groq" (API)
        """
        self.mode = mode
        
        if mode == "local":
            # Use small model for speed
            self.model = WhisperModel("small", device="cpu", compute_type="int8")
        elif mode == "groq":
            self.api_key = api_key
    
    def transcribe(self, audio_path):
        """Transcribe audio file to text"""
        if self.mode == "local":
            return self._transcribe_local(audio_path)
        else:
            return self._transcribe_groq(audio_path)
    
    def _transcribe_local(self, audio_path):
        """Transcribe using local faster-whisper"""
        try:
            segments, info = self.model.transcribe(
                audio_path,
                language="en",
                vad_filter=True,
                beam_size=5
            )
            
            # Combine all segments
            text = " ".join([segment.text for segment in segments])
            return text.strip()
            
        except Exception as e:
            print(f"Transcription error: {e}")
            return ""
    
    def _transcribe_groq(self, audio_path):
        """Transcribe using Groq API (requires internet)"""
        try:
            from groq import Groq
            
            client = Groq(api_key=self.api_key)
            
            with open(audio_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=(os.path.basename(audio_path), audio_file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            
            return transcription.text
            
        except Exception as e:
            print(f"Groq API error: {e}")
            # Fallback to local
            return self._transcribe_local(audio_path)