# rag_system/text_to_speech.py
import os
from pathlib import Path
from gtts import gTTS

def generate_audio(text, lang='en'):
    """
    Generates an audio file from text using gTTS.
    Saves the file and returns the path.
    """
    # Create a directory for audio files if it doesn't exist
    audio_dir = Path("audio_responses")
    audio_dir.mkdir(exist_ok=True)
    
    # Create a unique filename
    audio_file = audio_dir / f"response_{len(os.listdir(audio_dir)) + 1}.mp3"
    
    try:
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(audio_file)
        return str(audio_file)
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None