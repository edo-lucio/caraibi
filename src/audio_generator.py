import os
import io
import torch
import torchaudio as ta
import tempfile
from pydub import AudioSegment
from zyphra import AsyncZyphraClient
from zyphra.models.audio import EmotionWeights
from chatterbox.tts import ChatterboxTTS
from dotenv import load_dotenv
from typing import List, Optional
# Load environment variables
load_dotenv()

# Define emotions
# emotions = EmotionWeights(happiness=0.8)

# async def zyphra_request(text: str, output_path: str = "output_audio.mp3") -> str:
#     """
#     Generate audio using Zyphra's API and save to MP3 file.
    
#     Args:
#         text (str): Input text to convert to speech
#         output_path (str): Path to save the generated audio file
    
#     Returns:
#         str: Path to the generated audio file
#     """
#     try:
#         api_key = os.getenv("ZYPHRA_API_KEY", "zsk-aa92b1884686a91852db8bce6079b03bd523b6cf733562314bb589342a73087b")
#         async with AsyncZyphraClient(api_key=api_key) as client:
#             audio_data = await client.audio.speech.create(
#                 text=text,
#                 default_voice_name="american_male",
#                 speaking_rate=1.0,  # Normalized speaking rate (1.0 is default)
#                 emotions=emotions
#             )
            
#             with open(output_path, "wb") as f:
#                 f.write(audio_data)
                
#             return output_path
#     except Exception as e:
#         raise Exception(f"Error in Zyphra audio generation: {str(e)}")

def append_audios(paths: List[str], output_path: str) -> str:
    sounds = [AudioSegment.from_file(path, format="wav") for path in paths]
    combined = sum(sounds)
    os.remove(output_path)
    combined.export(output_path, format="wav")

def generate_audio_chatterbox(text: str, output_path: str = "output_audio.wav", use_tmp=False) -> str:
    audio_format = output_path.split(".")[-1]
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        output_path = f"{tmp.name}.{audio_format}" if use_tmp else output_path
        model = ChatterboxTTS.from_pretrained(device="cuda" if torch.cuda.is_available() else "cpu")
        wav = model.generate(text, audio_prompt_path="WhatsApp Audio 2025-08-17 at 12.13.17 PM.mp3")
        ta.save(output_path, wav, model.sr)
        return output_path
    
def generate_audio(text: str,  output_path: str = "output_audio.wav", chunk_length: int = 1500) -> str:
    text_chunks = [chunk.strip() + "." for chunk in text.split(".") if chunk.strip()] 
    current_audio_path = generate_audio_chatterbox(text_chunks[0], output_path=output_path)
    text = text_chunks[0]
    char_counter = len(text_chunks[0])

    for chunk in text_chunks[1:]:
        text += chunk
        char_counter += len(chunk)
        if char_counter >= chunk_length:
            tmp_audio_path = generate_audio_chatterbox(text, use_tmp=True)
            audio_paths = [current_audio_path, tmp_audio_path]
            append_audios(audio_paths, output_path)
            text, char_counter = "", 0
    
if __name__ == "__main__":
    import time
    start = time.time()
    with open("new_opt.txt", mode="r") as f:
        text = f.read()
        generate_audio(text)
    end = time.time()
    print(f"Elapsed {end - start}")