import io
import os
import asyncio
import time
import mss
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ServerError
import edge_tts

# Initialize Gemini Client
client = genai.Client()

# List of models in order of preference (Primary -> Fallbacks)
PRIMARY_MODEL = 'gemini-3.8-flash'
FALLBACK_MODELS = ['gemini-3.5-flash', 'gemini-3.5-flash-lite']

def capture_screen_bytes() -> bytes:
    """Captures primary monitor on Windows using updated mss.MSS context."""
    with mss.MSS() as sct:
        monitor = sct.monitors[1]  # Primary display
        sct_img = sct.grab(monitor)
        
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        img.thumbnail((1280, 720))  # Scale down to 720p
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        return buffer.getvalue()

def generate_with_fallback(screen_bytes: bytes, user_query: str):
    """Tries primary model first; falls back if 503 high-demand error occurs."""
    models_to_try = [PRIMARY_MODEL] + FALLBACK_MODELS
    
    contents = [
        types.Part.from_bytes(data=screen_bytes, mime_type='image/jpeg'),
        f"Context: The user is looking at their Windows desktop. Provide a clear, concise response in under 3 sentences for voice playback.\nUser Query: {user_query}"
    ]
    
    for model_name in models_to_try:
        try:
            print(f"[Prestige Engine] Requesting analysis from {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=contents
            )
            return response.text
        except ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                print(f"[!] {model_name} is currently busy (503). Switching to fallback model...")
                time.sleep(1)  # Brief pause before retry
                continue
            else:
                raise e
        except Exception as e:
            print(f"[!] Error with {model_name}: {e}")
            continue

    raise Exception("All Gemini API models are currently unavailable. Please try again in a few seconds.")

async def analyze_screen_and_speak(user_query: str):
    """Captures screen context, routes to available Gemini model, and speaks response."""
    print("\n[Prestige Engine] Capturing primary screen...")
    screen_bytes = capture_screen_bytes()
    
    answer_text = generate_with_fallback(screen_bytes, user_query)
    print(f"\n[Prestige AI]: {answer_text}\n")
    
    print("[Prestige Engine] Generating voice response...")
    communicate = edge_tts.Communicate(answer_text, "en-US-SteffanNeural")
    audio_path = "temp_response.mp3"
    await communicate.save(audio_path)
    
    # Play local audio output on Windows natively
    os.system(f'start "" "{audio_path}"')

if __name__ == "__main__":
    query = "Describe what application or window is currently open on my screen."
    asyncio.run(analyze_screen_and_speak(query))