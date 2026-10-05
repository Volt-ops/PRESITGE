import os
from dotenv import load_dotenv
from groq import Groq
import edge_tts
import asyncio

load_dotenv()

class PrestigeEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from your .env file.")
        
        self.client = Groq(api_key=api_key)
        # Using active free text model from your list
        self.model_name = "qwen/qwen3.8-27b"

    def ask(self, prompt: str) -> str:
        """Sends a text prompt to Groq and returns the response."""
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error communicating with Groq API: {str(e)}"

    async def speak_text(self, text: str, output_file: str = "output.mp3"):
        """Generates voice output using edge-tts."""
        communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
        await communicate.save(output_file)

if __name__ == "__main__":
    engine = PrestigeEngine()
    test_response = engine.ask("Hello! Give me a 1-sentence confirmation that you are online.")
    print("Engine Output:", test_response)