import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

def generate_answer(prompt: str,
                    model_name: str = DEFAULT_MODEL_NAME) -> str:

    response = client.models.generate_content(
        model=model_name,
        contents=prompt
    )

    return response.text