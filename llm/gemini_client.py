import os
from typing import Optional
from dotenv import load_dotenv
from google import genai

DEFAULT_MODEL_NAME = "gemini-flash-latest"
FALLBACK_MODELS = ["gemini-flash-latest", "gemini-flash-lite-latest", "gemini-pro-latest"]

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer(prompt: str,
                    model_name: str = DEFAULT_MODEL_NAME) -> str:
    client = get_client()
    models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]

    last_exception = None
    for m in models_to_try:
        try:
            response = client.models.generate_content(
                model=m,
                contents=prompt
            )
            return response.text
        except Exception as e:
            last_exception = e
            continue

    if last_exception:
        raise last_exception