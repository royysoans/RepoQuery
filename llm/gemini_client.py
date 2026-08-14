import os
from dotenv import load_dotenv
from google import genai

DEFAULT_MODEL_NAME = "gemini-flash-latest"


def _get_client(model_name: str = DEFAULT_MODEL_NAME):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a key from https://aistudio.google.com/apikey and "
            "export GEMINI_API_KEY=your_key_here"
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def generate_answer(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    model = _get_client(model_name)
    response = model.generate_content(prompt)

    if not response.candidates:
        raise RuntimeError("Gemini returned no candidates — request may have been blocked.")

    return response.text
