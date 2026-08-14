import os
import time
import random
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

DEFAULT_MODEL_NAME = "gemini-flash-latest"
FALLBACK_MODELS = [
    "gemini-flash-latest",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-flash-lite-latest",
    "gemini-pro-latest"
]

_client: Optional[genai.Client] = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set.\n"
                "Please set GEMINI_API_KEY in your .env file or export it: export GEMINI_API_KEY='...'"
            )
        _client = genai.Client(api_key=api_key)
    return _client


def generate_answer(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    max_retries_per_model: int = 3
) -> str:
    """
    Generate an answer using Gemini with model fallback and exponential backoff retry.
    Disables automatic function calling config to suppress SDK warning logs.
    """
    client = get_client()
    models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]

    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    last_exception = None
    for m in models_to_try:
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content(
                    model=m,
                    contents=prompt,
                    config=config
                )
                if response.text:
                    return response.text
            except genai_errors.APIError as e:
                last_exception = e
                # Check for rate limit or transient errors
                if hasattr(e, "code") and e.code in (429, 500, 503):
                    backoff = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(backoff)
                    continue
                else:
                    # Model might not be supported or bad request, try next model
                    break
            except Exception as e:
                last_exception = e
                backoff = (2 ** attempt) + random.uniform(0.5, 1.0)
                time.sleep(backoff)
                continue

    if last_exception:
        raise RuntimeError(f"All Gemini model attempts failed: {last_exception}") from last_exception
    raise RuntimeError("Failed to generate a response from Gemini.")