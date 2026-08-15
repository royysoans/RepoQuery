import os
import time
import random
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

DEFAULT_MODEL_NAME = "gemini-3.6-flash"
FALLBACK_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-flash-latest"
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
    
    # Deduplicate model list while preserving preference order
    models_to_try = []
    for m in [model_name] + FALLBACK_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

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
                err_msg = str(e).lower()
                if "404" in err_msg or "not_found" in err_msg or "not found" in err_msg or "no longer available" in err_msg:
                    # Invalid or deprecated model, break immediately to try next model
                    break
                if hasattr(e, "code") and e.code in (429, 500, 503):
                    backoff = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(backoff)
                    continue
                else:
                    break
            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()
                if "404" in err_msg or "not_found" in err_msg or "not found" in err_msg or "no longer available" in err_msg:
                    break
                backoff = (2 ** attempt) + random.uniform(0.5, 1.0)
                time.sleep(backoff)
                continue

    if last_exception:
        raise RuntimeError(f"All Gemini model attempts failed: {last_exception}") from last_exception
    raise RuntimeError("Failed to generate a response from Gemini.")


def generate_answer_stream(
    prompt: str,
    model_name: str = DEFAULT_MODEL_NAME,
    max_retries_per_model: int = 3
):
    """
    Stream answer tokens using Gemini with model fallback.
    Yields text chunks as they arrive.
    """
    client = get_client()
    models_to_try = []
    for m in [model_name] + FALLBACK_MODELS:
        if m not in models_to_try:
            models_to_try.append(m)

    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
    )

    last_exception = None
    for m in models_to_try:
        for attempt in range(max_retries_per_model):
            try:
                response = client.models.generate_content_stream(
                    model=m,
                    contents=prompt,
                    config=config
                )
                yielded = False
                for chunk in response:
                    if chunk.text:
                        yielded = True
                        yield chunk.text
                if yielded:
                    return
            except genai_errors.APIError as e:
                last_exception = e
                err_msg = str(e).lower()
                if "404" in err_msg or "not_found" in err_msg or "not found" in err_msg or "no longer available" in err_msg:
                    break
                if hasattr(e, "code") and e.code in (429, 500, 503):
                    backoff = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(backoff)
                    continue
                else:
                    break
            except Exception as e:
                last_exception = e
                err_msg = str(e).lower()
                if "404" in err_msg or "not_found" in err_msg or "not found" in err_msg or "no longer available" in err_msg:
                    break
                backoff = (2 ** attempt) + random.uniform(0.5, 1.0)
                time.sleep(backoff)
                continue

    if last_exception:
        raise RuntimeError(f"All Gemini model attempts failed: {last_exception}") from last_exception
    raise RuntimeError("Failed to generate a response from Gemini.")