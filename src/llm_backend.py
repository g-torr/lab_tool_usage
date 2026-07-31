import json
import logging
import os
from typing import Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _clean_json_text(text: str | None) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = [line for line in cleaned.splitlines() if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        cleaned = cleaned[start : end + 1]
    return cleaned


def query_structured_output(
    prompt: str,
    system_prompt: str,
    response_model: Type[T],
    *,
    remote_client=None,
    remote_model: str | None = None,
    max_retries: int = 1,
) -> T:
    """Query a remote model first, then fall back to a local Hugging Face model when enabled."""
    if remote_client is not None and remote_model:
        try:
            return remote_client.chat.completions.create(
                model=remote_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                response_model=response_model,
                max_retries=max_retries,
            )
        except Exception as exc:
            logger.warning("Remote model call failed, trying local backend: %s", exc)

    if _is_truthy(os.getenv("USE_LOCAL_MODEL")):
        try:
            from transformers import pipeline

            model_name = os.getenv("LOCAL_BIOMED_MODEL", "microsoft/Phi-3-mini-4k-instruct")
            pipe = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                device_map="auto",
            )

            prompt_text = f"{system_prompt}\n\n{prompt}\n\nReturn valid JSON only."
            output = pipe(prompt_text, max_new_tokens=250, do_sample=False, temperature=0.0)
            generated_text = output[0].get("generated_text", "") if isinstance(output, list) else str(output)
            cleaned_text = _clean_json_text(generated_text)
            if not cleaned_text:
                raise ValueError("Local model returned no usable JSON")

            try:
                payload = json.loads(cleaned_text)
            except json.JSONDecodeError as exc:
                logger.warning("Local model returned invalid JSON: %s", exc)
                raise

            try:
                return response_model.model_validate(payload)
            except AttributeError:
                return response_model(**payload)
        except Exception as exc:
            logger.warning("Local model backend failed: %s", exc)

    raise RuntimeError("No usable LLM backend available")
