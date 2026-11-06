"""LLM access layer: a thin wrapper around local Ollama.

The model is swappable by design (Bedrock slots in at Stage 7); callers only
use `OllamaClient.generate_structured`, `count_tokens` and `truncate_to_budget`.
"""

import json
import math
from typing import Any, Callable, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODEL = "llama3.1:latest"
DEFAULT_MAX_PROMPT_TOKENS = 6000
MAX_RETRIES = 2
CHARS_PER_TOKEN = 4

RETRY_SUFFIX = "Return ONLY valid JSON matching this schema"


class StructuredOutputError(RuntimeError):
    """The model never produced output that validates against the schema."""


def count_tokens(text: str) -> int:
    """Estimate the token count as ceil(characters / 4).

    This is a rough heuristic for English text, NOT an exact count: the real
    number depends on the model's tokenizer and can differ noticeably for code,
    non-English text, or unusual punctuation.
    """
    return math.ceil(len(text) / CHARS_PER_TOKEN)


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Cut `text` so that count_tokens(result) <= max_tokens (estimate-based)."""
    if max_tokens < 0:
        raise ValueError("max_tokens must be >= 0")
    return text[: max_tokens * CHARS_PER_TOKEN]


class OllamaClient:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_prompt_tokens: int = DEFAULT_MAX_PROMPT_TOKENS,
        chat_fn: Callable[..., Any] | None = None,
    ):
        """`chat_fn` defaults to ollama.chat; tests inject a fake."""
        self.model = model
        self.max_prompt_tokens = max_prompt_tokens
        self._chat = chat_fn or ollama.chat

    def _complete_json(self, prompt: str, schema: dict) -> str:
        resp = self._chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            format=schema,  # Ollama constrains output to this JSON schema
            options={"temperature": 0},
        )
        return resp["message"]["content"]

    def generate_structured(self, prompt: str, model_class: type[T]) -> T:
        """Ask the model for JSON matching `model_class` and validate it.

        Makes one attempt plus up to MAX_RETRIES retries on invalid output
        (bad JSON or schema violation); each retry appends a reminder to the
        prompt. Raises StructuredOutputError if all attempts fail.
        """
        schema = model_class.model_json_schema()
        schema_text = json.dumps(schema)
        # Truncate only the caller's text so the schema/instructions survive.
        body = truncate_to_budget(prompt, self.max_prompt_tokens)
        full = f"{body}\n\nRespond with JSON matching this schema:\n{schema_text}"

        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            if attempt > 0:
                full = f"{full}\n\n{RETRY_SUFFIX}: {schema_text}"
            raw = self._complete_json(full, schema)
            try:
                return model_class.model_validate_json(raw)
            except ValidationError as exc:
                last_error = exc
        raise StructuredOutputError(
            f"{self.model} failed to return valid {model_class.__name__} JSON after "
            f"{MAX_RETRIES} retries. Last error: {last_error}"
        ) from last_error
