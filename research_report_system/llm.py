"""LLM access layer. Placeholder for Stage 1.

Bedrock slots in here in Stage 7; for now everything is local Ollama.
"""


def generate(prompt: str, model: str = "llama3.1:latest") -> str:
    """Return a completion for `prompt`. Implemented in Stage 1."""
    raise NotImplementedError("llm.generate is implemented in Stage 1")
