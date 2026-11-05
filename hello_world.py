"""Stage 0 smoke test: local Ollama answers with an exact string."""

import sys

import ollama

MODEL = "llama3.1:latest"
PROMPT = "Reply with exactly: schemas ready"
EXPECTED = "schemas ready"


def main() -> None:
    try:
        resp = ollama.chat(model=MODEL, messages=[{"role": "user", "content": PROMPT}])
    except Exception as exc:  # connection refused, model missing, etc.
        sys.exit(f"FAIL: could not reach Ollama / model {MODEL}: {exc!r}\n"
                 "Is `ollama serve` running and the model pulled?")
    reply = resp["message"]["content"].strip()
    print(reply)
    assert reply == EXPECTED, f"expected {EXPECTED!r}, got {reply!r}"
    print("OK")


if __name__ == "__main__":
    main()
