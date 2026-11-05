# research-report-system

A multi-agent research & report system. Stage 0 (this repo state) provides the
foundation: Pydantic v2 schemas for the whole pipeline and an Ollama smoke test.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com).

```bash
uv sync                        # creates .venv with Python 3.12
ollama pull llama3.1:latest
ollama pull nomic-embed-text:latest
```

## Run

```bash
uv run pytest                  # schema tests
uv run python hello_world.py   # needs `ollama serve` running; asserts "schemas ready"
```

## Layout

- `research_report_system/schemas.py` – Subtask, ResearchPlan, Source, Chunk, Claim, Insight, ReportDraft, GateDecision
- `research_report_system/llm.py` – placeholder for Stage 1 (Bedrock slots in at Stage 7)
- `tests/test_schemas.py` – good data passes, bad data raises

Not in Stage 0: LangGraph, vector DB, MCP (Stages 2–5). `GateDecision` only
reserves the shape for the Stage 4 graph interrupts.
