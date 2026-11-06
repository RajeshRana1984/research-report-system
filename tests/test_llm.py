import pytest

from research_report_system.llm import (
    OllamaClient, StructuredOutputError, count_tokens, truncate_to_budget,
)
from research_report_system.schemas import GateDecision


class FakeChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, **kw):
        self.prompts.append(kw["messages"][0]["content"])
        return {"message": {"content": self.replies.pop(0)}}


GOOD = '{"gate": "plan", "decision": "approved"}'


def test_valid_first_try():
    fake = FakeChat([GOOD])
    out = OllamaClient(chat_fn=fake).generate_structured("hi", GateDecision)
    assert out.decision == "approved" and len(fake.prompts) == 1


def test_retries_then_succeeds():
    fake = FakeChat(["not json", GOOD])
    out = OllamaClient(chat_fn=fake).generate_structured("hi", GateDecision)
    assert out.gate == "plan"
    assert "Return ONLY valid JSON matching this schema" in fake.prompts[1]
    assert "Return ONLY valid JSON" not in fake.prompts[0]


def test_invalid_json_raises_after_two_retries():
    fake = FakeChat(["garbage", '{"gate": "nope"}', "{}"])
    with pytest.raises(StructuredOutputError, match="2 retries"):
        OllamaClient(chat_fn=fake).generate_structured("hi", GateDecision)
    assert len(fake.prompts) == 3  # 1 attempt + 2 retries, no more


def test_token_estimate():
    assert count_tokens("") == 0
    assert count_tokens("abcd") == 1
    assert count_tokens("abcde") == 2


def test_truncate_keeps_within_budget():
    text = "word " * 10_000
    out = truncate_to_budget(text, 100)
    assert count_tokens(out) <= 100 and len(out) > 0
    assert truncate_to_budget("short", 100) == "short"


def test_long_prompt_is_truncated_but_schema_kept():
    fake = FakeChat([GOOD])
    OllamaClient(max_prompt_tokens=50, chat_fn=fake).generate_structured("x" * 10_000, GateDecision)
    assert "schema" in fake.prompts[0] and len(fake.prompts[0]) < 10_000
