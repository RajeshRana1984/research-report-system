import pytest

from research_report_system.llm import OllamaClient, StructuredOutputError
from research_report_system.planner import plan_topic
from research_report_system.schemas import ResearchPlan

TOPIC = "Why do retirement plans change recordkeepers?"


def test_planner_with_stub():
    import json
    plan = {"topic": TOPIC, "subtasks": [
        {"id": f"t{i}", "description": "d", "route": "web"} for i in range(4)]}
    client = OllamaClient(chat_fn=lambda **kw: {"message": {"content": json.dumps(plan)}})
    assert len(plan_topic(TOPIC, client).subtasks) == 4


def test_planner_rejects_three_subtasks():
    import json
    bad = {"topic": TOPIC, "subtasks": [
        {"id": f"t{i}", "description": "d", "route": "web"} for i in range(3)]}
    client = OllamaClient(chat_fn=lambda **kw: {"message": {"content": json.dumps(bad)}})
    with pytest.raises(StructuredOutputError):
        plan_topic(TOPIC, client)


@pytest.mark.slow
def test_real_llama_returns_valid_plan():
    plan = plan_topic(TOPIC)
    assert isinstance(plan, ResearchPlan) and 4 <= len(plan.subtasks) <= 6
