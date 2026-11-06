"""Planner: break a topic into 4-6 research subtasks."""

from datetime import datetime

from research_report_system.llm import OllamaClient
from research_report_system.schemas import ResearchPlan

PLAN_PROMPT = """You are a research planner. Break the topic below into 4 to 6 \
research subtasks. Each subtask needs a short unique "id" (e.g. "t1"), a \
one-sentence "description", and a "route" that is exactly one of: \
"library", "web", "arxiv", "wikipedia". Leave "status" as "pending". \
Set "topic" to the topic text.

Topic: {topic}"""


def plan_topic(topic: str, client: OllamaClient | None = None) -> ResearchPlan:
    client = client or OllamaClient()
    plan = client.generate_structured(PLAN_PROMPT.format(topic=topic), ResearchPlan)
    # The model must not invent timestamps; the planner owns created_at.
    return plan.model_copy(update={"created_at": datetime.now()})
