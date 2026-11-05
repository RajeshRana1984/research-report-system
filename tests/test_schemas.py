from datetime import date

import pytest
from pydantic import ValidationError

from research_report_system.schemas import (
    Chunk, Claim, GateDecision, Insight, ReportDraft, ResearchPlan, Source, Subtask,
)


def subtask(i=1, **kw):
    return Subtask(**{"id": f"s{i}", "description": "d", "route": "web", **kw})


def claim(**kw):
    base = dict(claim_id="c1", text="t", quote="q", chunk_id="k1", strength="single-source")
    return Claim(**{**base, **kw})


def test_subtask_good():
    assert subtask().status == "pending"


def test_subtask_bad_route():
    with pytest.raises(ValidationError):
        subtask(route="twitter")


def test_subtask_bad_status():
    with pytest.raises(ValidationError):
        subtask(status="finished")


@pytest.mark.parametrize("n", [4, 5, 6])
def test_plan_good(n):
    plan = ResearchPlan(topic="x", subtasks=[subtask(i) for i in range(n)])
    assert len(plan.subtasks) == n and plan.created_at


@pytest.mark.parametrize("n", [0, 3, 7])
def test_plan_bad_count(n):
    with pytest.raises(ValidationError):
        ResearchPlan(topic="x", subtasks=[subtask(i) for i in range(n)])


def test_source_good():
    s = Source(source_id="a", url="https://example.com", title="t",
               fetched_date=date(2026, 1, 1), source_type="paper",
               credibility_tier="high", topic="x")
    assert str(s.url).startswith("https://example.com")


def test_source_bad_url():
    with pytest.raises(ValidationError):
        Source(source_id="a", url="not a url", title="t", fetched_date=date(2026, 1, 1),
               source_type="web", credibility_tier="low", topic="x")


def test_chunk_good():
    assert Chunk(chunk_id="k", source_id="a", text="hi").metadata == {}


def test_chunk_empty_text():
    with pytest.raises(ValidationError):
        Chunk(chunk_id="k", source_id="a", text="")


def test_claim_supported_with_cross_check():
    assert claim(strength="supported", cross_check_source_id="s2").strength == "supported"


def test_claim_supported_without_cross_check():
    with pytest.raises(ValidationError):
        claim(strength="supported")


def test_claim_bad_strength():
    with pytest.raises(ValidationError):
        claim(strength="maybe")


def test_insight_good():
    assert len(Insight(insight_id="i", source_id="a", summary="s", claims=[claim()]).claims) == 1


def test_insight_bad_claims():
    with pytest.raises(ValidationError):
        Insight(insight_id="i", source_id="a", summary="s", claims=[{"claim_id": "x"}])


def test_report_draft_good():
    assert ReportDraft(title="t", sections=["a"], claims=[claim()]).created_at


def test_report_draft_missing_title():
    with pytest.raises(ValidationError):
        ReportDraft(sections=[], claims=[])


def test_gate_decision_good():
    assert GateDecision(gate="plan", decision="approved").comment == ""


def test_gate_decision_bad():
    with pytest.raises(ValidationError):
        GateDecision(gate="final", decision="approved")
