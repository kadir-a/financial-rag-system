import json
from unittest.mock import Mock

import pytest

from eval.evaluate import evaluate, read_cases, summarize

CASE = dict(id="a", category="value", report="a.pdf", split="dev", question="Gelir?",
            expected_answer="42", answerable=True, gold_evidence=["gelir", "42"])


@pytest.mark.parametrize("missing", list(CASE))
def test_required_fields(tmp_path, missing):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps(
        {k: v for k, v in CASE.items() if k != missing}), encoding="utf-8")
    with pytest.raises(ValueError, match=missing):
        read_cases(dataset, "a.pdf", "dev")


def test_trace_and_hit_denominator_without_token_counts():
    chain = Mock()
    timing = dict(total=3, retrieval=1, generation=2)
    chain.invoke_with_trace.side_effect = [
        dict(contexts=["aligned"], raw_contexts=chunks, latency_seconds=timing, answer="result")
        for chunks in [["GELİR   42"], ["gelir", "42"], ["other"]]
    ]
    rows = evaluate(chain, [CASE, CASE, dict(CASE, answerable=False, gold_evidence=[])])
    assert rows[0]["contexts"] == ["aligned"]
    assert [row["evidence_hit"] for row in rows] == [True, False, None]
    summary = summarize(rows)
    assert summary["evidence_question_count"] == 2
    assert summary["evidence_hit_rate"] == 0.5


def test_latency_nearest_rank():
    rows = [dict(evidence_hit=None, latency_seconds=dict(total=n, retrieval=n/2, generation=n/4))
            for n in range(21, 0, -1)]
    summary = summarize(rows)
    for stage, factor in [("total", 1), ("retrieval", 2), ("generation", 4)]:
        assert summary[f"{stage}_p50_seconds"] == 11 / factor
        assert summary[f"{stage}_p95_seconds"] == 20 / factor
