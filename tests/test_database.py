import json
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.database import engine_for
from app.db.models import ChatMessage, ChatSession, EvalResult, EvalRun, Report
from app.db.repository import record_turn, register_report
from eval.import_results import import_results


@pytest.fixture
def db():
    engine = engine_for("sqlite://")
    with engine.connect() as connection:
        config = Config("alembic.ini")
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
        with Session(connection) as session:
            yield session
    engine.dispose()


@pytest.fixture
def metadata():
    return dict(filename="r.pdf", sha256="a" * 64, chroma_collection="vault_a")


def test_repeat_upload_reuses_report(db, metadata):
    first = register_report(db, metadata)
    second = register_report(db, {**metadata, "chroma_collection": "new"})
    assert first == second
    assert db.scalars(select(Report.chroma_collection)).all() == ["new"]


def test_turns_and_cascade(db, metadata):
    sid = uuid4()
    for question in ["Revenue?", "Last year?"]:
        record_turn(db, metadata, sid, question, "100", {"latency_ms": 51})
    db.flush()
    messages = db.scalars(select(ChatMessage).order_by(ChatMessage.id)).all()
    assert [(m.role, m.content) for m in messages] == [
        ("user", "Revenue?"),
        ("assistant", "100"),
        ("user", "Last year?"),
        ("assistant", "100"),
    ]
    assert messages[-1].latency_ms == 51
    db.execute(delete(ChatSession).where(ChatSession.id == sid))
    assert db.scalar(select(func.count()).select_from(ChatMessage)) == 0


def test_foreign_key_enforced(db):
    db.add(ChatSession(report_id=999))
    with pytest.raises(IntegrityError):
        db.flush()


def test_role_check(db, metadata):
    sid = uuid4()
    record_turn(db, metadata, sid, "Q", "A", {})
    db.add(ChatMessage(role="invalid", session_id=sid, content="X"))
    with pytest.raises(IntegrityError):
        db.flush()


def test_unique_collection(db, metadata):
    register_report(db, metadata)
    with pytest.raises(IntegrityError):
        register_report(db, {**metadata, "sha256": "b" * 64})


def test_api_survives_database_outage(monkeypatch, metadata, caplog):
    from unittest.mock import Mock

    from api import main
    from fastapi.testclient import TestClient

    # SQLite opens a real connection and fails on this missing parent directory.
    monkeypatch.setenv("DATABASE_URL", "sqlite:///missing-parent-for-test/outage.sqlite")
    chain = Mock()
    chain.invoke.return_value = "available"
    monkeypatch.setattr(main, "global_state", dict(rag_chain=chain, report_metadata=metadata))
    response = TestClient(main.app).post("/ask/", json={"question": "Q"})
    assert response.status_code == 200
    assert response.json() == {"answer": "available"}
    assert "Database logging failed" in caplog.text


@pytest.fixture
def artifact(tmp_path):
    case = dict(
        report="r.pdf",
        split="dev",
        id="q",
        category="table",
        question="Q",
        expected_answer="A",
        answerable=True,
        gold_evidence=["A"],
    )
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(json.dumps(case))
    source = tmp_path / "run.json"
    source.write_text(
        json.dumps(
            dict(
                report="r.pdf",
                split="dev",
                k=5,
                questions=[
                    dict(
                        case=case,
                        answer="B",
                        evidence_hit=False,
                        raw_contexts=["private"],
                        latency_seconds={"total": 0.31},
                    )
                ],
            )
        )
    )
    return source, dataset, "sample"


def test_import_repeat_and_manual_grade(db, artifact, tmp_path):
    run_id, created, _ = import_results(db, *artifact)
    assert created
    q = db.scalars(select(EvalResult)).one()
    assert (q.answer, q.correct, q.failure_mode, q.retrieval_hit) == ("B", None, None, False)
    assert (q.question_id, q.category, q.answerable, q.latency_ms) == ("q", "table", True, 310)
    grades = tmp_path / "grades.csv"
    grades.write_text("question_id,correct,failure_mode\nq,false,wrong_period\n")
    assert import_results(db, *artifact, grades=grades) == (run_id, False, 1)
    assert db.scalars(select(EvalRun)).one().notes == "sample"
    assert db.scalars(select(EvalResult)).one().failure_mode == "wrong_period"
    assert q.correct is False
    db.execute(delete(EvalRun))
    assert db.scalar(select(func.count()).select_from(EvalResult)) == 0


@pytest.mark.parametrize("bad", ["duplicate", "invalid_failure"])
def test_result_constraints(db, artifact, bad):
    run_id, _, _ = import_results(db, *artifact)
    db.add(
        EvalResult(
            run_id=run_id,
            question_id="q" if bad == "duplicate" else "other",
            answerable=True,
            category="table",
            answer="X",
            failure_mode="fake" if bad == "invalid_failure" else None,
        )
    )
    with pytest.raises(IntegrityError):
        db.flush()


def test_api_keeps_turns_in_one_chat_and_separates_new_clients(tmp_path, monkeypatch, metadata):
    from api import main
    from fastapi.testclient import TestClient

    class Chain:
        def invoke_with_trace(self, payload):
            return dict(
                answer="A",
                search_query="standalone",
                contexts=["sensitive"],
                chat_usage={
                    "generation": {"input_tokens": 5, "output_tokens": 3},
                    "rewrite": {"input_tokens": 2, "output_tokens": 1},
                },
            )

    url = "sqlite:///" + (tmp_path / "api.sqlite").as_posix()
    monkeypatch.setenv("DATABASE_URL", url)
    command.upgrade(Config("alembic.ini"), "head")
    monkeypatch.setattr(main, "global_state", dict(rag_chain=Chain(), report_metadata=metadata))
    with TestClient(main.app) as client:
        client.post("/ask/", json={"question": "Q"})
        response = client.post(
            "/ask/", json={"question": "Again?", "chat_history": [["user", "Q"]]}
        )
        assert response.json() == {"answer": "A"}
    with TestClient(main.app) as other:
        other.post("/ask/", json={"question": "New"})
    engine = engine_for(url)
    with Session(engine) as db:
        turns = db.scalars(
            select(ChatMessage).where(ChatMessage.role == "assistant").order_by(ChatMessage.id)
        ).all()
        assert len(turns) == 3
        first, second, third = turns
        assert first.session_id == second.session_id != third.session_id
        assert (second.rewritten_query, second.prompt_tokens, second.completion_tokens) == (
            "standalone",
            7,
            4,
        )
        assert second.latency_ms >= 0
    engine.dispose()
