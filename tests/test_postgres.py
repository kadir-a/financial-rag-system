import os
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from app.db.database import engine_for
from app.db.models import EvalResult, EvalRun
from eval.report import print_report

pytestmark = pytest.mark.postgres


def test_postgres_percentiles_and_comparison(capsys):
    url = os.environ.get("DATABASE_URL", "")
    if not url.startswith("postgresql"):
        pytest.skip("Set DATABASE_URL to a migrated PostgreSQL database")
    with Session(engine_for(url)) as db:
        runs = []
        for correct in (False, True):
            run = EvalRun(
                source_sha256=uuid4().hex, dataset_name="fixture", split="dev", notes="demo", k=5
            )
            db.add(run)
            db.flush()
            for question, ms in (("q1", 100), ("q2", 300)):
                db.add(
                    EvalResult(
                        run_id=run.id,
                        question_id=question,
                        category="table",
                        answerable=True,
                        correct=correct,
                        retrieval_hit=True,
                        answer="after" if correct else "before",
                        latency_ms=ms,
                    )
                )
            runs.append(run.id)
        db.flush()
        print_report(db, run_id=runs[1], baseline_id=runs[0])
        output = capsys.readouterr().out
        assert "200.00" in output and "290.00" in output
        assert any(
            line.split() == ["q1", "|", "before", "|", "after"] for line in output.splitlines()
        )
        assert "100.00" in output
