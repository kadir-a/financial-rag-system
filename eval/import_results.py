"""Import one FinSight evaluation artifact and optional manually reviewed grades."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from sqlalchemy import select

from app.db.database import transaction
from app.db.models import FAILURE_MODES, EvalResult, EvalRun
from app.db.repository import insert_for, token_counts
from eval.evaluate import read_cases


def read_grades(path):
    grades = {}
    if path is None:
        return grades
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["question_id", "correct", "failure_mode"]:
            raise ValueError("Expected CSV header: question_id,correct,failure_mode")
        for row in reader:
            qid = row["question_id"].strip()
            value = row["correct"].strip().lower()
            mode = row["failure_mode"].strip() or None
            if not qid or qid in grades or value not in ("true", "false", ""):
                raise ValueError(f"Invalid or duplicate grade for {qid!r}")
            if mode and (mode not in FAILURE_MODES or value == "true"):
                raise ValueError(f"Invalid failure mode for {qid}")
            grades[qid] = ({"true": True, "false": False, "": None}[value], mode)
    return grades


def import_results(db, source, dataset, run_name, grades=None):
    raw = source.read_bytes()
    artifact = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    cases = {q["id"]: q for q in read_cases(dataset, artifact["report"], artifact["split"])}
    manual = read_grades(grades)
    stmt = (
        insert_for(db, EvalRun)
        .values(
            source_sha256=digest,
            dataset_name=dataset.name,
            split=artifact["split"],
            k=artifact["k"],
            model=artifact.get("model"),
            embedding_model=artifact.get("embedding_model"),
            git_commit=artifact.get("git_commit"),
            notes=run_name,
        )
        .on_conflict_do_nothing(index_elements=["source_sha256"])
        .returning(EvalRun.id)
    )
    run_id = db.scalar(stmt)
    created = run_id is not None
    if created:
        seen = set()
        for row in artifact["questions"]:
            qid = row["case"]["id"]
            if qid not in cases or qid in seen:
                raise ValueError(f"Unknown or duplicate question {qid!r}")
            seen.add(qid)
            case = cases[qid]
            review = row.get("review") or {}
            seconds = (row.get("latency_seconds") or {}).get("total")
            db.add(
                EvalResult(
                    run_id=run_id,
                    question_id=qid,
                    category=case["category"],
                    answerable=case["answerable"],
                    answer=row["answer"],
                    correct=review.get("answer_correct", row.get("correct")),
                    failure_mode=review.get("failure_mode", row.get("failure_mode")),
                    retrieval_hit=row.get("evidence_hit") if case["answerable"] else None,
                    latency_ms=None if seconds is None else seconds * 1000,
                    **token_counts({**row, "had_history": bool(case.get("history"))}),
                )
            )
        if not seen:
            raise ValueError("Result file contains no questions")
        db.flush()
    else:
        run_id = db.scalar(select(EvalRun.id).where(EvalRun.source_sha256 == digest))
    applied = 0
    if manual:
        for result in db.scalars(select(EvalResult).where(EvalResult.run_id == run_id)):
            if result.question_id in manual:
                result.correct, result.failure_mode = manual[result.question_id]
                applied += 1
    return run_id, created, applied


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--grades", type=Path)
    args = parser.parse_args()
    with transaction() as db:
        run_id, created, applied = import_results(db, **vars(args))
    status = "Imported" if created else "Already imported (same file)"
    print(f"{status}: run {run_id}; {applied} matching manual grades applied.")


if __name__ == "__main__":
    main()
