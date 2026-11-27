"""Run the FinSight RAG pipeline against verified questions for one report."""

import argparse
import json
import math
import tempfile
from pathlib import Path


def normalize(text):
    return " ".join(text.replace("İ", "i").casefold().split())


def read_cases(path, report, split):
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            if case["report"] != report or case["split"] != split:
                continue
            case_id = case["id"]
            if case_id in seen:
                raise ValueError(f"duplicate id {case_id}")
            for key in ("id", "category", "question", "expected_answer"):
                if not isinstance(case[key], str) or not case[key].strip():
                    raise ValueError(f"{key} must be nonempty text")
            evidence = case["gold_evidence"]
            if not isinstance(evidence, list) or any(
                not isinstance(item, str) or not item.strip() for item in evidence
            ):
                raise ValueError("gold_evidence must contain nonempty text fragments")
            if type(case["answerable"]) is not bool or bool(evidence) != case["answerable"]:
                raise ValueError("supply gold evidence for answerable questions only")
            history = case.get("history", [])
            if not isinstance(history, list) or any(
                not isinstance(turn, list) or len(turn) != 2
                or not all(isinstance(item, str) for item in turn)
                for turn in history
            ):
                raise ValueError("history must contain [role, message] pairs")
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        seen.add(case_id)
        cases.append(case)
    if not cases:
        raise ValueError(f"No questions for report {report!r} and split {split!r}")
    return cases


def evidence_hit(contexts, evidence):
    fragments = [normalize(item) for item in evidence]
    return any(all(fragment in chunk for fragment in fragments)
               for chunk in map(normalize, contexts))


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[math.ceil(fraction * len(ordered)) - 1]


def evaluate(chain, cases):
    rows = []
    for case in cases:
        trace = chain.invoke_with_trace({
            "input": case["question"],
            "chat_history": case.get("history", []),
        })
        rows.append(
            {
                "case": case,
                **trace,
                "evidence_hit": (
                    evidence_hit(
                        trace.get("raw_contexts", trace["contexts"]), case["gold_evidence"]
                    )
                    if case["answerable"]
                    else None
                ),
                "review": {"answer_correct": None, "failure_mode": None},
            }
        )
    return rows


def summarize(rows):
    hits = [row["evidence_hit"] for row in rows if row["evidence_hit"] is not None]
    summary = {
        "question_count": len(rows),
        "evidence_hit_count": sum(hits),
        "evidence_question_count": len(hits),
        "evidence_hit_rate": sum(hits) / len(hits) if hits else None,
    }
    for stage in ("total", "retrieval", "generation"):
        timings = [row["latency_seconds"][stage] for row in rows]
        summary[f"{stage}_p50_seconds"] = percentile(timings, 0.50)
        summary[f"{stage}_p95_seconds"] = percentile(timings, 0.95)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if not args.pdf.is_file() or args.k < 1 or args.output.exists():
        parser.error("Pass an existing PDF, a positive k, and an unused output path.")
    try:
        cases = read_cases(args.dataset, args.pdf.name, args.split)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    from app.services.ingestion import FinancialPDFProcessor
    from app.services.llm_service import LLMService
    from app.services.vector_store import VectorStoreManager

    chunks = FinancialPDFProcessor().process_pdf(str(args.pdf))
    if not chunks:
        parser.error("The PDF parser returned no text.")

    # The Chroma client can keep SQLite open until the process exits on Windows.
    with tempfile.TemporaryDirectory(
        prefix="finsight-eval-", ignore_cleanup_errors=True
    ) as temp_dir:
        manager = VectorStoreManager(base_db_path=temp_dir)
        database = manager.create_and_store_embeddings(chunks, "finsight_eval")
        retriever = database.as_retriever(search_kwargs={"k": args.k})
        service = LLMService()
        rows = evaluate(service.create_rag_chain(retriever), cases)
        result = {
            "report": args.pdf.name,
            "model": service.llm.model_name,
            "embedding_model": manager.embeddings.model,
            "split": args.split,
            "k": args.k,
            "summary": summarize(rows),
            "questions": rows,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(result, output, ensure_ascii=False, indent=2)
            output.write("\n")

    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
