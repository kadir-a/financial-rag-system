"""Print PostgreSQL benchmark and conversation statistics."""

import argparse
from pathlib import Path

from sqlalchemy import text

from app.db.database import transaction

SQL_FILE = Path(__file__).resolve().parents[1] / "sql" / "analysis.sql"


def queries():
    for block in SQL_FILE.read_text(encoding="utf-8").split(";"):
        if block.strip():
            title, sql = block.strip().split("\n", 1)
            yield title.removeprefix("-- "), sql


def print_report(db, run_id=None, baseline_id=None):
    for title, sql in queries():
        print(f"\n{title}")
        if ":baseline_id" in sql and (baseline_id is None or run_id is None):
            print("Supply --baseline-id and --run-id to compare two runs.")
            continue
        result = db.execute(text(sql), {"run_id": run_id, "baseline_id": baseline_id})
        rows = [list(result.keys())]
        for row in result:
            rows.append(
                [
                    "NULL"
                    if value is None
                    else f"{value:.2f}"
                    if isinstance(value, float)
                    else str(value)
                    for value in row
                ]
            )
        widths = [max(map(len, column)) for column in zip(*rows)]
        for row in rows:
            print(" | ".join(cell.ljust(width) for cell, width in zip(row, widths)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--baseline-id", type=int)
    args = parser.parse_args()
    if args.baseline_id is not None and args.run_id is None:
        parser.error("--baseline-id needs --run-id")
    with transaction() as db:
        print_report(db, **vars(args))


if __name__ == "__main__":
    main()
