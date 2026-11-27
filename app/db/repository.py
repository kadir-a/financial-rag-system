from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.db.models import ChatMessage, ChatSession, Report


def insert_for(db, table):
    return (sqlite_insert if db.get_bind().dialect.name == "sqlite" else pg_insert)(table)


def register_report(db, metadata, refresh=True):
    stmt = insert_for(db, Report).values(**metadata)
    if refresh:
        stmt = stmt.on_conflict_do_update(
            index_elements=["sha256"],
            set_={
                key: value
                for key, value in metadata.items()
                if key != "sha256" and value is not None
            },
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=["sha256"])
    db.execute(stmt)
    return db.scalar(select(Report.id).where(Report.sha256 == metadata["sha256"]))


def token_counts(trace):
    usage = trace.get("chat_usage") or {}
    stages = [usage.get("generation")]
    if usage.get("rewrite") is not None or trace.get("had_history"):
        stages.append(usage.get("rewrite"))

    def total(key):
        if all(stage and stage.get(key) is not None for stage in stages):
            return sum(stage[key] for stage in stages)
        return None

    return {"prompt_tokens": total("input_tokens"), "completion_tokens": total("output_tokens")}


def record_turn(db, metadata, session_id, question, answer, metrics):
    report_id = register_report(db, metadata, refresh=False)
    stmt = insert_for(db, ChatSession).values(id=session_id, report_id=report_id)
    db.execute(stmt.on_conflict_do_nothing(index_elements=["id"]))
    if db.get(ChatSession, session_id).report_id != report_id:
        raise ValueError("Chat session belongs to another report")
    db.add_all(
        [
            ChatMessage(session_id=session_id, role="user", content=question),
            ChatMessage(session_id=session_id, role="assistant", content=answer, **metrics),
        ]
    )
