import logging
import os
from contextlib import contextmanager
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

log = logging.getLogger(__name__)


@lru_cache(maxsize=4)
def engine_for(url):
    dialect = make_url(url).get_backend_name()
    options = {"connect_timeout": 2} if dialect == "postgresql" else {}
    engine = create_engine(url, connect_args=options, pool_pre_ping=True, hide_parameters=True)
    if dialect == "sqlite":

        @event.listens_for(engine, "connect")
        def enable_foreign_keys(connection, _record):
            connection.execute("PRAGMA foreign_keys=ON")

    return engine


def database_url():
    load_dotenv()
    return os.environ["DATABASE_URL"]


@contextmanager
def transaction():
    with Session(engine_for(database_url())) as db, db.begin():
        yield db


def best_effort(operation, *args):
    if not os.environ.get("DATABASE_URL"):
        return None
    try:
        with transaction() as db:
            return operation(db, *args)
    except Exception as exc:
        # Driver errors can include SQL parameters or credentials.
        log.warning("Database logging failed in %s (%s)", operation.__name__, type(exc).__name__)
        return None
