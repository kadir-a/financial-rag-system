from alembic import context

from app.db.database import database_url, engine_for
from app.db.models import Base


def migrate(connection):
    context.configure(connection=connection, target_metadata=Base.metadata)
    with context.begin_transaction():
        context.run_migrations()


provided = context.config.attributes.get("connection")
if provided is not None:
    migrate(provided)
else:
    with engine_for(database_url()).connect() as connection:
        migrate(connection)
