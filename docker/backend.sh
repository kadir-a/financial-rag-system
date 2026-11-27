#!/bin/sh
if ! alembic upgrade head; then
    echo "WARNING: database migration unavailable; serving without guaranteed persistence" >&2
fi
exec uvicorn api.main:app --app-dir app --host 0.0.0.0 --port 8000
