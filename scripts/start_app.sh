#!/usr/bin/env bash
# Start the whole app: database + Python API + generation worker + web page.
# Ctrl+C stops everything.
#
# Usage: bash scripts/start_app.sh    then open http://localhost:3000
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose up -d --wait
uv run alembic upgrade head

uv run uvicorn --factory music_gen.api.app:build_app --port 8000 &
API_PID=$!
# The worker holds the music model (~10 GB) and generates songs for "Create new".
uv run python scripts/worker.py &
WORKER_PID=$!
trap 'kill $API_PID $WORKER_PID 2>/dev/null' EXIT

if [ ! -d web/node_modules ]; then (cd web && npm install); fi
echo "App: http://localhost:3000   (API docs: http://localhost:8000/docs)"
echo "The worker takes ~30 s to load the music model before 'Create new' works."
cd web && npm run dev
