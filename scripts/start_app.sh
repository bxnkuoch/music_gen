#!/usr/bin/env bash
# Start the rating app: database + Python API + web page. Ctrl+C stops everything.
#
# Usage: bash scripts/start_app.sh    then open http://localhost:3000
set -euo pipefail
cd "$(dirname "$0")/.."

docker compose up -d --wait
uv run alembic upgrade head

uv run uvicorn --factory music_gen.api.app:build_app --port 8000 &
API_PID=$!
trap 'kill $API_PID 2>/dev/null' EXIT

if [ ! -d web/node_modules ]; then (cd web && npm install); fi
echo "Rating app: http://localhost:3000   (API docs: http://localhost:8000/docs)"
cd web && npm run dev
