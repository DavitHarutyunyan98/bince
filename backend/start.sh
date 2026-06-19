#!/usr/bin/env bash
# Start the FastAPI server and RQ worker.
# Run this from the project root: bash backend/start.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export REDIS_URL="${REDIS_URL:-redis://localhost:6379/0}"
export PYTHONPATH="$PROJECT_ROOT:$SCRIPT_DIR"

echo "Starting RQ worker..."
cd "$SCRIPT_DIR"
rq worker optimizer --url "$REDIS_URL" &
WORKER_PID=$!

echo "Starting FastAPI server..."
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &
API_PID=$!

echo "API PID=$API_PID  Worker PID=$WORKER_PID"
echo "Press Ctrl+C to stop both."

trap "kill $API_PID $WORKER_PID 2>/dev/null; exit 0" INT TERM
wait
