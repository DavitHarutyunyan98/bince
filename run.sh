#!/usr/bin/env bash
# One-command launcher for the single-process Dash dashboard (main.py).
# Creates a virtualenv on first run, installs deps, then starts the app.
set -e

cd "$(dirname "$0")"

VENV="${VENV:-.venv}"
if [ ! -d "$VENV" ]; then
  echo "Creating virtualenv in $VENV ..."
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install -r requirements-app.txt
fi

echo "Starting dashboard on http://0.0.0.0:${APP_PORT:-8080}"
exec "$VENV/bin/python" main.py
