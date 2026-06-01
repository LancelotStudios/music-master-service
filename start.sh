#!/usr/bin/env bash
# Run the mastering service locally. First run sets up a venv + installs deps.
# Needs ffmpeg on your machine (brew install ffmpeg).
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Setting up .venv (first run)…"
  python3 -m venv .venv
  ./.venv/bin/pip install --upgrade pip
  ./.venv/bin/pip install -r requirements.txt
fi

export PORT="${PORT:-8000}"
export MASTER_TOKEN="${MASTER_TOKEN:-local-dev-token}"
echo "Mastering service on http://localhost:$PORT  (token: $MASTER_TOKEN)"
exec ./.venv/bin/uvicorn app:app --host 0.0.0.0 --port "$PORT" --reload
