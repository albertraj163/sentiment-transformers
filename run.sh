#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! python3 -c "import streamlit" >/dev/null 2>&1; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

find_free_port() {
  local port=8501
  while lsof -ti:"$port" >/dev/null 2>&1; do
    port=$((port + 1))
  done
  echo "$port"
}

PORT="$(find_free_port)"

echo "Starting Sentiment Transformers UI..."
echo "Open: http://localhost:${PORT}"
exec streamlit run app.py --server.address 0.0.0.0 --server.port "$PORT"
