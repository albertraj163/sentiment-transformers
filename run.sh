#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "Checking dependencies..."
pip install -q -r requirements.txt

find_free_port() {
  local port=8501
  while lsof -ti:"$port" >/dev/null 2>&1; do
    port=$((port + 1))
  done
  echo "$port"
}

PORT="$(find_free_port)"

echo "Starting Sentiment Transformers UI..."
echo "Local app : http://localhost:${PORT}"
echo "GitHub repo: https://github.com/albertraj163/sentiment-transformers"
exec streamlit run app.py --server.address 0.0.0.0 --server.port "$PORT" --server.fileWatcherType none
