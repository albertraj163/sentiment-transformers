#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if ! python3 -c "import streamlit" >/dev/null 2>&1; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

echo "Starting Sentiment Transformers UI..."
echo "Open: http://localhost:8501"
exec streamlit run app.py --server.address 0.0.0.0 --server.port 8501
