#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

GITHUB_URL="https://github.com/albertraj163/sentiment-transformers"
DEFAULT_PORT=8501

echo "Checking dependencies..."
pip install -q -r requirements.txt

find_free_port() {
  local port=$DEFAULT_PORT
  while lsof -ti:"$port" >/dev/null 2>&1; do
    port=$((port + 1))
  done
  echo "$port"
}

PORT="$(find_free_port)"
LOCAL_URL="http://localhost:${PORT}"
NETWORK_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
NETWORK_URL=""
if [[ -n "$NETWORK_IP" ]]; then
  NETWORK_URL="http://${NETWORK_IP}:${PORT}"
fi

echo ""
echo "=========================================="
echo " Sentiment Transformers — Local Server"
echo "=========================================="
echo " Local app   : ${LOCAL_URL}"
if [[ -n "$NETWORK_URL" ]]; then
  echo " Network app : ${NETWORK_URL}"
fi
echo " GitHub repo : ${GITHUB_URL}"
echo "=========================================="
echo " Open the Local app link in your browser."
echo ""

exec streamlit run app.py \
  --server.address 0.0.0.0 \
  --server.port "$PORT" \
  --server.fileWatcherType none \
  --server.headless true
