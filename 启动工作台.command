#!/bin/zsh
set -e
cd "$(dirname "$0")"
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
if [ ! -x .venv/bin/python ]; then
  print "尚未安装 Python 环境，请按 README.md 安装。"
  read -k 1
  exit 1
fi
if curl --fail --silent http://127.0.0.1:8765/api/health > /dev/null; then
  open http://127.0.0.1:8765
  exit 0
fi
.venv/bin/python -m ashare serve --port 8765 &
ashare_server_pid=$!
trap 'kill "$ashare_server_pid" 2>/dev/null || true' EXIT INT TERM
for ashare_attempt in {1..30}; do
  if curl --fail --silent http://127.0.0.1:8765/api/health > /dev/null; then
    open http://127.0.0.1:8765
    break
  fi
  sleep 0.5
done
wait "$ashare_server_pid"
