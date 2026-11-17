#!/bin/zsh
# The detached launcher owns this web process; the product's own switch controls the schedule.
set -e
cd "$(dirname "$0")/.."
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi
export ASHARE_INTERNAL_SCHEDULER=1
exec .venv/bin/python -m ashare serve --host 127.0.0.1 --port 8765
