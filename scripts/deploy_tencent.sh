#!/usr/bin/env bash
set -euo pipefail

DEPLOY_HOST="${DEPLOY_HOST:?Set DEPLOY_HOST to the new Lighthouse public IP}"
DEPLOY_PORT="${DEPLOY_PORT:-22}"
DEPLOY_USER="${DEPLOY_USER:-ubuntu}"
REMOTE_DIR="${REMOTE_DIR:-/home/$DEPLOY_USER/a-share-investment}"
APP_SITE="${APP_SITE:-}"
LOCAL_PYTHON="${LOCAL_PYTHON:-.venv/bin/python}"
SSH_IDENTITY="${SSH_IDENTITY:-}"
if [[ -z "$SSH_IDENTITY" && -f "$HOME/.ssh/a_share_investment_tencent" ]]; then
  SSH_IDENTITY="$HOME/.ssh/a_share_investment_tencent"
fi
SSH=(ssh -p "$DEPLOY_PORT" -o BatchMode=yes -o ConnectTimeout=10)
if [[ -n "$SSH_IDENTITY" ]]; then
  SSH+=(-i "$SSH_IDENTITY" -o IdentitiesOnly=yes)
fi
printf -v RSYNC_SSH '%q ' "${SSH[@]}"
RSYNC_SSH="${RSYNC_SSH% }"
SSH+=("$DEPLOY_USER@$DEPLOY_HOST")

# After a domain has been bound, ordinary code updates keep that hostname.
if [[ -z "$APP_SITE" ]]; then
  APP_SITE="$("${SSH[@]}" "if test -f '$REMOTE_DIR/.env'; then sed -n 's/^APP_SITE=//p' '$REMOTE_DIR/.env' | head -1; fi")"
  APP_SITE="${APP_SITE:-https://$DEPLOY_HOST}"
fi
if [[ ! "$APP_SITE" =~ ^https://[A-Za-z0-9.-]+$|^[A-Za-z0-9.-]+$ ]]; then
  echo "APP_SITE must be a domain or https://IP" >&2
  exit 2
fi

"${SSH[@]}" "mkdir -p '$REMOTE_DIR/data/inbox' '$REMOTE_DIR/runs'"
rsync -az --delete \
  --exclude '.venv' --exclude '.pytest_cache' --exclude 'data' --exclude 'runs' \
  --exclude '.env' --exclude 'deployment-credentials.txt' \
  --exclude 'caddy_data' --exclude 'caddy_config' \
  --exclude '.DS_Store' --exclude '__pycache__' \
  -e "$RSYNC_SSH" ./ "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_DIR/"
rsync -az --delete -e "$RSYNC_SSH" data/inbox/ "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_DIR/data/inbox/"

# Seed forward state and the latest audited result only on a new server.  Once
# the cloud scheduler has started, its account and latest pointer are canonical
# and must never be replaced by an older laptop copy during a code update.
if [[ -d data/live ]] && ! "${SSH[@]}" "test -f '$REMOTE_DIR/data/live/current/manifest.json'"; then
  rsync -az -e "$RSYNC_SSH" data/live/ "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_DIR/data/live/"
fi
if ! "${SSH[@]}" "test -f '$REMOTE_DIR/runs/latest.json'"; then
  LATEST_RUN="$("$LOCAL_PYTHON" -c 'import json; print(json.load(open("runs/latest.json"))["run_id"])')"
  "${SSH[@]}" "mkdir -p '$REMOTE_DIR/runs/$LATEST_RUN'"
  rsync -az -e "$RSYNC_SSH" \
    --exclude 'oos_predictions.parquet' \
    "runs/$LATEST_RUN/" "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_DIR/runs/$LATEST_RUN/"
  rsync -az -e "$RSYNC_SSH" runs/latest.json "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_DIR/runs/latest.json"
fi
# A daily paper report has no historical equity curve.  Keep the latest audited
# real-data backtest separately available for the cloud research page without
# changing the paper account or the latest forward-run pointer.
RESEARCH_RUN="$("$LOCAL_PYTHON" -c 'import json,pathlib; paths=sorted(pathlib.Path("runs").glob("*/report.json"),reverse=True); match=next((p for p in paths if (lambda r:r.get("kind")=="research" and r.get("mode")=="real_research" and bool(r.get("curve")))(json.loads(p.read_text()))),None); print(match.parent.name if match else "")')"
if [[ -n "$RESEARCH_RUN" ]]; then
  "${SSH[@]}" "mkdir -p '$REMOTE_DIR/runs/$RESEARCH_RUN'"
  for artifact in report.json equity_curve.csv manifest.json; do
    if [[ -f "runs/$RESEARCH_RUN/$artifact" ]]; then
      rsync -az -e "$RSYNC_SSH" "runs/$RESEARCH_RUN/$artifact" "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_DIR/runs/$RESEARCH_RUN/$artifact"
    fi
  done
fi
# Keep the newest point-in-time replay reviewable on the server as well. It is
# an additional artifact and never changes the live paper account pointer.
REPLAY_RUN="$("$LOCAL_PYTHON" -c 'import json,pathlib; paths=sorted(pathlib.Path("runs").glob("replay-*/report.json"),reverse=True); match=next((p for p in paths if json.loads(p.read_text()).get("kind")=="point_in_time_replay"),None); print(match.parent.name if match else "")')"
if [[ -n "$REPLAY_RUN" ]]; then
  "${SSH[@]}" "mkdir -p '$REMOTE_DIR/runs/$REPLAY_RUN'"
  for artifact in report.json summary.md equity_curve.csv executions.csv scores.parquet manifest.json config.json; do
    if [[ -f "runs/$REPLAY_RUN/$artifact" ]]; then
      rsync -az -e "$RSYNC_SSH" "runs/$REPLAY_RUN/$artifact" "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_DIR/runs/$REPLAY_RUN/$artifact"
    fi
  done
fi
if [[ -f runs/paper_schedule.json ]] && ! "${SSH[@]}" "test -f '$REMOTE_DIR/runs/paper_schedule.json'"; then
  SEED_SCHEDULE="$(mktemp)"
  trap 'rm -f "$SEED_SCHEDULE"' EXIT
  "$LOCAL_PYTHON" -c 'import json,sys; source=json.load(open(sys.argv[1])); print(json.dumps({"last_success_signal":source["last_success_signal"],"run_id":source["run_id"],"last_result":"seeded"}))' runs/paper_schedule.json > "$SEED_SCHEDULE"
  rsync -az -e "$RSYNC_SSH" "$SEED_SCHEDULE" "$DEPLOY_USER@$DEPLOY_HOST:$REMOTE_DIR/runs/paper_schedule.json"
fi

"${SSH[@]}" "cd '$REMOTE_DIR' && chmod +x deploy/remote_setup.sh && APP_SITE='$APP_SITE' DEPLOY_DIR='$REMOTE_DIR' ./deploy/remote_setup.sh"
echo "Deployment complete. Retrieve the one-time login with:"
echo "  ssh -p $DEPLOY_PORT $DEPLOY_USER@$DEPLOY_HOST 'cat $REMOTE_DIR/deployment-credentials.txt'"
