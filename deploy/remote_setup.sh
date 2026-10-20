#!/usr/bin/env bash
set -euo pipefail

cd "${DEPLOY_DIR:-$HOME/a-share-investment}"
APP_SITE_VALUE="${APP_SITE:-}"
if [[ -z "$APP_SITE_VALUE" || ! "$APP_SITE_VALUE" =~ ^https://[A-Za-z0-9.-]+$|^[A-Za-z0-9.-]+$ ]]; then
  echo "APP_SITE must be a domain or https://IP" >&2
  exit 2
fi
APP_HOST_VALUE="${APP_SITE_VALUE#https://}"
APP_HOST_VALUE="${APP_HOST_VALUE#http://}"
if [[ "$APP_HOST_VALUE" =~ ^[0-9.]+$ ]]; then
  CADDYFILE_VALUE="./deploy/Caddyfile.ip"
else
  CADDYFILE_VALUE="./deploy/Caddyfile.domain"
fi

as_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo "$@"
  else
    echo "This 2GB deployment needs passwordless sudo once to create /swapfile." >&2
    return 1
  fi
}

docker_cmd() {
  if docker info >/dev/null 2>&1; then
    docker "$@"
  else
    as_root docker "$@"
  fi
}

ensure_budget_swap() {
  local mem_kb swap_kb
  mem_kb="$(awk '/MemTotal:/ {print $2}' /proc/meminfo)"
  swap_kb="$(awk '/SwapTotal:/ {print $2}' /proc/meminfo)"
  if (( mem_kb >= 3145728 || swap_kb >= 3145728 )); then
    return
  fi
  echo "Low-memory server detected; ensuring a persistent 4GB swap file."
  if ! awk '$1=="/swapfile" {found=1} END {exit !found}' /proc/swaps; then
    if [[ ! -f /swapfile ]]; then
      as_root fallocate -l 4G /swapfile || as_root dd if=/dev/zero of=/swapfile bs=1M count=4096 status=progress
    fi
    as_root chmod 600 /swapfile
    as_root mkswap /swapfile >/dev/null
    as_root swapon /swapfile
  fi
  if ! grep -qF '/swapfile none swap sw 0 0' /etc/fstab; then
    printf '%s\n' '/swapfile none swap sw 0 0' | as_root tee -a /etc/fstab >/dev/null
  fi
  printf '%s\n' 'vm.swappiness=15' | as_root tee /etc/sysctl.d/99-ashare-memory.conf >/dev/null
  as_root sysctl vm.swappiness=15 >/dev/null
}

ensure_budget_swap

mkdir -p data/inbox data/live runs caddy_data caddy_config
if [[ ! -f .env ]]; then
  umask 077
  PASSWORD="$(openssl rand -base64 18 | tr -d '/+=' | cut -c1-20)"
  HASH="$(docker_cmd run --rm caddy:2.10-alpine caddy hash-password --plaintext "$PASSWORD")"
  cat > .env <<EOF
APP_SITE=$APP_SITE_VALUE
APP_HOST=$APP_HOST_VALUE
CADDYFILE=$CADDYFILE_VALUE
APP_USER=investor
APP_PASSWORD_HASH='$HASH'
APP_UID=$(id -u)
APP_GID=$(id -g)
AI_MODEL=
AI_BASE_URL=
AI_API_KEY=
EOF
  cat > deployment-credentials.txt <<EOF
URL: https://$APP_HOST_VALUE
Username: investor
Password: $PASSWORD
EOF
  chmod 600 .env deployment-credentials.txt
else
  # A hostname change must preserve the existing password hash, AI settings,
  # and paper-trading state. Update only the three address-related entries.
  sed -i "s|^APP_SITE=.*|APP_SITE=$APP_SITE_VALUE|" .env
  sed -i "s|^APP_HOST=.*|APP_HOST=$APP_HOST_VALUE|" .env
  sed -i "s|^CADDYFILE=.*|CADDYFILE=$CADDYFILE_VALUE|" .env
  chmod 600 .env
  if [[ -f deployment-credentials.txt ]]; then
    sed -i "s|^URL: .*|URL: https://$APP_HOST_VALUE|" deployment-credentials.txt
    chmod 600 deployment-credentials.txt
  fi
fi

docker_cmd compose build --pull
docker_cmd compose up -d --remove-orphans
docker_cmd compose ps
echo
echo "Credentials are stored in: $(pwd)/deployment-credentials.txt"
