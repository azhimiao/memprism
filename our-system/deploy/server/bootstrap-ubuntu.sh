#!/usr/bin/env bash
# Idempotent host bootstrap for AML Docker + nginx + TLS tooling. Requires sudo.
set -euo pipefail

log() { printf '[bootstrap] %s\n' "$*"; }

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  exec sudo -E bash "$0" "$@"
fi

export DEBIAN_FRONTEND=noninteractive

apt-get update -qq
# --no-upgrade leaves already-installed packages alone; an existing nginx (any origin) is reused, not replaced.
packages=(ca-certificates curl gnupg lsb-release ufw certbot sqlite3 git)
if ! command -v nginx >/dev/null 2>&1; then
  packages+=(nginx)
else
  log "nginx already present ($(nginx -v 2>&1)); reusing it."
fi
apt-get install -y -qq --no-upgrade "${packages[@]}"

if ! command -v docker >/dev/null 2>&1; then
  log 'Installing Docker Engine (official apt repository)...'
  install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
  chmod a+r /etc/apt/keyrings/docker.gpg
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${codename} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -qq
  apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
  systemctl enable --now docker
else
  log 'Docker already installed; ensuring compose plugin...'
  apt-get install -y -qq docker-compose-plugin 2>/dev/null || true
fi

install -d -m 0755 /opt/aml
# The container user (uid/gid 10001 in deploy/Dockerfile) writes the SQLite database through the bind mount.
install -d -m 0750 -o 10001 -g 10001 /var/lib/aml
install -d -m 0700 /etc/aml
if [[ ! -f /etc/aml/aml.env ]]; then
  touch /etc/aml/aml.env
  chmod 600 /etc/aml/aml.env
fi
chown root:root /etc/aml/aml.env

# Existing rules and defaults on a shared host are left as they are; only these three are added.
ufw allow 22/tcp comment 'SSH'
ufw allow 80/tcp comment 'HTTP (ACME + redirect)'
ufw allow 443/tcp comment 'HTTPS'
if [[ "${AML_ENABLE_UFW:-0}" == "1" ]]; then
  ufw --force enable
else
  log "ufw rules added; firewall state unchanged ($(ufw status | head -n 1)). Set AML_ENABLE_UFW=1 to enable it."
fi

log 'Done. Next: write /etc/aml/aml.env, DNS, deploy.sh, nginx + certbot.'
