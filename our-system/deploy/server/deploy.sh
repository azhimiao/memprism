#!/usr/bin/env bash
# Deploy AML memory service on Ubuntu host (Docker + compose overlays). No secrets on CLI.
set -euo pipefail

CODE_ROOT="${AML_CODE_ROOT:-/opt/aml}"
ENV_FILE="${AML_ENV_FILE:-/etc/aml/aml.env}"
IMAGE_CURRENT="aml-cycle2-v4:server"
IMAGE_PREVIOUS="aml-cycle2-v4:server-previous"
HEALTH_URL="http://127.0.0.1:28717/health"
HEALTH_RETRIES=60
HEALTH_SLEEP=2

usage() {
  cat <<'EOF'
Usage:
  deploy.sh --ref <git-ref>     Clone/pull repo under /opt/aml, build and up
  deploy.sh --ref <path.tar.gz> Extract tarball into /opt/aml, build and up
  deploy.sh --rollback          Restore previous image tag and recreate container

Environment:
  AML_CODE_ROOT   Default /opt/aml
  AML_ENV_FILE    Default /etc/aml/aml.env (compose env_file)
EOF
}

log() { printf '[deploy] %s\n' "$*"; }
die() { log "ERROR: $*"; exit 1; }

resolve_our_system() {
  if [[ -f "${CODE_ROOT}/our-system/deploy/compose.yaml" ]]; then
    echo "${CODE_ROOT}/our-system"
  elif [[ -f "${CODE_ROOT}/deploy/compose.yaml" ]]; then
    echo "${CODE_ROOT}"
  else
    die "Cannot find deploy/compose.yaml under ${CODE_ROOT}"
  fi
}

materialize_ref() {
  local ref="$1"
  if [[ -f "${ref}" ]] && [[ "${ref}" == *.tar.gz || "${ref}" == *.tgz || "${ref}" == *.tar ]]; then
    log "Extracting archive ${ref} -> ${CODE_ROOT}"
    install -d -m 0755 "${CODE_ROOT}"
    tar -xf "${ref}" -C "${CODE_ROOT}"
    return 0
  fi
  if [[ -d "${CODE_ROOT}/.git" ]]; then
    log "Updating git checkout in ${CODE_ROOT} (ref=${ref})"
    git -C "${CODE_ROOT}" fetch --all --tags
    git -C "${CODE_ROOT}" checkout "${ref}"
    git -C "${CODE_ROOT}" pull --ff-only 2>/dev/null || true
    return 0
  fi
  if [[ -n "${ref}" ]]; then
    log "Cloning into ${CODE_ROOT} (ref=${ref})"
    install -d -m 0755 "$(dirname "${CODE_ROOT}")"
    git clone --depth 1 --branch "${ref}" "${AML_GIT_REMOTE:-https://github.com/REPLACE_ORG/REPLACE_REPO.git}" "${CODE_ROOT}" \
      || git clone "${AML_GIT_REMOTE:-https://github.com/REPLACE_ORG/REPLACE_REPO.git}" "${CODE_ROOT}" \
      && git -C "${CODE_ROOT}" checkout "${ref}"
    return 0
  fi
  die "--ref is required unless using --rollback"
}

compose_cmd() {
  local our_system="$1"
  shift
  (cd "${our_system}" && AML_ENV_FILE="${ENV_FILE}" docker compose \
    -f deploy/compose.yaml -f deploy/server/compose.server.yaml "$@")
}

tag_previous_image() {
  if docker image inspect "${IMAGE_CURRENT}" >/dev/null 2>&1; then
    docker tag "${IMAGE_CURRENT}" "${IMAGE_PREVIOUS}"
    log "Tagged ${IMAGE_CURRENT} -> ${IMAGE_PREVIOUS}"
  fi
}

wait_health() {
  local i payload
  for ((i = 1; i <= HEALTH_RETRIES; i++)); do
    if payload="$(curl -fsS "${HEALTH_URL}" 2>/dev/null)" \
      && [[ "${payload}" == *'"status":"ok"'* ]] \
      && [[ "${payload}" == *'"control_ready":true'* ]] \
      && [[ "${payload}" == *'"data_route_ready":true'* ]]; then
      log "Health OK (${HEALTH_URL})"
      return 0
    fi
    sleep "${HEALTH_SLEEP}"
  done
  return 1
}

on_failure() {
  local our_system="$1"
  log "Deployment failed; last 100 log lines:"
  compose_cmd "${our_system}" logs --tail 100 || true
  exit 1
}

do_deploy() {
  local ref="${1:-}"
  [[ -f "${ENV_FILE}" ]] || die "Missing ${ENV_FILE}; copy deploy/server/env.server.example and fill secrets."
  local our_system
  our_system="$(resolve_our_system)"
  if [[ -n "${ref}" ]]; then
    materialize_ref "${ref}"
    our_system="$(resolve_our_system)"
  elif [[ ! -f "${our_system}/deploy/compose.yaml" ]]; then
    die "No code in ${CODE_ROOT}; pass --ref <git-ref|tarball>"
  fi
  tag_previous_image
  log "Building image (context ${our_system})..."
  compose_cmd "${our_system}" build
  log "Starting container..."
  compose_cmd "${our_system}" up -d
  if ! wait_health; then
    on_failure "${our_system}"
  fi
  log "Deploy complete."
}

do_rollback() {
  docker image inspect "${IMAGE_PREVIOUS}" >/dev/null 2>&1 \
    || die "No ${IMAGE_PREVIOUS} image; run a successful deploy first."
  docker tag "${IMAGE_PREVIOUS}" "${IMAGE_CURRENT}"
  local our_system
  our_system="$(resolve_our_system)"
  compose_cmd "${our_system}" up -d
  if ! wait_health; then
    on_failure "${our_system}"
  fi
  log "Rollback complete."
}

main() {
  local ref="" rollback=0
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --ref)
        shift
        [[ $# -gt 0 ]] || die "--ref requires a value"
        ref="$1"
        shift
        ;;
      --rollback) rollback=1; shift ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown argument: $1" ;;
    esac
  done
  command -v docker >/dev/null 2>&1 || die "docker not installed; run bootstrap-ubuntu.sh"
  if [[ "${rollback}" -eq 1 ]]; then
    do_rollback
  else
    do_deploy "${ref}"
  fi
}

main "$@"
