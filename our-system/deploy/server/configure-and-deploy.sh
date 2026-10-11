#!/usr/bin/env bash
# Write /etc/aml/aml.env from env.server.example, deploy, and check the loopback service.
# remote-configure.ps1 streams this file over ssh with the DashScope key in AML_DASHSCOPE_KEY_INPUT;
# run directly on the host, it asks for the key without echo.
# Keys are never printed, passed as process arguments, or written outside /etc/aml/aml.env.
set -euo pipefail

main() {
  local root=/opt/aml/our-system
  local example="${root}/deploy/server/env.server.example"
  local env_file=/etc/aml/aml.env
  local base=http://127.0.0.1:28717
  local dashscope_key="${AML_DASHSCOPE_KEY_INPUT:-}" service_key="" line tmp code health
  unset AML_DASHSCOPE_KEY_INPUT

  [[ "${EUID:-$(id -u)}" -eq 0 ]] || { echo '[configure] run as root' >&2; return 1; }
  [[ -f "${example}" ]] || { echo "[configure] missing ${example}; unpack the release first" >&2; return 1; }
  if [[ -z "${dashscope_key}" ]]; then
    if ! read -r -s -p 'DashScope API key (input hidden): ' dashscope_key </dev/tty; then
      echo '[configure] no DashScope key supplied' >&2
      return 1
    fi
    echo
  fi
  [[ "${dashscope_key}" =~ ^[A-Za-z0-9._-]{8,256}$ ]] \
    || { echo '[configure] DashScope key has an unexpected format' >&2; return 1; }

  # A key already handed to the platform must survive re-runs.
  if [[ -f "${env_file}" ]]; then
    service_key="$(sed -n 's/^AML_SERVICE_API_KEY=//p' "${env_file}" | head -n 1)"
  fi
  if [[ "${service_key}" =~ ^[A-Za-z0-9._-]{16,256}$ ]]; then
    echo '[configure] keeping the existing service key'
  else
    service_key="$(openssl rand -hex 32)"
    echo '[configure] generated a new service key'
  fi

  install -d -m 0700 /etc/aml
  tmp="$(mktemp /etc/aml/.aml.env.XXXXXX)"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    case "${line}" in
      AML_SERVICE_API_KEY=*) printf 'AML_SERVICE_API_KEY=%s\n' "${service_key}" ;;
      DASHSCOPE_API_KEY=*) printf 'DASHSCOPE_API_KEY=%s\n' "${dashscope_key}" ;;
      *) printf '%s\n' "${line}" ;;
    esac
  done <"${example}" >"${tmp}"
  chmod 600 "${tmp}"
  chown root:root "${tmp}"
  mv -f "${tmp}" "${env_file}"
  unset dashscope_key
  echo "[configure] wrote ${env_file} (mode $(stat -c %a "${env_file}"))"

  AML_ENV_FILE="${env_file}" bash "${root}/deploy/server/deploy.sh"

  health="$(curl -fsS "${base}/health")"
  echo "[configure] health: ${health}"
  code="$(curl -s -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{}' "${base}/search")"
  echo "[configure] search without key: ${code} (expect 401)"
  code="$(printf 'header = "Authorization: Bearer %s"\n' "${service_key}" \
    | curl -s -o /dev/null -w '%{http_code}' -K - -H 'Content-Type: application/json' \
      -d '{"query":"probe","user_id":"selftest-0","top_k":1}' "${base}/search")"
  unset service_key
  echo "[configure] search with key: ${code} (expect 200; this also exercises the DashScope key)"
  [[ "${code}" == 200 ]]
}

# Parsed whole before running, so nothing it starts can consume the rest of a streamed script.
main "$@" </dev/null; exit $?
