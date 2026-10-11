#!/usr/bin/env bash
# Read-only service check on the host. Prints no keys; stream it with
#   Get-Content -Raw status.sh | ssh -T root@<host> 'bash -s'
set -uo pipefail

main() {
  local base=http://127.0.0.1:28717 code
  echo "[status] container: $(docker ps -a --filter name=aml --format '{{.Names}} {{.Status}}' | head -n 1)"
  echo "[status] health: $(curl -sS -m 10 "${base}/health" || echo unreachable)"
  code="$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST -H 'Content-Type: application/json' -d '{}' "${base}/search")"
  echo "[status] search without key: ${code} (expect 401)"
  code="$(printf 'header = "Authorization: Bearer %s"\n' "$(sed -n 's/^AML_SERVICE_API_KEY=//p' /etc/aml/aml.env | head -n 1)" \
    | curl -s -m 60 -o /dev/null -w '%{http_code}' -K - -H 'Content-Type: application/json' \
      -d '{"query":"probe","user_id":"selftest-0","top_k":1}' "${base}/search")"
  echo "[status] search with key: ${code} (expect 200)"
  echo "[status] env file: $(stat -c '%a %U' /etc/aml/aml.env 2>/dev/null || echo missing) (expect 600 root)"
  echo "[status] dns: $(getent ahostsv4 test.memprism.com | awk 'NR==1{print $1}' || true) (expect 47.83.232.55)"
  echo "[status] https health: $(curl -s -m 10 -o /dev/null -w '%{http_code}' https://test.memprism.com/health || true) (000 until DNS and certificate are set up)"
}

main </dev/null; exit 0
