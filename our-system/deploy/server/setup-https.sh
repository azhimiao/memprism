#!/usr/bin/env bash
# Issue the test.memprism.com certificate and install its nginx site. Other sites and certificates
# are untouched; nginx reloads only after `nginx -t` passes. Stream it with
#   Get-Content -Raw setup-https.sh | ssh -T root@<host> 'bash -s'
set -euo pipefail

main() {
  local domain=test.memprism.com expected_ip=47.83.232.55
  local src=/opt/aml/our-system/deploy/server
  local site="/etc/nginx/sites-available/${domain}" link="/etc/nginx/sites-enabled/${domain}"
  local resolved

  resolved="$(getent ahostsv4 "${domain}" | awk 'NR==1{print $1}' || true)"
  if [[ "${resolved}" != "${expected_ip}" ]]; then
    echo "[https] ${domain} resolves to '${resolved:-nothing}', expected ${expected_ip}." >&2
    echo "[https] Add the A record (DNS only, not proxied) and retry once it resolves." >&2
    return 1
  fi

  apply_site() {
    cp "$1" "${site}"
    ln -sf "${site}" "${link}"
    if ! nginx -t; then
      rm -f "${link}"
      echo '[https] nginx -t failed; the new site is disabled and nginx was not reloaded' >&2
      return 1
    fi
    systemctl reload nginx
  }

  mkdir -p /var/www/html
  if [[ ! -f "/etc/letsencrypt/live/${domain}/fullchain.pem" ]]; then
    apply_site "${src}/nginx-${domain}.acme.conf"
    certbot certonly --webroot -w /var/www/html -d "${domain}" --non-interactive --agree-tos \
      --register-unsafely-without-email --deploy-hook 'systemctl reload nginx'
  fi
  apply_site "${src}/nginx-${domain}.conf"
  certbot renew --dry-run --cert-name "${domain}"

  echo "[https] health: $(curl -sS -m 10 "https://${domain}/health")"
  echo "[https] search without key: $(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST \
    -H 'Content-Type: application/json' -d '{}' "https://${domain}/search") (expect 401)"
}

main </dev/null; exit $?
