#!/usr/bin/env bash
# Online SQLite backup for AML databases under /var/lib/aml. Retain last 7 runs.
set -euo pipefail

DATA_DIR="${AML_DATA_DIR:-/var/lib/aml}"
BACKUP_ROOT="${AML_BACKUP_ROOT:-/var/backups/aml}"
KEEP="${AML_BACKUP_KEEP:-7}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
DEST="${BACKUP_ROOT}/${STAMP}"

if ! command -v sqlite3 >/dev/null 2>&1; then
  echo 'sqlite3 not found' >&2
  exit 1
fi

install -d -m 0750 "${DEST}"
shopt -s nullglob
found=0
for db in "${DATA_DIR}"/*.sqlite; do
  [[ -f "${db}" ]] || continue
  base="$(basename "${db}" .sqlite)"
  sqlite3 "${db}" ".backup '${DEST}/${base}.sqlite'"
  found=1
done

if [[ "${found}" -eq 0 ]]; then
  echo "No *.sqlite in ${DATA_DIR}; nothing backed up." >&2
  rmdir "${DEST}" 2>/dev/null || true
  exit 0
fi

mapfile -t runs < <(find "${BACKUP_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort)
while [[ "${#runs[@]}" -gt "${KEEP}" ]]; do
  rm -rf "${runs[0]}"
  runs=("${runs[@]:1}")
done

echo "Backup written to ${DEST}"
