#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIGRATIONS_DIR="${ROOT_DIR}/migrations"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required (Postgres connection string)." >&2
  exit 1
fi

if [[ ! -d "${MIGRATIONS_DIR}" ]]; then
  echo "Migrations directory not found: ${MIGRATIONS_DIR}" >&2
  exit 1
fi

psql "${DATABASE_URL}" -v ON_ERROR_STOP=1 <<'SQL'
create table if not exists public.schema_migrations (
  filename text primary key,
  applied_at timestamptz not null default now()
);
SQL

applied=$(
  psql "${DATABASE_URL}" -v ON_ERROR_STOP=1 -t -A \
    -c "select filename from public.schema_migrations order by filename;"
)

mapfile -t migration_files < <(ls -1 "${MIGRATIONS_DIR}"/*.sql 2>/dev/null | sort)

if [[ ${#migration_files[@]} -eq 0 ]]; then
  echo "No migrations found in ${MIGRATIONS_DIR}."
  exit 0
fi

for file in "${migration_files[@]}"; do
  filename="$(basename "${file}")"
  if echo "${applied}" | grep -Fxq "${filename}"; then
    echo "Skipping ${filename} (already applied)."
    continue
  fi
  echo "Applying ${filename}..."
  psql "${DATABASE_URL}" -v ON_ERROR_STOP=1 <<SQL
begin;
\\i ${file}
insert into public.schema_migrations (filename) values ('${filename}');
commit;
SQL
done

echo "Migrations complete."
