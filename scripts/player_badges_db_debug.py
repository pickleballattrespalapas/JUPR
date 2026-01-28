#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from typing import Any


QUERIES = {
    "player_badges_columns": """
        select column_name, data_type, is_nullable
        from information_schema.columns
        where table_schema='public' and table_name='player_badges'
        order by ordinal_position;
    """,
    "player_badges_constraints": """
        select conname, pg_get_constraintdef(c.oid) as def
        from pg_constraint c
        join pg_class t on t.oid=c.conrelid
        join pg_namespace n on n.oid=t.relnamespace
        where n.nspname='public' and t.relname='player_badges' and c.contype in ('p','u')
        order by conname;
    """,
    "player_badges_indexes": """
        select indexname, indexdef
        from pg_indexes
        where schemaname='public' and tablename='player_badges'
        order by indexname;
    """,
    "player_badges_duplicates_top50": """
        select club_id, player_id, badge_id, count(*) n
        from public.player_badges
        group by club_id, player_id, badge_id
        having count(*)>1
        order by n desc
        limit 50;
    """,
    "player_badges_duplicates_total": """
        select sum(n-1) as extra_rows
        from (
          select count(*) n
          from public.player_badges
          group by club_id, player_id, badge_id
          having count(*)>1
        ) s;
    """,
}


def _get_db_url() -> str:
    return (
        os.getenv("SUPABASE_DB_URL")
        or os.getenv("DATABASE_URL")
        or os.getenv("SUPABASE_DATABASE_URL")
        or ""
    )


def _load_psycopg2():
    try:
        import psycopg2
        from psycopg2.extras import RealDictCursor
    except Exception as exc:  # pragma: no cover - guard for missing dependency
        print("psycopg2 is required to run this script.", file=sys.stderr)
        print("Install with: pip install psycopg2-binary", file=sys.stderr)
        raise SystemExit(2) from exc
    return psycopg2, RealDictCursor


def _print_rows(label: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n--- {label} ({len(rows)} rows) ---")
    print(json.dumps(rows, indent=2, default=str))


def main() -> int:
    db_url = _get_db_url()
    if not db_url:
        print("Missing DATABASE_URL/SUPABASE_DB_URL env var.", file=sys.stderr)
        return 1

    psycopg2, RealDictCursor = _load_psycopg2()

    with psycopg2.connect(db_url) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            for label, sql in QUERIES.items():
                cur.execute(sql)
                rows = cur.fetchall()
                _print_rows(label, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
