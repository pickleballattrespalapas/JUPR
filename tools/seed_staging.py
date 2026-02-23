#!/usr/bin/env python3
"""WARNING: This script copies data from PROD Supabase to STAGING Supabase."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from supabase import Client, create_client

CHUNK_SIZE = 500
PAGE_SIZE = 1000


TABLE_CONFIG = [
    {"name": "badges", "club_col": "club_id", "time_col": None},
    {"name": "players", "club_col": "club_id", "time_col": None},
    {"name": "leagues_metadata", "club_col": "club_id", "time_col": None},
    {"name": "meta", "club_col": "club_id", "time_col": None},
    {"name": "matches", "club_col": "club_id", "time_col": "date"},
    {"name": "player_badges", "club_col": "club_id", "time_col": "created_at"},
    {"name": "player_stories", "club_col": "club_id", "time_col": "created_at"},
]


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def apply_filters(query: Any, club_col: str, club_id: str, time_col: Optional[str], since_iso: str) -> Any:
    query = query.eq(club_col, club_id)
    if time_col:
        query = query.gte(time_col, since_iso)
    return query


def fetch_all_rows(
    client: Client,
    table: str,
    club_col: str,
    club_id: str,
    time_col: Optional[str],
    since_iso: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    start = 0

    while True:
        end = start + PAGE_SIZE - 1
        query = client.table(table).select("*")
        query = apply_filters(query, club_col, club_id, time_col, since_iso)
        response = query.range(start, end).execute()

        batch = response.data or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    return rows


def delete_slice(
    client: Client,
    table: str,
    club_col: str,
    club_id: str,
    time_col: Optional[str],
    since_iso: str,
) -> int:
    query = client.table(table).delete(count="exact")
    query = apply_filters(query, club_col, club_id, time_col, since_iso)
    response = query.execute()
    return int(response.count or 0)


def upsert_rows(client: Client, table: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    total = 0
    for i in range(0, len(rows), CHUNK_SIZE):
        chunk = rows[i : i + CHUNK_SIZE]
        client.table(table).upsert(chunk).execute()
        total += len(chunk)
    return total


def process_table(
    prod: Client,
    staging: Client,
    table: str,
    club_col: str,
    club_id: str,
    time_col: Optional[str],
    since_iso: str,
) -> Tuple[int, int, int]:
    deleted = delete_slice(staging, table, club_col, club_id, time_col, since_iso)
    rows = fetch_all_rows(prod, table, club_col, club_id, time_col, since_iso)
    upserted = upsert_rows(staging, table, rows)
    return deleted, len(rows), upserted


def main() -> None:
    # WARNING: This operation copies data from PROD -> STAGING.
    prod_url = get_required_env("PROD_SUPABASE_URL")
    prod_key = get_required_env("PROD_SUPABASE_SERVICE_ROLE_KEY")
    staging_url = get_required_env("STAGING_SUPABASE_URL")
    staging_key = get_required_env("STAGING_SUPABASE_SERVICE_ROLE_KEY")

    club_id = os.getenv("CLUB_ID", "tres_palapas")
    days = int(os.getenv("DAYS", "180"))

    since = datetime.now(timezone.utc) - timedelta(days=days)
    since_iso = since.isoformat()

    prod = create_client(prod_url, prod_key)
    staging = create_client(staging_url, staging_key)

    print("WARNING: Seeding STAGING from PROD")
    print(f"Club: {club_id}")
    print(f"Window: last {days} days (since {since_iso})")

    for cfg in TABLE_CONFIG:
        table = cfg["name"]
        deleted, fetched, upserted = process_table(
            prod=prod,
            staging=staging,
            table=table,
            club_col=cfg["club_col"],
            club_id=club_id,
            time_col=cfg["time_col"],
            since_iso=since_iso,
        )
        print(
            f"[{table}] deleted_from_staging={deleted} "
            f"fetched_from_prod={fetched} upserted_to_staging={upserted}"
        )


if __name__ == "__main__":
    main()
