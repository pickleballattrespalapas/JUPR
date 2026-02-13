"""Add idempotency support to matches inserts.

This migration is additive and backward-compatible:
- Adds `matches.idempotency_key` as nullable text.
- Adds uniqueness on `(club_id, idempotency_key)`.
"""

from __future__ import annotations

from typing import Any


def up(conn: Any) -> None:
    """Apply schema changes to support idempotent match writes."""
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE public.matches
            ADD COLUMN IF NOT EXISTS idempotency_key TEXT;
            """
        )
        cur.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'matches_club_id_idempotency_key_key'
                      AND conrelid = 'public.matches'::regclass
                ) THEN
                    ALTER TABLE public.matches
                    ADD CONSTRAINT matches_club_id_idempotency_key_key
                    UNIQUE (club_id, idempotency_key);
                END IF;
            END
            $$;
            """
        )
    conn.commit()


def down(conn: Any) -> None:
    """Rollback idempotency schema changes for matches."""
    with conn.cursor() as cur:
        cur.execute(
            """
            ALTER TABLE public.matches
            DROP CONSTRAINT IF EXISTS matches_club_id_idempotency_key_key;
            """
        )
        cur.execute(
            """
            ALTER TABLE public.matches
            DROP COLUMN IF EXISTS idempotency_key;
            """
        )
    conn.commit()
