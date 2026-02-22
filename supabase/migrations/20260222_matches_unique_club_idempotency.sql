DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND indexname = 'matches_unique_club_idempotency'
    ) THEN
        IF EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = 'public'
              AND indexname = 'matches_idempotency_unique'
        ) THEN
            ALTER INDEX matches_idempotency_unique RENAME TO matches_unique_club_idempotency;
        ELSE
            CREATE UNIQUE INDEX matches_unique_club_idempotency
            ON matches (club_id, idempotency_key)
            WHERE idempotency_key IS NOT NULL;
        END IF;
    END IF;
END;
$$;
