DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM matches
        GROUP BY club_id, id
        HAVING COUNT(*) > 1
    ) THEN
        RAISE EXCEPTION
            'Cannot add unique index matches_club_id_id_key: duplicate (club_id, id) rows exist in matches. Clean duplicates first.';
    END IF;
END
$$;

CREATE UNIQUE INDEX IF NOT EXISTS matches_club_id_id_key ON matches (club_id, id);
