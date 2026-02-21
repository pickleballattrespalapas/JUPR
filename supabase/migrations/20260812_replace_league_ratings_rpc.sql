CREATE OR REPLACE FUNCTION public.replace_league_ratings(
    p_club_id text,
    p_rows jsonb,
    p_reset boolean DEFAULT true
)
RETURNS void
LANGUAGE plpgsql
AS $$
BEGIN
    IF p_reset THEN
        DELETE FROM league_ratings WHERE club_id = p_club_id;
    END IF;

    IF p_rows IS NULL OR jsonb_typeof(p_rows) <> 'array' OR jsonb_array_length(p_rows) = 0 THEN
        RETURN;
    END IF;

    INSERT INTO league_ratings (
        club_id,
        league_name,
        player_id,
        rating,
        wins,
        losses,
        matches_played,
        starting_rating
    )
    SELECT
        p_club_id,
        x.league_name,
        x.player_id,
        x.rating,
        x.wins,
        x.losses,
        x.matches_played,
        x.starting_rating
    FROM jsonb_to_recordset(p_rows) AS x(
        league_name text,
        player_id bigint,
        rating numeric,
        wins integer,
        losses integer,
        matches_played integer,
        starting_rating numeric
    );
END;
$$;
