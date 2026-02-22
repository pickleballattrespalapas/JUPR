CREATE OR REPLACE FUNCTION public.dedupe_tournament_games(t_id text)
RETURNS integer
LANGUAGE plpgsql
AS $$
DECLARE
    deleted_count integer := 0;
BEGIN
    WITH ranked_games AS (
        SELECT
            id,
            row_number() OVER (
                PARTITION BY club_id, tournament_id, stage, rr_round_number, rr_slot_number
                ORDER BY created_at ASC, id ASC
            ) AS row_num
        FROM public.tournament_games
        WHERE tournament_id = t_id
    ),
    deleted AS (
        DELETE FROM public.tournament_games tg
        USING ranked_games rg
        WHERE tg.id = rg.id
          AND rg.row_num > 1
        RETURNING 1
    )
    SELECT count(*) INTO deleted_count FROM deleted;

    RETURN deleted_count;
END;
$$;
