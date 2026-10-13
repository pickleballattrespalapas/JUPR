-- Run against staging after the migration, inside a transaction that is rolled
-- back. Fixed negative fixture IDs avoid advancing production player sequences.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';

INSERT INTO public.players
  (id, club_id, name, normalized_name, singles_replay_baseline,
   rating, wins, losses, matches_played, active, inactive_at, last_game_at)
SELECT -131000 - n, 'cabo-test-club',
       CASE WHEN n = 9 THEN 'Activity fixture (merged into Player #1)'
            ELSE 'Activity reconciliation fixture ' || n END,
       'activity-reconciliation-fixture-' || n,
       '{"rating":1200,"wins":0,"losses":0,"matches_played":0}'::jsonb,
       1400, 3, 2, 5,
       n IN (5, 6, 8),
       CASE WHEN n IN (1, 2, 9) THEN now() - interval '3 days'
            WHEN n = 3 THEN now() - interval '1 hour'
            WHEN n = 7 THEN now() - interval '1 day' END,
       CASE WHEN n IN (1, 2, 3, 4, 9) THEN now() - interval '2 days'
            WHEN n = 5 THEN now() - interval '14 days'
            WHEN n = 6 THEN now() - interval '14 days' + interval '1 second'
            WHEN n = 7 THEN now() + interval '1 day' END
FROM generate_series(1, 9) AS fixture(n);

CREATE TEMP TABLE activity_before ON COMMIT DROP AS
SELECT id, rating, wins, losses, matches_played, last_game_at
FROM public.players WHERE id BETWEEN -131009 AND -131001;

SELECT public.mark_inactive_players();

DO $test$
BEGIN
  IF (SELECT count(*) FROM public.players WHERE id IN (-131001, -131002)
      AND active IS TRUE AND inactive_at IS NULL) <> 2 THEN
    RAISE EXCEPTION 'Automatic inactive flags were not repaired';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM public.players WHERE id = -131003
                 AND active IS FALSE AND inactive_at = now() - interval '1 hour')
     OR NOT EXISTS (SELECT 1 FROM public.players WHERE id = -131004
                    AND active IS FALSE AND inactive_at IS NULL) THEN
    RAISE EXCEPTION 'Explicit exclusions were overwritten';
  END IF;
  IF (SELECT count(*) FROM public.players WHERE id IN (-131005, -131007, -131008, -131009)
      AND inactive_at IS NOT NULL) <> 4 THEN
    RAISE EXCEPTION 'Stale, future, never-played, or merged profiles became active';
  END IF;
  IF NOT EXISTS (SELECT 1 FROM public.players WHERE id = -131006
                 AND active IS TRUE AND inactive_at IS NULL) THEN
    RAISE EXCEPTION 'The 14-day boundary is incorrect';
  END IF;
  IF EXISTS (
    (SELECT id, rating, wins, losses, matches_played, last_game_at
     FROM public.players WHERE id BETWEEN -131009 AND -131001)
    EXCEPT SELECT * FROM activity_before
  ) THEN
    RAISE EXCEPTION 'Activity reconciliation changed ratings, results, or game dates';
  END IF;
END;
$test$;

CREATE TEMP TABLE activity_after ON COMMIT DROP AS
SELECT id, active, inactive_at FROM public.players
WHERE id BETWEEN -131009 AND -131001;
SELECT public.mark_inactive_players();
DO $test$
BEGIN
  IF EXISTS (
    (SELECT id, active, inactive_at FROM public.players
     WHERE id BETWEEN -131009 AND -131001)
    EXCEPT SELECT * FROM activity_after
  ) THEN
    RAISE EXCEPTION 'Repeated reconciliation is not idempotent';
  END IF;
END;
$test$;
ROLLBACK;
