-- Staging only: fixtures and their trigger side effects are rolled back.
BEGIN;
DO $test$
DECLARE
  fixture_club text := 'inactivity-job-' || gen_random_uuid()::text;
  base_id bigint := -(txid_current() * 100);
  i integer;
  before_rows jsonb;
  first_pass jsonb;
BEGIN
  INSERT INTO public.clubs(id, slug, name, is_active, status, onboarding_status)
  VALUES(fixture_club, fixture_club, 'Inactivity job fixture', false, 'draft', 'draft');

  FOR i IN 1..8 LOOP
    INSERT INTO public.players(
      id, club_id, name, normalized_name, rating, wins, losses, matches_played,
      active, inactive_at, last_game_at, created_at
    ) VALUES (
      base_id - i, fixture_club, 'Inactivity fixture ' || i,
      'inactivity fixture ' || i, 1600, 3, 2, 5,
      i <> 7,
      CASE WHEN i = 6 THEN now() - interval '30 days' END,
      CASE
        WHEN i = 1 THEN now() - interval '15 days'
        WHEN i = 2 THEN now() - interval '14 days' + interval '1 second'
        WHEN i = 3 THEN now() - interval '14 days'
        WHEN i IN (6, 7) THEN now() - interval '1 day'
      END,
      CASE
        WHEN i = 4 THEN now() - interval '15 days'
        WHEN i <> 8 THEN now() - interval '1 day'
      END
    );
  END LOOP;
  SELECT jsonb_agg(to_jsonb(p) - 'inactive_at' ORDER BY id)
    INTO before_rows FROM public.players p WHERE club_id = fixture_club;

  PERFORM public.mark_inactive_players();
  IF EXISTS (
    SELECT 1 FROM public.players
     WHERE club_id = fixture_club
       AND (inactive_at IS NOT NULL) IS DISTINCT FROM
           (id IN (base_id - 1, base_id - 3, base_id - 4, base_id - 6))
  ) THEN
    RAISE EXCEPTION '14-day boundary, creation fallback, or recent-player classification failed';
  END IF;
  IF (SELECT inactive_at FROM public.players WHERE id = base_id - 6)
      IS DISTINCT FROM now() - interval '30 days' THEN
    RAISE EXCEPTION 'Existing inactivity timestamp changed';
  END IF;
  IF before_rows IS DISTINCT FROM (
    SELECT jsonb_agg(to_jsonb(p) - 'inactive_at' ORDER BY id)
      FROM public.players p WHERE club_id = fixture_club
  ) THEN
    RAISE EXCEPTION 'Maintenance changed a field other than inactive_at';
  END IF;
  SELECT jsonb_agg(to_jsonb(p) ORDER BY id)
    INTO first_pass FROM public.players p WHERE club_id = fixture_club;
  PERFORM public.mark_inactive_players();
  IF first_pass IS DISTINCT FROM (
    SELECT jsonb_agg(to_jsonb(p) ORDER BY id)
      FROM public.players p WHERE club_id = fixture_club
  ) THEN
    RAISE EXCEPTION 'Repeated inactivity sweep changed player rows';
  END IF;

  IF has_function_privilege('anon', 'public.mark_inactive_players()', 'EXECUTE')
    OR has_function_privilege('authenticated', 'public.mark_inactive_players()', 'EXECUTE')
    OR has_function_privilege('service_role', 'public.mark_inactive_players()', 'EXECUTE') THEN
    RAISE EXCEPTION 'Maintenance function exposed through an application role';
  END IF;
  IF (SELECT count(*) FROM cron.job
       WHERE jobname = 'mark-inactive-players-daily'
         AND schedule = '0 3 * * *'
         AND command = 'SELECT public.mark_inactive_players();'
         AND username = 'postgres' AND active) <> 1 THEN
    RAISE EXCEPTION 'Expected exactly one enabled daily postgres job';
  END IF;
END;
$test$;
ROLLBACK;
SELECT 'PASS: cutoff, new players, manual flags, unchanged ratings, idempotency, permissions, and daily schedule' AS result;
