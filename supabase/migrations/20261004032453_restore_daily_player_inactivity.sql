-- Restore the documented 14-day leaderboard inactivity sweep from the legacy
-- migrations directory. Activity timestamps remain owned by match processing.
CREATE EXTENSION IF NOT EXISTS pg_cron WITH SCHEMA pg_catalog;

GRANT USAGE ON SCHEMA cron TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA cron TO postgres;

CREATE OR REPLACE FUNCTION public.mark_inactive_players()
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog
AS $function$
BEGIN
  UPDATE public.players
     SET inactive_at = now()
   WHERE inactive_at IS NULL
     AND COALESCE(last_game_at, created_at) <= now() - interval '14 days';
END;
$function$;

COMMENT ON FUNCTION public.mark_inactive_players() IS
  'Daily 14-day leaderboard inactivity sweep; preserves manual active flags, existing inactivity timestamps, match dates, and ratings.';

-- This is database maintenance, not a public or application RPC. The cron job
-- runs as its postgres owner, with no elevated function execution context.
REVOKE ALL ON FUNCTION public.mark_inactive_players()
  FROM PUBLIC, anon, authenticated, service_role;

-- Named scheduling updates the existing job for this owner on reapplication;
-- it does not accumulate duplicate jobs. Keep the documented 03:00 UTC slot.
DO $job$
DECLARE
  inactivity_job_id bigint;
BEGIN
  SELECT cron.schedule(
    'mark-inactive-players-daily',
    '0 3 * * *',
    'SELECT public.mark_inactive_players();'
  ) INTO inactivity_job_id;
  PERFORM cron.alter_job(inactivity_job_id, active := true);
END;
$job$;

-- Reconcile overdue players immediately, without waiting for the next night.
SELECT public.mark_inactive_players();
