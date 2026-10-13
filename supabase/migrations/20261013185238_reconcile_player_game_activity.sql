-- The existing daily job must reconcile automatic inactivity in both directions.
-- A date-only game formerly assigned noon could be in the future when entered
-- that morning. Its inactivity marker predates the game; once that game time
-- passes, the marker must not keep an otherwise recent player hidden forever.
CREATE OR REPLACE FUNCTION public.mark_inactive_players()
RETURNS void
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog
AS $function$
BEGIN
  UPDATE public.players
     SET active = true,
         inactive_at = NULL
   WHERE inactive_at IS NOT NULL
     AND inactive_at <= last_game_at
     AND last_game_at > now() - interval '14 days'
     AND last_game_at <= now()
     AND strpos(lower(coalesce(name, '')), '(merged into ') = 0;

  -- Keep deliberate exclusions after the latest game and explicit active=false
  -- flags with no inactivity marker. Registration alone is never activity.
  UPDATE public.players
     SET inactive_at = now()
   WHERE inactive_at IS NULL
     AND (
       last_game_at IS NULL
       OR last_game_at <= now() - interval '14 days'
       OR last_game_at > now()
       OR strpos(lower(coalesce(name, '')), '(merged into ') > 0
     );
END;
$function$;

COMMENT ON FUNCTION public.mark_inactive_players() IS
  'Reconciles automatic leaderboard inactivity from recorded games within 14 days, including elapsed future timestamps. Preserves later manual exclusions, merged profiles, ratings, and game dates.';
REVOKE ALL ON FUNCTION public.mark_inactive_players()
  FROM PUBLIC, anon, authenticated, service_role;

-- Repair eligible automatic flags immediately; retain the installed daily job.
SELECT public.mark_inactive_players();
