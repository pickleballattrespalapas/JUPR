-- Automatic inactivity must clear in the same transaction that records recent
-- play. Date-only game timestamps can precede a new profile's automatic marker;
-- a newer marker is not an explicit exclusion when active is still true/null.
CREATE OR REPLACE FUNCTION public.enforce_player_game_activity()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
SET search_path = pg_catalog
AS $function$
BEGIN
  IF NEW.last_game_at IS NULL
     OR NEW.last_game_at <= now() - interval '14 days'
     OR NEW.last_game_at > now()
     OR strpos(lower(coalesce(NEW.name, '')), '(merged into ') > 0 THEN
    NEW.inactive_at := coalesce(NEW.inactive_at, now());
  ELSIF NEW.active IS DISTINCT FROM false THEN
    NEW.inactive_at := NULL;
  END IF;
  RETURN NEW;
END;
$function$;

REVOKE ALL ON FUNCTION public.enforce_player_game_activity()
  FROM PUBLIC, anon, authenticated, service_role;

-- Retain the existing schedule as a repair/sweep, not as a prerequisite for
-- results. Preserve explicit later active=false exclusions and merged profiles.
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
     AND (active IS DISTINCT FROM false OR inactive_at <= last_game_at)
     AND last_game_at > now() - interval '14 days'
     AND last_game_at <= now()
     AND strpos(lower(coalesce(name, '')), '(merged into ') = 0;

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
  'Repairs automatic inactivity for recent recorded play regardless of profile-creation time. Preserves later manual active=false exclusions, merged profiles, ratings, results, and game dates.';
REVOKE ALL ON FUNCTION public.mark_inactive_players()
  FROM PUBLIC, anon, authenticated, service_role;

-- Repair existing affected players, including first games entered for a profile
-- created later on the same date. No ratings, results, or game dates are changed.
SELECT public.mark_inactive_players();
