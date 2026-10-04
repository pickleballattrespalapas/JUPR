-- Account creation/registration is not play. Keep the existing daily schedule,
-- but remove the 14-day creation-date grace period for never-played profiles.
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
     AND (
       last_game_at IS NULL
       OR last_game_at <= now() - interval '14 days'
       OR last_game_at > now()
       OR strpos(lower(coalesce(name, '')), '(merged into ') > 0
     );
END;
$function$;

COMMENT ON FUNCTION public.mark_inactive_players() IS
  'Leaderboard inactivity requires a recorded game within 14 days; registration and account creation do not count. Preserves ratings, game dates, and explicit active flags.';
REVOKE ALL ON FUNCTION public.mark_inactive_players()
  FROM PUBLIC, anon, authenticated, service_role;

-- Enforce the same invariant immediately when a profile is created or an
-- activity field changes, including old clients and historical score imports.
-- Never change ratings, match dates, registrations, or manual active flags.
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
  END IF;
  RETURN NEW;
END;
$function$;

REVOKE ALL ON FUNCTION public.enforce_player_game_activity()
  FROM PUBLIC, anon, authenticated, service_role;

DROP TRIGGER IF EXISTS trg_players_game_activity ON public.players;
CREATE TRIGGER trg_players_game_activity
BEFORE INSERT OR UPDATE OF last_game_at, inactive_at, active, name
ON public.players
FOR EACH ROW EXECUTE FUNCTION public.enforce_player_game_activity();

-- Retire the existing no-game grace-period records without touching results.
SELECT public.mark_inactive_players();
