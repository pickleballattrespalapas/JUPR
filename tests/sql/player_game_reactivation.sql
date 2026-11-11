-- Staging regression. All fixtures and activity changes roll back.
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '30s';
DO $test$
DECLARE
  fixture_club text := 'immediate-activity-' || gen_random_uuid()::text;
  fixture_id bigint := -(txid_current() * 100);
  player_row public.players%rowtype;
  created_marker timestamptz;
BEGIN
  INSERT INTO public.clubs(id,slug,name,is_active,status,onboarding_status)
  VALUES(fixture_club,fixture_club,'Immediate activity fixture',false,'draft','draft');
  INSERT INTO public.players(id,club_id,name,normalized_name,rating,wins,losses,matches_played,active)
  VALUES(fixture_id,fixture_club,'New same-day player','new same-day player',1600,0,0,0,true)
  RETURNING * INTO player_row;
  created_marker := player_row.inactive_at;
  IF created_marker IS NULL THEN
    RAISE EXCEPTION 'Signup must stay inactive until a game is recorded';
  END IF;

  -- Match timestamps can be midnight even though the new profile was created
  -- later. The save path updates last_game_at inside the scored-game write.
  UPDATE public.players SET last_game_at=date_trunc('day',now())
  WHERE id=fixture_id RETURNING * INTO player_row;
  IF player_row.inactive_at IS NOT NULL OR player_row.active IS NOT TRUE THEN
    RAISE EXCEPTION 'Same-day scored game still waits for activity reconciliation';
  END IF;
  UPDATE public.players SET inactive_at=created_marker
  WHERE id=fixture_id RETURNING * INTO player_row;
  IF player_row.inactive_at IS NOT NULL THEN
    RAISE EXCEPTION 'A legacy write restored the stale automatic marker';
  END IF;
  UPDATE public.players SET active=false,inactive_at=now()
  WHERE id=fixture_id RETURNING * INTO player_row;
  IF player_row.active IS NOT FALSE OR player_row.inactive_at IS NULL THEN
    RAISE EXCEPTION 'Manual exclusion was lost';
  END IF;
  PERFORM public.mark_inactive_players();
  SELECT * INTO player_row FROM public.players WHERE id=fixture_id;
  IF player_row.active IS NOT FALSE OR player_row.inactive_at IS NULL THEN
    RAISE EXCEPTION 'Daily cleanup lost the later manual exclusion';
  END IF;
  UPDATE public.players SET active=true,inactive_at=null,last_game_at=null
  WHERE id=fixture_id RETURNING * INTO player_row;
  IF player_row.inactive_at IS NULL THEN
    RAISE EXCEPTION 'Removing the final game left the profile active';
  END IF;
  UPDATE public.players SET active=true,inactive_at=null,last_game_at=now(),
    name='Old (MERGED into Remaining #1)'
  WHERE id=fixture_id RETURNING * INTO player_row;
  IF player_row.inactive_at IS NULL THEN
    RAISE EXCEPTION 'Merged profile became visible';
  END IF;
  IF player_row.rating <> 1600 OR player_row.wins <> 0
     OR player_row.losses <> 0 OR player_row.matches_played <> 0 THEN
    RAISE EXCEPTION 'Activity maintenance changed recorded results';
  END IF;
END;
$test$;
ROLLBACK;
SELECT 'PASS: immediate same-day reactivation, legacy writes, manual exclusion, final-game removal, merged profile, unchanged results, and rollback' AS result;
