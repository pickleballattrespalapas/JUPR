-- Staging only. No fixture or match/rating change survives this transaction.
BEGIN;
DO $test$
DECLARE
  fixture_club text := 'game-activity-' || gen_random_uuid()::text;
  fixture_id bigint := -(txid_current() * 100);
  player_row public.players%rowtype;
BEGIN
  INSERT INTO public.clubs(id,slug,name,is_active,status,onboarding_status)
  VALUES(fixture_club,fixture_club,'Game activity fixture',false,'draft','draft');

  INSERT INTO public.players(id,club_id,name,normalized_name,rating,wins,losses,matches_played,active,last_game_at,inactive_at)
  VALUES(fixture_id,fixture_club,'New Signup','new signup',1600,0,0,0,true,null,null)
  RETURNING * INTO player_row;
  IF player_row.inactive_at IS NULL OR player_row.last_game_at IS NOT NULL THEN
    RAISE EXCEPTION 'A new signup must start inactive without manufacturing a game date';
  END IF;

  UPDATE public.players SET active=true,inactive_at=null WHERE id=fixture_id
  RETURNING * INTO player_row;
  IF player_row.inactive_at IS NULL THEN
    RAISE EXCEPTION 'A non-game update reactivated a never-played profile';
  END IF;

  UPDATE public.players SET last_game_at=now()-interval '30 days',active=true,inactive_at=null
  WHERE id=fixture_id RETURNING * INTO player_row;
  IF player_row.inactive_at IS NULL THEN
    RAISE EXCEPTION 'A historical game import made a player active';
  END IF;

  UPDATE public.players SET last_game_at=now(),active=true,inactive_at=null
  WHERE id=fixture_id RETURNING * INTO player_row;
  IF player_row.inactive_at IS NOT NULL THEN
    RAISE EXCEPTION 'A recent game did not reactivate the player';
  END IF;

  UPDATE public.players SET last_game_at=null WHERE id=fixture_id
  RETURNING * INTO player_row;
  IF player_row.inactive_at IS NULL THEN
    RAISE EXCEPTION 'Removing the final game left the player active';
  END IF;

  UPDATE public.players SET last_game_at=now(),active=true,inactive_at=null,
    name='New Signup (MERGED into Remaining #1)' WHERE id=fixture_id
  RETURNING * INTO player_row;
  IF player_row.inactive_at IS NULL THEN
    RAISE EXCEPTION 'Merged identity was reactivated';
  END IF;
  IF player_row.rating <> 1600 OR player_row.wins <> 0 OR player_row.losses <> 0 OR player_row.matches_played <> 0 THEN
    RAISE EXCEPTION 'Activity maintenance changed player results';
  END IF;

  IF has_function_privilege('anon','public.enforce_player_game_activity()','EXECUTE')
     OR has_function_privilege('authenticated','public.enforce_player_game_activity()','EXECUTE')
     OR has_function_privilege('service_role','public.enforce_player_game_activity()','EXECUTE') THEN
    RAISE EXCEPTION 'Trigger helper is exposed as an application RPC';
  END IF;
END;
$test$;
ROLLBACK;
SELECT 'PASS: signup, non-game update, historic game, recent game, final-game removal, merged identity, unchanged results, and ACLs' AS result;
