-- A league baseline is always the player's reviewed current overall rating.
-- This removes the historic singles-rating/1200 fallback for roster activation
-- without affecting Replay History, backfills, or direct-match inserts.

create or replace function private.league_rating_start_from_overall_v1()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_overall_rating numeric;
begin
  if pg_catalog.current_setting(
    'jupr.league_roster_overall_baseline',
    true
  ) is distinct from 'on' then
    return new;
  end if;

  select player.rating::numeric
    into v_overall_rating
    from public.players as player
   where player.club_id = new.club_id
     and player.id = new.player_id;

  if not found then
    raise exception using
      errcode = '23503',
      message = 'JUPR_LEAGUE_OVERALL_PLAYER_NOT_FOUND: the league member must be a club player.';
  end if;

  if v_overall_rating is null
     or v_overall_rating < 400
     or v_overall_rating > 2800 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_LEAGUE_OVERALL_RATING_REQUIRED: set a reviewed Overall JUPR before adding this player to a league.';
  end if;

  new.rating := pg_catalog.round(v_overall_rating, 4);
  new.starting_rating := pg_catalog.round(v_overall_rating, 4);
  return new;
end;
$function$;

revoke all on function private.league_rating_start_from_overall_v1() from public;
revoke all on function private.league_rating_start_from_overall_v1() from anon;
revoke all on function private.league_rating_start_from_overall_v1() from authenticated;

drop trigger if exists trg_league_rating_start_from_overall_v1
  on public.league_ratings;
create trigger trg_league_rating_start_from_overall_v1
before insert
on public.league_ratings
for each row
execute function private.league_rating_start_from_overall_v1();

-- The v2 function remains the reviewed lock/idempotency authority. This thin
-- entry point scopes the INSERT trigger to roster activation only.
create or replace function public.admin_apply_league_roster_batch_atomic_v3(
  p_operation_id uuid,
  p_club_id text,
  p_league_name text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_action text,
  p_player_ids jsonb,
  p_starting_rating numeric,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  perform pg_catalog.set_config(
    'jupr.league_roster_overall_baseline',
    'on',
    true
  );
  return public.admin_apply_league_roster_batch_atomic_v2(
    p_operation_id,
    p_club_id,
    p_league_name,
    p_idempotency_key,
    p_request_fingerprint,
    p_action,
    p_player_ids,
    p_starting_rating,
    p_actor_email,
    p_actor_role,
    p_source
  );
end;
$function$;

revoke all on function public.admin_apply_league_roster_batch_atomic_v3(
  uuid, text, text, text, text, text, jsonb, numeric, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_apply_league_roster_batch_atomic_v3(
  uuid, text, text, text, text, text, jsonb, numeric, text, text, text
) to service_role;

comment on function public.admin_apply_league_roster_batch_atomic_v3(
  uuid, text, text, text, text, text, jsonb, numeric, text, text, text
) is
  'Runs the guarded roster batch while seeding only newly inserted league ratings from each player current overall rating.';

notify pgrst, 'reload schema';
