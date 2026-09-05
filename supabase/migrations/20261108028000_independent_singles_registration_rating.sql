-- Keep a self-reported tournament singles level independent from the club's
-- canonical overall/doubles rating. The registration row retains
-- singles_skill for tournament placement, but only doubles_skill may seed a
-- newly created player's rating.

create or replace function private.resolve_public_tournament_primary_player_v1(
  p_club_id text,
  p_tournament_id text,
  p_registration jsonb
)
returns jsonb
language plpgsql
security definer
set search_path = ''
as $function$
declare
  v_registration_id text := nullif(pg_catalog.btrim(p_registration ->> 'id'), '');
  v_email text := pg_catalog.lower(nullif(pg_catalog.btrim(p_registration ->> 'email'), ''));
  v_display_name text := nullif(pg_catalog.btrim(coalesce(
    p_registration ->> 'display_name',
    pg_catalog.concat_ws(
      ' ',
      nullif(pg_catalog.btrim(p_registration ->> 'first_name'), ''),
      nullif(pg_catalog.btrim(p_registration ->> 'last_name'), '')
    )
  )), '');
  v_normalized_name text;
  v_email_lock bigint;
  v_name_lock bigint;
  v_existing_registration public.tournament_registrations%rowtype;
  v_existing_reconciliation public.tournament_primary_player_reconciliation%rowtype;
  v_prior_identity_count integer := 0;
  v_doubles_skill numeric;
  v_player public.players%rowtype;
  v_state text;
begin
  if v_registration_id is null or v_email is null or v_display_name is null then
    raise exception 'JUPR_PRIMARY_PLAYER_IDENTITY_INVALID: name and email are required.';
  end if;
  if not exists (
    select 1
      from public.tournaments as tournament
     where tournament.id::text = p_tournament_id
       and tournament.club_id::text = p_club_id
  ) then
    raise exception 'JUPR_PRIMARY_PLAYER_TOURNAMENT_MISMATCH: tournament does not belong to this club.';
  end if;

  v_normalized_name := pg_catalog.lower(pg_catalog.btrim(v_display_name));
  v_email_lock := pg_catalog.hashtextextended(
    'public-primary-player-email:' || p_club_id || ':' || v_email,
    0
  );
  v_name_lock := pg_catalog.hashtextextended(
    'public-primary-player-name:' || p_club_id || ':' || v_normalized_name,
    0
  );
  perform pg_catalog.pg_advisory_xact_lock(case
    when v_email_lock <= v_name_lock then v_email_lock else v_name_lock
  end);
  if v_email_lock <> v_name_lock then
    perform pg_catalog.pg_advisory_xact_lock(case
      when v_email_lock > v_name_lock then v_email_lock else v_name_lock
    end);
  end if;

  select registration.*
    into v_existing_registration
    from public.tournament_registrations as registration
   where registration.id = v_registration_id
     and registration.tournament_id::text = p_tournament_id
   limit 1;
  if v_existing_registration.id is not null
     and v_existing_registration.player_id is not null then
    select reconciliation.*
      into v_existing_reconciliation
      from public.tournament_primary_player_reconciliation as reconciliation
     where reconciliation.registration_id = v_registration_id
       and reconciliation.tournament_id::text = p_tournament_id
       and reconciliation.club_id = p_club_id
     limit 1;
    if v_existing_reconciliation.registration_id is not null
       and v_existing_reconciliation.created_player_id = v_existing_registration.player_id
       and v_existing_reconciliation.state in ('CREATED', 'CREATED_UNRATED') then
      return pg_catalog.jsonb_build_object(
        'player_id', v_existing_registration.player_id,
        'state', v_existing_reconciliation.state,
        'reason_code', v_existing_reconciliation.reason_code
      );
    end if;
    return pg_catalog.jsonb_build_object(
      'player_id', v_existing_registration.player_id,
      'state', 'EXISTING_REGISTRATION',
      'reason_code', null
    );
  end if;

  select pg_catalog.count(*)::integer
    into v_prior_identity_count
    from public.tournament_registrations as registration
    join public.tournaments as tournament
      on tournament.id::text = registration.tournament_id::text
   where tournament.club_id::text = p_club_id
     and pg_catalog.lower(pg_catalog.btrim(registration.email)) = v_email
     and registration.id <> v_registration_id;

  if v_prior_identity_count > 0 then
    return pg_catalog.jsonb_build_object(
      'player_id', null,
      'state', 'STAFF_RECONCILIATION_REQUIRED',
      'reason_code', 'EMAIL_COLLISION'
    );
  end if;

  if exists (
    select 1
      from public.players as player
     where player.club_id = p_club_id
       and pg_catalog.lower(pg_catalog.btrim(player.name)) = v_normalized_name
  ) then
    return pg_catalog.jsonb_build_object(
      'player_id', null,
      'state', 'STAFF_RECONCILIATION_REQUIRED',
      'reason_code', 'EXISTING_PLAYER_NAME_COLLISION'
    );
  end if;

  begin
    v_doubles_skill := nullif(p_registration ->> 'doubles_skill', '')::numeric;
  exception when others then
    v_doubles_skill := null;
  end;
  if (
    v_doubles_skill is null
    or v_doubles_skill < 1
    or v_doubles_skill > 7
  ) then
    -- A singles-only self-rating belongs to this registration, not the
    -- player's canonical overall/doubles history.
    v_doubles_skill := 3.0;
    v_state := 'CREATED_UNRATED';
  else
    v_state := 'CREATED';
  end if;

  begin
    insert into public.players (
      club_id,
      name,
      rating,
      starting_rating,
      singles_rating,
      wins,
      losses,
      matches_played,
      active,
      last_game_at,
      inactive_at
    ) values (
      p_club_id,
      v_display_name,
      v_doubles_skill * 400.0,
      v_doubles_skill * 400.0,
      null,
      0,
      0,
      0,
      true,
      null,
      null
    )
    returning * into v_player;
  exception when unique_violation then
    return pg_catalog.jsonb_build_object(
      'player_id', null,
      'state', 'STAFF_RECONCILIATION_REQUIRED',
      'reason_code', 'EXISTING_PLAYER_NAME_COLLISION'
    );
  end;

  return pg_catalog.jsonb_build_object(
    'player_id', v_player.id,
    'state', v_state,
    'reason_code', case
      when v_state = 'CREATED_UNRATED' then 'TEMPORARY_3_0_BASELINE'
      else null
    end
  );
end
$function$;

revoke all on function private.resolve_public_tournament_primary_player_v1(
  text,
  text,
  jsonb
) from public, anon, authenticated;
grant execute on function private.resolve_public_tournament_primary_player_v1(
  text,
  text,
  jsonb
) to service_role;

comment on function private.resolve_public_tournament_primary_player_v1(
  text,
  text,
  jsonb
) is
  'Resolves public tournament registrants without deriving canonical overall/doubles ratings from self-reported singles skill.';

-- The immutable replay baseline is required before a player's first managed
-- singles result can be calculated. Historically, the insert trigger copied
-- the overall/doubles rating when singles_rating was absent. Keep every
-- existing baseline untouched, but initialize future players independently at
-- the neutral 3.0 baseline unless an explicit official singles rating exists.
create or replace function public.initialize_player_singles_replay_baseline()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if new.singles_replay_baseline is null then
    new.singles_replay_baseline := pg_catalog.jsonb_build_object(
      'rating', coalesce(new.singles_rating, 1200.0),
      'wins', coalesce(new.singles_wins, 0),
      'losses', coalesce(new.singles_losses, 0),
      'matches_played', coalesce(new.singles_matches_played, 0),
      'last_game_at', new.singles_last_game_at
    );
  end if;
  return new;
end
$function$;

revoke all on function public.initialize_player_singles_replay_baseline()
  from public, anon, authenticated;
grant execute on function public.initialize_player_singles_replay_baseline()
  to service_role;

comment on function public.initialize_player_singles_replay_baseline() is
  'Initializes future singles replay history independently at an explicit singles rating or the neutral 3.0 baseline.';
