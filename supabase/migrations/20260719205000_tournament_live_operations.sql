-- Order-28 draw-scoped Tournament Live idempotency and recovery extension.
--
-- This deliberately extends the order-26 Tournament Admin operation ledger
-- after the order-27 Tournament Ops surface extension. The table remains
-- FastAPI/service-role only; browsers retain only their UUID and the returned
-- deterministic server operation key.

alter table public.tournament_admin_operations
  add column if not exists client_idempotency_key text;

alter table public.tournament_admin_operations
  drop constraint if exists tournament_admin_surface_check;

alter table public.tournament_admin_operations
  add constraint tournament_admin_surface_check
  check (surface in ('tournament', 'setup', 'registration', 'import_handoff', 'operations', 'tournament_live'));

alter table public.tournament_admin_operations
  drop constraint if exists tournament_admin_client_idempotency_key_check;

alter table public.tournament_admin_operations
  add constraint tournament_admin_client_idempotency_key_check
  check (
    client_idempotency_key is null
    or client_idempotency_key ~* '^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
  );

create unique index if not exists idx_tournament_admin_operations_client_idempotency
  on public.tournament_admin_operations (club_id, surface, client_idempotency_key)
  where client_idempotency_key is not null;

create index if not exists idx_tournament_live_operations_draw_updated
  on public.tournament_admin_operations (club_id, entity_id, updated_at desc)
  where surface = 'tournament_live';

comment on column public.tournament_admin_operations.client_idempotency_key is
  'Browser-retained UUID for exact Tournament Live response-loss retry; the server operation_key remains canonical.';

-- A publish processor spans multiple Python calls, so its database transaction
-- cannot retain a row lock for the whole rating update. Treat the durable active
-- operation as a claim: every direct draw update and every Order-27 child/badge
-- trigger that tries to advance the draw version is rejected until publish is
-- completed or explicitly reconciled.
create or replace function public.block_tournament_draw_change_during_official_publish()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
begin
  if exists (
    select 1
      from public.tournament_admin_operations operation
      join public.tournaments tournament on tournament.id = old.tournament_id
     where operation.club_id = tournament.club_id
       and operation.entity_type = 'tournament_event_draw'
       and operation.entity_id = old.id::text
       and operation.action in ('ops_official_publish', 'tournament_live_official_publish')
       and operation.status in ('intent', 'mutated', 'recovery_required')
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LOCK';
  end if;
  if tg_op = 'DELETE' then return old; end if;
  return new;
end;
$$;

drop trigger if exists trg_00_tournament_draw_official_publish_lock on public.tournament_event_draws;
create trigger trg_00_tournament_draw_official_publish_lock
before update or delete on public.tournament_event_draws
for each row execute function public.block_tournament_draw_change_during_official_publish();

revoke all on function public.block_tournament_draw_change_during_official_publish()
  from public, anon, authenticated;
grant execute on function public.block_tournament_draw_change_during_official_publish()
  to service_role;

-- Award the exact reviewed podium recipient set in one transaction. The
-- Order-27 child triggers advance the draw version for direct writes; this RPC
-- additionally compares the reviewed draw/team/podium/candidate sets while the
-- complete dependency scope is locked, so it cannot rebuild recipients from a
-- concurrently replaced podium.
drop function if exists public.admin_award_tournament_draw_podium_cas(
  text, text, text, timestamptz, jsonb, jsonb, jsonb, jsonb
);
create or replace function public.admin_award_tournament_draw_podium_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_expected_teams jsonb,
  p_expected_podium jsonb,
  p_expected_awards jsonb,
  p_badges jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_saved jsonb;
begin
  if jsonb_typeof(coalesce(p_expected_teams, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_teams, '[]'::jsonb)) = 0
     or jsonb_typeof(coalesce(p_expected_podium, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_podium, '[]'::jsonb)) <> 3
     or jsonb_typeof(coalesce(p_expected_awards, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_expected_awards, '[]'::jsonb)) = 0
     or jsonb_typeof(coalesce(p_badges, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_badges, '[]'::jsonb)) <> jsonb_array_length(p_expected_awards) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_PLAN_STALE';
  end if;

  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  perform podium.id
    from public.tournament_podium podium
   where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
   order by podium.id
   for update;
  perform team.id
    from public.tournament_teams team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
   order by team.id
   for no key update;
  perform badge.id
    from public.player_badges badge
   where badge.club_id = p_club_id
     and badge.context_type = 'tournament'
     and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
   order by badge.id
   for update;

  select d.* into v_draw
    from public.tournament_event_draws d
    join public.tournaments t on t.id = d.tournament_id
   where d.id::text = p_draw_id
     and d.tournament_id::text = p_tournament_id
     and t.club_id = p_club_id
     and d.updated_at = p_expected_draw_updated_at
   for update of d;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  if (
       select count(*) from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
     ) <> jsonb_array_length(p_expected_teams)
     or (
       select count(distinct x.id) from jsonb_to_recordset(p_expected_teams) as x(id text)
     ) <> jsonb_array_length(p_expected_teams)
     or exists (
       select 1 from public.tournament_teams team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
          and not exists (
            select 1 from jsonb_to_recordset(p_expected_teams) as x(id text, updated_at timestamptz)
             where x.id = team.id::text and x.updated_at = team.updated_at
          )
     )
     or exists (
       select 1 from jsonb_to_recordset(p_expected_teams) as x(id text, updated_at timestamptz)
        where nullif(x.id, '') is null or x.updated_at is null or not exists (
          select 1 from public.tournament_teams team
           where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
             and team.id::text = x.id and team.updated_at = x.updated_at
        )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;

  if (
       select count(*) from public.tournament_podium podium
        where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
     ) <> jsonb_array_length(p_expected_podium)
     or (
       select count(distinct x.placement)
         from jsonb_to_recordset(p_expected_podium) as x(placement integer)
     ) <> jsonb_array_length(p_expected_podium)
     or exists (
       select 1 from public.tournament_podium podium
        where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
          and not exists (
            select 1
              from jsonb_to_recordset(p_expected_podium) as x(placement integer, team_id text, source text)
             where x.placement = podium.placement
               and x.team_id = podium.team_id::text
               and upper(coalesce(x.source, '')) = upper(coalesce(podium.source, ''))
          )
     )
     or exists (
       select 1
         from jsonb_to_recordset(p_expected_podium) as x(placement integer, team_id text, source text)
        where x.placement not between 1 and 3 or nullif(x.team_id, '') is null or not exists (
          select 1 from public.tournament_podium podium
           where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
             and podium.placement = x.placement
             and podium.team_id::text = x.team_id
             and upper(coalesce(podium.source, '')) = upper(coalesce(x.source, ''))
        )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_PODIUM_SNAPSHOT_STALE';
  end if;

  if (
    with derived as (
      select team.player1_id::bigint as player_id,
             case podium.placement
               when 1 then 'tournament_champion'
               when 2 then 'tournament_runner_up'
               when 3 then 'tournament_third_place'
             end as badge_id,
             p_tournament_id || ':draw:' || p_draw_id || ':podium:' || podium.placement::text as context_id
        from public.tournament_podium podium
        join public.tournament_teams team on team.id = podium.team_id
       where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
         and team.player1_id is not null
      union all
      select team.player2_id::bigint,
             case podium.placement
               when 1 then 'tournament_champion'
               when 2 then 'tournament_runner_up'
               when 3 then 'tournament_third_place'
             end,
             p_tournament_id || ':draw:' || p_draw_id || ':podium:' || podium.placement::text
        from public.tournament_podium podium
        join public.tournament_teams team on team.id = podium.team_id
       where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
         and team.player2_id is not null
    ), expected as (
      select x.player_id, x.badge_id, x.context_id
        from jsonb_to_recordset(p_expected_awards) as x(player_id bigint, badge_id text, context_id text)
    )
    select (select count(*) from derived) <> (select count(*) from expected)
        or (select count(*) from expected) <> (select count(*) from (select distinct * from expected) distinct_expected)
        or exists (select * from derived except select * from expected)
        or exists (select * from expected except select * from derived)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_PLAN_STALE';
  end if;

  if exists (
       select 1
         from jsonb_to_recordset(p_badges) as badge(club_id text, player_id bigint, badge_id text, context_type text, context_id text)
        where badge.club_id <> p_club_id
           or badge.context_type <> 'tournament'
           or not exists (
             select 1
               from jsonb_to_recordset(p_expected_awards) as expected(player_id bigint, badge_id text, context_id text)
              where expected.player_id = badge.player_id
                and expected.badge_id = badge.badge_id
                and expected.context_id = badge.context_id
           )
     )
     or (
       select count(distinct (badge.player_id, badge.badge_id, badge.context_id))
         from jsonb_to_recordset(p_badges) as badge(player_id bigint, badge_id text, context_id text)
     ) <> jsonb_array_length(p_badges) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_PLAN_STALE';
  end if;
  if exists (
    select 1 from public.player_badges badge
     where badge.club_id = p_club_id
       and badge.context_type = 'tournament'
       and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_ALREADY_EXISTS';
  end if;

  with inserted as (
    insert into public.player_badges (
      id, club_id, player_id, badge_id, earned_at, context_type,
      context_id, match_id, value_num, value_json
    )
    select
      coalesce(nullif(x.id, '')::uuid, gen_random_uuid()),
      p_club_id,
      x.player_id,
      x.badge_id,
      coalesce(x.earned_at, clock_timestamp()),
      'tournament',
      x.context_id,
      x.match_id,
      x.value_num,
      coalesce(x.value_json, '{}'::jsonb)
    from jsonb_to_recordset(p_badges) as x(
      id text, player_id bigint, badge_id text, earned_at timestamptz,
      context_id text, match_id text, value_num numeric, value_json jsonb
    )
    returning *
  )
  select coalesce(jsonb_agg(to_jsonb(inserted) order by inserted.context_id, inserted.badge_id, inserted.player_id), '[]'::jsonb)
    into v_saved from inserted;
  if jsonb_array_length(v_saved) <> jsonb_array_length(p_expected_awards) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_AWARD_PLAN_STALE';
  end if;
  return jsonb_build_object('ok', true, 'badges', v_saved);
end;
$$;

revoke all on function public.admin_award_tournament_draw_podium_cas(
  text, text, text, timestamptz, jsonb, jsonb, jsonb, jsonb
) from public, anon, authenticated;
grant execute on function public.admin_award_tournament_draw_podium_cas(
  text, text, text, timestamptz, jsonb, jsonb, jsonb, jsonb
) to service_role;

-- Only one official publish may own a club's rating domain at a time. This is
-- intentionally broader than the draw lock because players and league ratings
-- are shared across every draw and every rated-match ingestion surface.
create unique index if not exists idx_tournament_admin_operations_active_official_rating
  on public.tournament_admin_operations (club_id)
  where action in ('ops_official_publish', 'tournament_live_official_publish')
    and status in ('intent', 'mutated', 'recovery_required');

create or replace function public.block_tournament_publish_metadata_change()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_tournament_id text;
  v_club_id text;
begin
  if tg_table_name = 'tournaments' then
    v_tournament_id := old.id::text;
    v_club_id := old.club_id;
  else
    v_tournament_id := old.tournament_id::text;
    select tournament.club_id into v_club_id
      from public.tournaments tournament
     where tournament.id::text = v_tournament_id;
  end if;
  if exists (
    select 1
      from public.tournament_admin_operations operation
      join public.tournament_event_draws draw
        on draw.id::text = operation.entity_id
       and draw.tournament_id::text = v_tournament_id
     where operation.club_id = v_club_id
       and operation.action in ('ops_official_publish', 'tournament_live_official_publish')
       and operation.status in ('intent', 'mutated', 'recovery_required')
       and (
         tg_table_name = 'tournaments'
         or draw.event_option_id::text = old.id::text
       )
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_METADATA_LOCK';
  end if;
  if tg_op = 'DELETE' then return old; end if;
  return new;
end;
$$;

drop trigger if exists trg_00_tournament_official_publish_metadata_lock on public.tournaments;
create trigger trg_00_tournament_official_publish_metadata_lock
before update or delete on public.tournaments
for each row execute function public.block_tournament_publish_metadata_change();

drop trigger if exists trg_00_tournament_event_option_official_publish_metadata_lock
  on public.tournament_event_options;
create trigger trg_00_tournament_event_option_official_publish_metadata_lock
before update or delete on public.tournament_event_options
for each row execute function public.block_tournament_publish_metadata_change();

revoke all on function public.block_tournament_publish_metadata_change()
  from public, anon, authenticated;
grant execute on function public.block_tournament_publish_metadata_change()
  to service_role;

-- A non-CAS writer is rejected at its first write while official publish owns
-- the active claim. In particular, a legacy processor cannot insert a match
-- and then become blocked only at its later player update. The CAS RPC sets a
-- transaction-local operation identity that permits only its own exact plan.
create or replace function public.block_rating_change_during_tournament_official_publish()
returns trigger
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_club_id text;
  v_operation_key text;
  v_transaction_operation_key text;
begin
  if tg_op = 'DELETE' then
    v_club_id := old.club_id;
  else
    v_club_id := new.club_id;
  end if;
  select operation.operation_key into v_operation_key
    from public.tournament_admin_operations operation
   where operation.club_id = v_club_id
     and operation.action in ('ops_official_publish', 'tournament_live_official_publish')
     and operation.status in ('intent', 'mutated', 'recovery_required')
   limit 1;
  if v_operation_key is null then
    if tg_op = 'DELETE' then return old; end if;
    return new;
  end if;
  v_transaction_operation_key := current_setting('jupr.official_publish_operation_key', true);
  if coalesce(v_transaction_operation_key, '') <> v_operation_key then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_RATING_LOCK';
  end if;
  if tg_op = 'DELETE' then return old; end if;
  return new;
end;
$$;

drop trigger if exists trg_00_matches_official_publish_rating_lock on public.matches;
create trigger trg_00_matches_official_publish_rating_lock
before insert or update or delete on public.matches
for each row execute function public.block_rating_change_during_tournament_official_publish();

drop trigger if exists trg_00_players_official_publish_rating_lock on public.players;
create trigger trg_00_players_official_publish_rating_lock
before insert or update or delete on public.players
for each row execute function public.block_rating_change_during_tournament_official_publish();

drop trigger if exists trg_00_league_ratings_official_publish_rating_lock on public.league_ratings;
create trigger trg_00_league_ratings_official_publish_rating_lock
before insert or update or delete on public.league_ratings
for each row execute function public.block_rating_change_during_tournament_official_publish();

drop trigger if exists trg_00_leagues_metadata_official_publish_rating_lock on public.leagues_metadata;
create trigger trg_00_leagues_metadata_official_publish_rating_lock
before insert or update or delete on public.leagues_metadata
for each row execute function public.block_rating_change_during_tournament_official_publish();

revoke all on function public.block_rating_change_during_tournament_official_publish()
  from public, anon, authenticated;
grant execute on function public.block_rating_change_during_tournament_official_publish()
  to service_role;

alter table public.players
  add column if not exists singles_rating double precision,
  add column if not exists singles_wins integer not null default 0,
  add column if not exists singles_losses integer not null default 0,
  add column if not exists singles_matches_played integer not null default 0,
  add column if not exists singles_last_game_at timestamptz;
alter table public.matches
  add column if not exists match_format text not null default 'doubles',
  add column if not exists rating_bonus_elo double precision not null default 0,
  add column if not exists rating_bonus_reason text;
alter table public.matches
  alter column t1_p2 drop not null,
  alter column t2_p2 drop not null,
  alter column t1_p2_r drop not null,
  alter column t2_p2_r drop not null,
  alter column t1_p2_r_end drop not null,
  alter column t2_p2_r_end drop not null;

drop function if exists public.admin_apply_tournament_official_rating_plan_cas(
  text, text, text, text, text, text, jsonb, jsonb, jsonb, jsonb, jsonb
);
create or replace function public.admin_apply_tournament_official_rating_plan_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_operation_key text,
  p_request_fingerprint text,
  p_publish_plan_fingerprint text,
  p_publish_plan jsonb,
  p_match_rows jsonb,
  p_player_updates jsonb,
  p_league_rating_updates jsonb,
  p_league_metadata_expectations jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
  v_operation public.tournament_admin_operations%rowtype;
  v_tournament public.tournaments%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_item record;
  v_player public.players%rowtype;
  v_league_rating public.league_ratings%rowtype;
  v_meta public.leagues_metadata%rowtype;
  v_inserted integer := 0;
  v_row_count integer := 0;
  v_expected jsonb;
  v_after jsonb;
begin
  if nullif(btrim(p_operation_key), '') is null
     or nullif(btrim(p_request_fingerprint), '') is null
     or nullif(btrim(p_publish_plan_fingerprint), '') is null
     or jsonb_typeof(coalesce(p_publish_plan, '{}'::jsonb)) <> 'object'
     or jsonb_typeof(coalesce(p_match_rows, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_match_rows, '[]'::jsonb)) = 0
     or jsonb_typeof(coalesce(p_player_updates, '[]'::jsonb)) <> 'array'
     or jsonb_array_length(coalesce(p_player_updates, '[]'::jsonb)) = 0
     or jsonb_typeof(coalesce(p_league_rating_updates, '[]'::jsonb)) <> 'array'
     or jsonb_typeof(coalesce(p_league_metadata_expectations, '[]'::jsonb)) <> 'array' then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAN_INVALID';
  end if;

  select operation.* into v_operation
    from public.tournament_admin_operations operation
   where operation.club_id = p_club_id
     and operation.operation_key = p_operation_key
     and operation.request_fingerprint = p_request_fingerprint
     and operation.entity_type = 'tournament_event_draw'
     and operation.entity_id = p_draw_id
     and operation.action in ('ops_official_publish', 'tournament_live_official_publish')
     and operation.status = 'intent'
   for update;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_OPERATION_STALE';
  end if;
  if p_publish_plan is distinct from v_operation.request_json #> '{payload,publish_plan}' then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAN_INVALID';
  end if;
  if jsonb_array_length(p_match_rows) <> coalesce((p_publish_plan->>'match_count')::integer, -1)
     or (select count(distinct item.tournament_game_id) from jsonb_to_recordset(p_match_rows) as item(tournament_game_id text))
          <> jsonb_array_length(p_match_rows)
     or exists (
       select item.tournament_game_id from jsonb_to_recordset(p_match_rows) as item(tournament_game_id text)
       except
       select jsonb_array_elements_text(coalesce(p_publish_plan->'tournament_game_ids', '[]'::jsonb))
     )
     or exists (
       select jsonb_array_elements_text(coalesce(p_publish_plan->'tournament_game_ids', '[]'::jsonb))
       except
       select item.tournament_game_id from jsonb_to_recordset(p_match_rows) as item(tournament_game_id text)
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAN_INVALID';
  end if;
  if (
    with actual as (
      select jsonb_build_object(
        'club_id', coalesce(row.value->>'club_id', p_club_id),
        'date', coalesce(row.value->>'date', ''),
        'league', coalesce(row.value->>'league', ''),
        'week_tag', coalesce(row.value->>'week_tag', ''),
        'match_type', coalesce(row.value->>'match_type', ''),
        'match_format', coalesce(row.value->>'match_format', 'doubles'),
        't1_p1', (row.value->>'t1_p1')::bigint,
        't1_p2', (row.value->>'t1_p2')::bigint,
        't2_p1', (row.value->>'t2_p1')::bigint,
        't2_p2', (row.value->>'t2_p2')::bigint,
        'score_t1', (row.value->>'score_t1')::integer,
        'score_t2', (row.value->>'score_t2')::integer,
        'context_type', coalesce(row.value->>'context_type', ''),
        'context_id', coalesce(row.value->>'context_id', ''),
        'tournament_id', coalesce(row.value->>'tournament_id', ''),
        'tournament_game_id', coalesce(row.value->>'tournament_game_id', ''),
        'rating_scope', coalesce(row.value->>'rating_scope', ''),
        'rating_bonus_elo', coalesce((row.value->>'rating_bonus_elo')::double precision, 0),
        'rating_bonus_reason', coalesce(row.value->>'rating_bonus_reason', '')
      ) as projection
      from jsonb_array_elements(p_match_rows) as row(value)
    ), expected as (
      select row.value as projection
        from jsonb_array_elements(coalesce(p_publish_plan->'match_payload_projections', '[]'::jsonb)) as row(value)
    )
    select (select count(*) from actual) <> (select count(*) from expected)
        or exists (select projection from actual except select projection from expected)
        or exists (select projection from expected except select projection from actual)
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_MATCH_PAYLOAD_STALE';
  end if;
  perform set_config('jupr.official_publish_operation_key', p_operation_key, true);

  select tournament.* into v_tournament
    from public.tournaments tournament
   where tournament.id::text = p_tournament_id
     and tournament.club_id = p_club_id
   for update;
  if not found
     or v_tournament.name is distinct from p_publish_plan #>> '{tournament_metadata,name}'
     or v_tournament.status is distinct from p_publish_plan #>> '{tournament_metadata,status}'
     or v_tournament.start_date is distinct from nullif(p_publish_plan #>> '{tournament_metadata,start_date}', '')::date
     or v_tournament.end_date is distinct from nullif(p_publish_plan #>> '{tournament_metadata,end_date}', '')::date
     or v_tournament.updated_at is distinct from nullif(p_publish_plan #>> '{tournament_metadata,updated_at}', '')::timestamptz then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_TOURNAMENT_STALE';
  end if;

  select draw.* into v_draw
    from public.tournament_event_draws draw
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and draw.updated_at = nullif(p_publish_plan->>'draw_updated_at', '')::timestamptz
   for update;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_DRAW_STALE';
  end if;

  perform team.id
    from public.tournament_teams team
   where team.tournament_id::text = p_tournament_id and team.draw_id::text = p_draw_id
   order by team.id for no key update;
  if (select count(*) from public.tournament_teams team where team.tournament_id::text = p_tournament_id and team.draw_id::text = p_draw_id)
       <> jsonb_array_length(coalesce(p_publish_plan->'team_versions', '[]'::jsonb))
     or (select count(distinct expected.id) from jsonb_to_recordset(coalesce(p_publish_plan->'team_versions', '[]'::jsonb)) as expected(id text))
          <> jsonb_array_length(coalesce(p_publish_plan->'team_versions', '[]'::jsonb))
     or exists (
       select 1 from public.tournament_teams team
        where team.tournament_id::text = p_tournament_id and team.draw_id::text = p_draw_id
          and not exists (
            select 1 from jsonb_to_recordset(coalesce(p_publish_plan->'team_versions', '[]'::jsonb)) as expected(id text, updated_at timestamptz)
             where expected.id = team.id::text and expected.updated_at = team.updated_at
          )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_TEAM_STALE';
  end if;

  perform game.id
    from public.tournament_games game
   where game.tournament_id::text = p_tournament_id and game.draw_id::text = p_draw_id
   order by game.id for no key update;
  if (select count(*) from public.tournament_games game where game.tournament_id::text = p_tournament_id and game.draw_id::text = p_draw_id)
       <> jsonb_array_length(coalesce(p_publish_plan->'game_versions', '[]'::jsonb))
     or (select count(distinct expected.id) from jsonb_to_recordset(coalesce(p_publish_plan->'game_versions', '[]'::jsonb)) as expected(id text))
          <> jsonb_array_length(coalesce(p_publish_plan->'game_versions', '[]'::jsonb))
     or exists (
       select 1 from public.tournament_games game
        where game.tournament_id::text = p_tournament_id and game.draw_id::text = p_draw_id
          and not exists (
            select 1 from jsonb_to_recordset(coalesce(p_publish_plan->'game_versions', '[]'::jsonb)) as expected(id text, updated_at timestamptz)
             where expected.id = game.id::text and expected.updated_at = game.updated_at
          )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_GAME_STALE';
  end if;

  if p_publish_plan->'event_option_metadata' is null
     or p_publish_plan->'event_option_metadata' = 'null'::jsonb then
    if v_draw.event_option_id is not null then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_EVENT_OPTION_STALE';
    end if;
  else
    perform option.id from public.tournament_event_options option
     where option.id::text = p_publish_plan #>> '{event_option_metadata,id}'
       and option.tournament_id::text = p_tournament_id
       and option.id::text = v_draw.event_option_id::text
       and option.label is not distinct from p_publish_plan #>> '{event_option_metadata,label}'
       and option.event_family_label is not distinct from p_publish_plan #>> '{event_option_metadata,event_family_label}'
       and option.division_name is not distinct from p_publish_plan #>> '{event_option_metadata,division_name}'
     for update;
    if not found then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_EVENT_OPTION_STALE';
    end if;
  end if;

  if (select count(distinct item.player_id) from jsonb_to_recordset(p_player_updates) as item(player_id bigint))
       <> jsonb_array_length(p_player_updates) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_PLAN_INVALID';
  end if;

  if (select count(distinct (item.player_id, item.league_name)) from jsonb_to_recordset(p_league_rating_updates) as item(player_id bigint, league_name text))
       <> jsonb_array_length(p_league_rating_updates)
     or (select count(distinct item.league_name) from jsonb_to_recordset(p_league_metadata_expectations) as item(league_name text))
       <> jsonb_array_length(p_league_metadata_expectations) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_PLAN_INVALID';
  end if;
  perform player.id
    from public.players player
    join jsonb_to_recordset(p_player_updates) as item(player_id bigint) on item.player_id = player.id
   where player.club_id = p_club_id
   order by player.id for update;

  for v_item in
    select * from jsonb_to_recordset(p_player_updates)
      as item(player_id bigint, rating_mode text, expected jsonb, after jsonb)
  loop
    select player.* into v_player from public.players player
     where player.club_id = p_club_id and player.id = v_item.player_id
     for update;
    if not found then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_STALE';
    end if;
    v_expected := v_item.expected;
    if v_item.rating_mode = 'doubles' then
      if v_player.rating is distinct from (v_expected->>'rating')::double precision
         or v_player.wins is distinct from (v_expected->>'wins')::integer
         or v_player.losses is distinct from (v_expected->>'losses')::integer
         or v_player.matches_played is distinct from (v_expected->>'matches_played')::integer
         or v_player.last_game_at is distinct from nullif(v_expected->>'last_game_at', '')::timestamptz
         or v_player.inactive_at is distinct from nullif(v_expected->>'inactive_at', '')::timestamptz
         or v_player.active is distinct from (v_expected->>'active')::boolean then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_STALE';
      end if;
    elsif v_item.rating_mode = 'singles' then
      if v_player.singles_rating is distinct from (v_expected->>'singles_rating')::double precision
         or v_player.singles_wins is distinct from (v_expected->>'singles_wins')::integer
         or v_player.singles_losses is distinct from (v_expected->>'singles_losses')::integer
         or v_player.singles_matches_played is distinct from (v_expected->>'singles_matches_played')::integer
         or v_player.singles_last_game_at is distinct from nullif(v_expected->>'singles_last_game_at', '')::timestamptz
         or v_player.last_game_at is distinct from nullif(v_expected->>'last_game_at', '')::timestamptz
         or v_player.inactive_at is distinct from nullif(v_expected->>'inactive_at', '')::timestamptz
         or v_player.active is distinct from (v_expected->>'active')::boolean then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_STALE';
      end if;
    else
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_PLAN_INVALID';
    end if;
  end loop;

  for v_item in
    select * from jsonb_to_recordset(p_league_metadata_expectations)
      as item(league_name text, expected jsonb)
  loop
    select metadata.* into v_meta from public.leagues_metadata metadata
     where metadata.club_id = p_club_id and lower(metadata.league_name) = lower(v_item.league_name)
     for update;
    if v_item.expected is null then
      if found then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_METADATA_STALE';
      end if;
    elsif not found
       or v_meta.id::text is distinct from v_item.expected->>'id'
       or v_meta.league_name is distinct from v_item.expected->>'league_name'
       or v_meta.k_factor is distinct from (v_item.expected->>'k_factor')::integer then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_METADATA_STALE';
    end if;
  end loop;

  perform rating.id
    from public.league_ratings rating
    join jsonb_to_recordset(p_league_rating_updates) as item(player_id bigint, league_name text)
      on item.player_id = rating.player_id and item.league_name = rating.league_name
   where rating.club_id = p_club_id
   order by rating.id for update;
  for v_item in
    select * from jsonb_to_recordset(p_league_rating_updates)
      as item(player_id bigint, league_name text, expected jsonb, after jsonb)
  loop
    select rating.* into v_league_rating from public.league_ratings rating
     where rating.club_id = p_club_id
       and rating.player_id = v_item.player_id
       and rating.league_name = v_item.league_name
     for update;
    if v_item.expected is null then
      if found then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_RATING_STALE';
      end if;
    elsif not found
       or v_league_rating.id is distinct from (v_item.expected->>'id')::bigint
       or v_league_rating.rating is distinct from (v_item.expected->>'rating')::double precision
       or v_league_rating.wins is distinct from (v_item.expected->>'wins')::integer
       or v_league_rating.losses is distinct from (v_item.expected->>'losses')::integer
       or v_league_rating.matches_played is distinct from (v_item.expected->>'matches_played')::integer
       or v_league_rating.starting_rating is distinct from (v_item.expected->>'starting_rating')::double precision
       or v_league_rating.is_active is distinct from (v_item.expected->>'is_active')::boolean
       or v_league_rating.inactive_at is distinct from nullif(v_item.expected->>'inactive_at', '')::timestamptz then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_RATING_STALE';
    end if;
  end loop;

  if exists (
    select 1 from public.matches match
     where match.club_id = p_club_id
       and match.tournament_game_id::text in (
         select item.tournament_game_id from jsonb_to_recordset(p_match_rows) as item(tournament_game_id text)
       )
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_MATCH_EXISTS';
  end if;

  insert into public.matches (
    club_id, date, league, t1_p1, t1_p2, t2_p1, t2_p2, score_t1, score_t2,
    elo_delta, match_type, week_tag, t1_p1_r, t1_p2_r, t2_p1_r, t2_p2_r,
    t1_p1_r_end, t1_p2_r_end, t2_p1_r_end, t2_p2_r_end,
    context_type, context_id, tournament_id, tournament_game_id, rating_scope,
    match_format, rating_bonus_elo, rating_bonus_reason
  )
  select
    p_club_id, item.date, item.league, item.t1_p1, item.t1_p2, item.t2_p1, item.t2_p2,
    item.score_t1, item.score_t2, item.elo_delta, item.match_type, item.week_tag,
    item.t1_p1_r, item.t1_p2_r, item.t2_p1_r, item.t2_p2_r,
    item.t1_p1_r_end, item.t1_p2_r_end, item.t2_p1_r_end, item.t2_p2_r_end,
    item.context_type, nullif(item.context_id, '')::text, p_tournament_id::uuid,
    item.tournament_game_id::uuid, item.rating_scope, item.match_format,
    coalesce(item.rating_bonus_elo, 0), item.rating_bonus_reason
  from jsonb_to_recordset(p_match_rows) as item(
    club_id text, date timestamptz, league text,
    t1_p1 bigint, t1_p2 bigint, t2_p1 bigint, t2_p2 bigint,
    score_t1 integer, score_t2 integer, elo_delta double precision,
    match_type text, week_tag text,
    t1_p1_r double precision, t1_p2_r double precision, t2_p1_r double precision, t2_p2_r double precision,
    t1_p1_r_end double precision, t1_p2_r_end double precision, t2_p1_r_end double precision, t2_p2_r_end double precision,
    context_type text, context_id text, tournament_id text, tournament_game_id text,
    rating_scope text, match_format text, rating_bonus_elo double precision, rating_bonus_reason text
  );
  get diagnostics v_inserted = row_count;
  if v_inserted <> jsonb_array_length(p_match_rows) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_MATCH_INSERT_INCOMPLETE';
  end if;

  for v_item in
    select * from jsonb_to_recordset(p_player_updates)
      as item(player_id bigint, rating_mode text, expected jsonb, after jsonb)
  loop
    v_after := v_item.after;
    if v_item.rating_mode = 'doubles' then
      update public.players set
        rating = (v_after->>'rating')::double precision,
        wins = (v_after->>'wins')::integer,
        losses = (v_after->>'losses')::integer,
        matches_played = (v_after->>'matches_played')::integer,
        last_game_at = nullif(v_after->>'last_game_at', '')::timestamptz,
        inactive_at = nullif(v_after->>'inactive_at', '')::timestamptz,
        active = (v_after->>'active')::boolean
       where club_id = p_club_id and id = v_item.player_id;
    else
      update public.players set
        singles_rating = (v_after->>'singles_rating')::double precision,
        singles_wins = (v_after->>'singles_wins')::integer,
        singles_losses = (v_after->>'singles_losses')::integer,
        singles_matches_played = (v_after->>'singles_matches_played')::integer,
        singles_last_game_at = nullif(v_after->>'singles_last_game_at', '')::timestamptz,
        last_game_at = nullif(v_after->>'last_game_at', '')::timestamptz,
        inactive_at = nullif(v_after->>'inactive_at', '')::timestamptz,
        active = (v_after->>'active')::boolean
       where club_id = p_club_id and id = v_item.player_id;
    end if;
    get diagnostics v_row_count = row_count;
    if v_row_count <> 1 then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_PLAYER_WRITE_INCOMPLETE';
    end if;
  end loop;

  for v_item in
    select * from jsonb_to_recordset(p_league_rating_updates)
      as item(player_id bigint, league_name text, expected jsonb, after jsonb)
  loop
    v_after := v_item.after;
    if v_item.expected is null then
      insert into public.league_ratings (
        club_id, player_id, league_name, rating, wins, losses, matches_played,
        starting_rating, is_active, inactive_at
      ) values (
        p_club_id, v_item.player_id, v_item.league_name,
        (v_after->>'rating')::double precision,
        (v_after->>'wins')::integer,
        (v_after->>'losses')::integer,
        (v_after->>'matches_played')::integer,
        (v_after->>'starting_rating')::double precision,
        (v_after->>'is_active')::boolean,
        nullif(v_after->>'inactive_at', '')::timestamptz
      );
    else
      update public.league_ratings set
        rating = (v_after->>'rating')::double precision,
        wins = (v_after->>'wins')::integer,
        losses = (v_after->>'losses')::integer,
        matches_played = (v_after->>'matches_played')::integer,
        starting_rating = (v_after->>'starting_rating')::double precision,
        is_active = (v_after->>'is_active')::boolean,
        inactive_at = nullif(v_after->>'inactive_at', '')::timestamptz
       where id = (v_item.expected->>'id')::bigint and club_id = p_club_id;
    end if;
    get diagnostics v_row_count = row_count;
    if v_row_count <> 1 then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OFFICIAL_PUBLISH_LEAGUE_RATING_WRITE_INCOMPLETE';
    end if;
  end loop;

  return jsonb_build_object(
    'ok', true,
    'inserted', v_inserted,
    'operation_key', p_operation_key,
    'request_fingerprint', p_request_fingerprint,
    'publish_plan_fingerprint', p_publish_plan_fingerprint,
    'player_update_count', jsonb_array_length(p_player_updates),
    'league_rating_update_count', jsonb_array_length(p_league_rating_updates)
  );
end;
$$;

revoke all on function public.admin_apply_tournament_official_rating_plan_cas(
  text, text, text, text, text, text, jsonb, jsonb, jsonb, jsonb, jsonb
) from public, anon, authenticated;
grant execute on function public.admin_apply_tournament_official_rating_plan_cas(
  text, text, text, text, text, text, jsonb, jsonb, jsonb, jsonb, jsonb
) to service_role;

notify pgrst, 'reload schema';
