-- Pools belong to one draw, so playoff/medal identity stays division-scoped.
-- A reviewed split is immutable once scheduling starts. No existing draws change.
alter table public.tournament_event_draws
  add column if not exists pool_config jsonb null;

comment on column public.tournament_event_draws.pool_config is
  'Versioned pool membership and advancement counts; null means a whole-draw round robin. Team UUIDs belong to this exact draw.';

create or replace function public.assert_tournament_draw_pool_config(
  p_draw_id uuid, p_config jsonb
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_pool jsonb;
  v_team_ids text[] := array[]::text[];
  v_ids text[] := array[]::text[];
  v_names text[] := array[]::text[];
  v_members text[];
  v_advance integer;
  v_total integer := 0;
  v_size integer;
  v_roster text[];
begin
  if p_config is null then return; end if;
  if pg_catalog.jsonb_typeof(p_config) is distinct from 'object'
     or pg_catalog.jsonb_typeof(p_config->'version') is distinct from 'number'
     or p_config->>'version' is distinct from '1'
     or pg_catalog.jsonb_typeof(p_config->'pools') is distinct from 'array' then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_POOL_CONFIG_INVALID: invalid version or pools.';
  end if;
  if pg_catalog.jsonb_array_length(p_config->'pools') not between 2 and 8 then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_POOL_CONFIG_INVALID: use two through eight pools.';
  end if;
  for v_pool in select value from pg_catalog.jsonb_array_elements(p_config->'pools') loop
    if pg_catalog.jsonb_typeof(v_pool) is distinct from 'object'
       or pg_catalog.jsonb_typeof(v_pool->'id') is distinct from 'string'
       or coalesce(v_pool->>'id', '') !~ '^[A-Za-z0-9_-]{1,32}$'
       or pg_catalog.jsonb_typeof(v_pool->'name') is distinct from 'string'
       or pg_catalog.length(pg_catalog.btrim(v_pool->>'name')) not between 1 and 80
       or pg_catalog.jsonb_typeof(v_pool->'team_ids') is distinct from 'array'
       or pg_catalog.jsonb_typeof(v_pool->'advance_count') is distinct from 'number'
       or coalesce(v_pool->>'advance_count', '') !~ '^[0-9]+$' then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_POOL_CONFIG_INVALID: invalid pool fields.';
    end if;
    if (v_pool->>'id') = any(v_ids)
       or pg_catalog.lower(pg_catalog.btrim(v_pool->>'name')) = any(v_names) then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_POOL_CONFIG_INVALID: duplicate pool.';
    end if;
    v_ids := pg_catalog.array_append(v_ids, v_pool->>'id');
    v_names := pg_catalog.array_append(v_names, pg_catalog.lower(pg_catalog.btrim(v_pool->>'name')));
    if exists (select 1 from pg_catalog.jsonb_array_elements(v_pool->'team_ids') member
               where pg_catalog.jsonb_typeof(member) is distinct from 'string') then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_POOL_CONFIG_INVALID: team IDs must be strings.';
    end if;
    select pg_catalog.array_agg(value) into v_members
      from pg_catalog.jsonb_array_elements_text(v_pool->'team_ids');
    v_size := coalesce(pg_catalog.cardinality(v_members), 0);
    if v_size not between 2 and 16 or pg_catalog.length(v_pool->>'advance_count') > 2 then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_POOL_CONFIG_INVALID: invalid pool size or advancement.';
    end if;
    v_advance := (v_pool->>'advance_count')::integer;
    if v_advance > v_size then
      raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_POOL_CONFIG_INVALID: advancement exceeds pool size.';
    end if;
    v_total := v_total + v_advance;
    v_team_ids := v_team_ids || v_members;
  end loop;
  select pg_catalog.array_agg(team.id::text order by team.id::text) into v_roster
    from public.tournament_teams team where team.draw_id = p_draw_id;
  if pg_catalog.cardinality(v_team_ids) not between 4 and 16
     or v_total not between 2 and 16
     or pg_catalog.cardinality(v_team_ids) <> (select pg_catalog.count(distinct member) from pg_catalog.unnest(v_team_ids) member)
     or (select pg_catalog.array_agg(member order by member) from pg_catalog.unnest(v_team_ids) member) is distinct from v_roster then
    raise exception using errcode = '22023', message = 'JUPR_TOURNAMENT_POOL_CONFIG_INVALID: assign every draw team exactly once and advance at least two teams.';
  end if;
end
$function$;

create or replace function public.guard_tournament_draw_pool_config()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  new.pool_config := nullif(new.pool_config, 'null'::jsonb);
  if tg_op = 'UPDATE' then
    if new.pool_config is not distinct from old.pool_config then return new; end if;
  elsif new.pool_config is null then
    return new;
  end if;
  if exists (select 1 from public.tournament_games where draw_id = new.id)
     or exists (select 1 from public.tournament_podium where draw_id = new.id)
     or exists (select 1 from public.tournament_day_live_draws where draw_id = new.id)
     or exists (select 1 from public.tournament_day_live_queue where draw_id = new.id) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_POOLS_LOCKED: save pool assignments before games or live play.';
  end if;
  perform public.assert_tournament_draw_pool_config(new.id, new.pool_config);
  new.updated_at := pg_catalog.clock_timestamp();
  return new;
end
$function$;

drop trigger if exists trg_guard_tournament_draw_pool_config on public.tournament_event_draws;
create trigger trg_guard_tournament_draw_pool_config
  before insert or update of pool_config on public.tournament_event_draws
  for each row execute function public.guard_tournament_draw_pool_config();

create or replace function public.guard_tournament_pool_roster_identity()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_old_draw uuid;
  v_new_draw uuid;
  v_has_pools boolean;
begin
  if tg_op <> 'INSERT' then v_old_draw := old.draw_id; end if;
  if tg_op <> 'DELETE' then v_new_draw := new.draw_id; end if;
  if tg_op = 'UPDATE' then
    -- Player substitutions preserve the team and pool; other guards validate them.
    if old.id = new.id and old.draw_id is not distinct from new.draw_id
       and old.tournament_id = new.tournament_id then return new; end if;
  end if;
  perform draw.id from public.tournament_event_draws draw
    where draw.id in (v_old_draw, v_new_draw) order by draw.id for update;
  select exists (select 1 from public.tournament_event_draws draw
                 where draw.id in (v_old_draw, v_new_draw) and draw.pool_config is not null)
    into v_has_pools;
  if v_has_pools then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_POOL_ROSTER_LOCKED: remove the pool split before replacing or importing teams.';
  end if;
  if tg_op = 'DELETE' then return old; end if;
  return new;
end
$function$;

drop trigger if exists trg_guard_tournament_pool_roster_identity on public.tournament_teams;
create trigger trg_guard_tournament_pool_roster_identity
  before insert or update or delete on public.tournament_teams
  for each row execute function public.guard_tournament_pool_roster_identity();

create or replace function public.tournament_pool_pair_allowed(p_config jsonb, p_team_a text, p_team_b text)
returns boolean
language sql
immutable
security invoker
set search_path = ''
as $function$
  select p_config is null or (
    p_team_a is not null and p_team_b is not null and p_team_a <> p_team_b
    and exists (select 1 from pg_catalog.jsonb_array_elements(p_config->'pools') pool
                 where (pool->'team_ids') ? p_team_a and (pool->'team_ids') ? p_team_b)
  );
$function$;

create or replace function public.guard_tournament_pool_game_pair()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare v_config jsonb;
begin
  if pg_catalog.upper(coalesce(new.stage, '')) <> 'ROUND_ROBIN' then return new; end if;
  select draw.pool_config into v_config from public.tournament_event_draws draw
    where draw.id = new.draw_id for update;
  if not public.tournament_pool_pair_allowed(v_config, new.team_a_id::text, new.team_b_id::text) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_POOL_PAIR_INVALID: round-robin opponents must be in the same pool.';
  end if;
  return new;
end
$function$;

drop trigger if exists trg_guard_tournament_pool_game_pair on public.tournament_games;
create trigger trg_guard_tournament_pool_game_pair
  before insert or update of draw_id, stage, team_a_id, team_b_id on public.tournament_games
  for each row execute function public.guard_tournament_pool_game_pair();

create or replace function public.admin_save_tournament_draw_pools_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_expected_teams jsonb,
  p_pool_config jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw public.tournament_event_draws%rowtype;
begin
  p_pool_config := nullif(p_pool_config, 'null'::jsonb);
  -- Match the established roster/game/draw locking order of schedule writes.
  perform team.id from public.tournament_teams team
    where team.draw_id::text = p_draw_id order by team.id for no key update;
  perform game.id from public.tournament_games game
    where game.draw_id::text = p_draw_id order by game.id for update;
  select draw.* into v_draw
    from public.tournament_event_draws draw
    join public.tournaments tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id and draw.tournament_id::text = p_tournament_id
     and tournament.club_id = p_club_id
     and pg_catalog.upper(coalesce(tournament.status, '')) not in ('COMPLETED', 'ARCHIVED')
     and pg_catalog.upper(coalesce(draw.status, '')) not in ('CANCELLED', 'CANCELED', 'INACTIVE', 'COMPLETED', 'ARCHIVED')
     and draw.updated_at = p_expected_draw_updated_at
   for update of draw;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  if pg_catalog.jsonb_typeof(p_expected_teams) is distinct from 'array' then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;
  if pg_catalog.jsonb_array_length(p_expected_teams) = 0
     or (select pg_catalog.count(*) from public.tournament_teams where draw_id = v_draw.id) <> pg_catalog.jsonb_array_length(p_expected_teams)
     or (select pg_catalog.count(distinct expected.id) from pg_catalog.jsonb_to_recordset(p_expected_teams) expected(id text)) <> pg_catalog.jsonb_array_length(p_expected_teams)
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_expected_teams) expected(id text, updated_at timestamptz)
       left join public.tournament_teams team on team.id::text = expected.id and team.draw_id = v_draw.id
        where team.id is null or expected.updated_at is null or expected.updated_at is distinct from team.updated_at
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;
  if exists (select 1 from public.tournament_games where draw_id = v_draw.id)
     or exists (select 1 from public.tournament_podium where draw_id = v_draw.id)
     or exists (select 1 from public.tournament_day_live_draws where draw_id = v_draw.id)
     or exists (select 1 from public.tournament_day_live_queue where draw_id = v_draw.id) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_POOLS_LOCKED';
  end if;
  perform public.assert_tournament_draw_pool_config(v_draw.id, p_pool_config);
  update public.tournament_event_draws set pool_config = p_pool_config, updated_at = pg_catalog.clock_timestamp()
    where id = v_draw.id returning * into v_draw;
  return pg_catalog.jsonb_build_object('ok', true, 'draw', pg_catalog.to_jsonb(v_draw));
end
$function$;

revoke all on function public.assert_tournament_draw_pool_config(uuid, jsonb) from public, anon, authenticated;
revoke all on function public.guard_tournament_draw_pool_config() from public, anon, authenticated;
revoke all on function public.guard_tournament_pool_roster_identity() from public, anon, authenticated;
revoke all on function public.tournament_pool_pair_allowed(jsonb, text, text) from public, anon, authenticated;
revoke all on function public.guard_tournament_pool_game_pair() from public, anon, authenticated;
revoke all on function public.admin_save_tournament_draw_pools_cas(text, text, text, timestamptz, jsonb, jsonb) from public, anon, authenticated;
grant execute on function public.assert_tournament_draw_pool_config(uuid, jsonb) to service_role;
grant execute on function public.guard_tournament_draw_pool_config() to service_role;
grant execute on function public.guard_tournament_pool_roster_identity() to service_role;
grant execute on function public.tournament_pool_pair_allowed(jsonb, text, text) to service_role;
grant execute on function public.guard_tournament_pool_game_pair() to service_role;
grant execute on function public.admin_save_tournament_draw_pools_cas(text, text, text, timestamptz, jsonb, jsonb) to service_role;

notify pgrst, 'reload schema';
