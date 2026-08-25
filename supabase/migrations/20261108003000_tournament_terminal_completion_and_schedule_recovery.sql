-- Atomic tournament completion evidence and guarded schedule/event recovery.
--
-- All callable functions are FastAPI/service-role only. They intentionally use
-- SECURITY INVOKER and an explicit empty search_path. The browser never owns a
-- Supabase service credential and cannot call these write seams directly.

do $migration_preflight$
begin
  if pg_catalog.to_regclass('public.tournaments') is null
     or pg_catalog.to_regclass('public.tournament_event_options') is null
     or pg_catalog.to_regclass('public.tournament_registration_selections') is null
     or pg_catalog.to_regclass('public.tournament_registration_team_links') is null
     or pg_catalog.to_regclass('public.tournament_registration_team_members') is null
     or pg_catalog.to_regclass('public.tournament_event_draws') is null
     or pg_catalog.to_regclass('public.tournament_teams') is null
     or pg_catalog.to_regclass('public.tournament_games') is null
     or pg_catalog.to_regclass('public.tournament_podium') is null
     or pg_catalog.to_regclass('public.matches') is null
     or pg_catalog.to_regclass('public.player_badges') is null
     or pg_catalog.to_regclass('public.admin_activity_log') is null
     or pg_catalog.to_regclass('public.tournament_admin_operations') is null
     or pg_catalog.to_regclass('public.match_exclusion_operations') is null
     or pg_catalog.to_regclass('public.replay_jobs') is null
     or pg_catalog.to_regclass('public.tournament_day_live_runs') is null
     or pg_catalog.to_regclass('public.tournament_day_live_draws') is null
     or pg_catalog.to_regclass('public.tournament_day_live_queue') is null then
    raise exception using
      errcode = '42P01',
      message = 'Tournament terminal/recovery dependencies are missing.';
  end if;
end
$migration_preflight$;

-- Generic guarded fixture repair. This deliberately contains no staging UUID.
-- A clearly labelled women's division may not retain the legacy MEN value.
update public.tournament_event_options as event
   set gender_restriction = 'WOMEN'
 where pg_catalog.concat_ws(
         ' ', event.event_family_label, event.division_name
       ) ~* '(^|[^[:alpha:]])women(''s|’s)?([^[:alpha:]]|$)'
   and pg_catalog.upper(coalesce(event.gender_restriction, '')) = 'MEN';

-- The executable RR contract is 4..16. Raising a sub-minimum capacity cannot
-- remove an entrant. A >16 capacity is reduced only when neither registrations
-- nor already-built teams would be invalidated; populated oversized divisions
-- remain visible for an explicit supported split/migration decision.
update public.tournament_event_options as event
   set capacity_teams = 4
 where event.capacity_teams is not null
   and event.capacity_teams < 4;

update public.tournament_event_options as event
   set capacity_teams = 16
 where event.capacity_teams > 16
   and (
     select pg_catalog.count(*)
       from public.tournament_registration_selections as selection
      where selection.tournament_id = event.tournament_id
        and selection.event_option_id = event.id
   ) <= 16
   and (
     select pg_catalog.count(*)
       from public.tournament_teams as team
      where team.tournament_id::text = event.tournament_id::text
        and team.event_option_id = event.id
   ) <= 16;

create table if not exists public.tournament_lifecycle_receipts (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  tournament_id uuid not null references public.tournaments(id) on delete restrict,
  action text not null,
  from_status text not null,
  to_status text not null,
  operation_key text not null references public.tournament_admin_operations(operation_key) on delete restrict,
  request_fingerprint text not null,
  snapshot_fingerprint text null,
  evidence_fingerprint text not null,
  evidence_json jsonb not null,
  created_by text not null,
  created_at timestamptz not null default pg_catalog.clock_timestamp(),
  constraint tournament_lifecycle_receipt_action_chk
    check (action in ('complete', 'archive', 'unarchive')),
  constraint tournament_lifecycle_receipt_status_chk
    check (
      (action = 'complete' and from_status = 'ACTIVE' and to_status = 'COMPLETED')
      or (action = 'archive' and from_status = 'COMPLETED' and to_status = 'ARCHIVED')
      or (action = 'unarchive' and from_status = 'ARCHIVED' and to_status = 'COMPLETED')
    ),
  constraint tournament_lifecycle_receipt_operation_key_chk
    check (pg_catalog.char_length(operation_key) = 64),
  constraint tournament_lifecycle_receipt_request_fingerprint_chk
    check (request_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint tournament_lifecycle_receipt_snapshot_fingerprint_chk
    check (snapshot_fingerprint is null or snapshot_fingerprint ~ '^[0-9a-f]{32}$'),
  constraint tournament_lifecycle_receipt_evidence_fingerprint_chk
    check (evidence_fingerprint ~ '^[0-9a-f]{64}$'),
  constraint tournament_lifecycle_receipt_operation_unique unique (operation_key)
);

create index if not exists idx_tournament_lifecycle_receipts_tournament_created
  on public.tournament_lifecycle_receipts (club_id, tournament_id, created_at desc);
create index if not exists idx_tournament_lifecycle_receipts_tournament_fk
  on public.tournament_lifecycle_receipts (tournament_id);

alter table public.tournament_lifecycle_receipts enable row level security;
alter table public.tournament_lifecycle_receipts force row level security;
revoke all on table public.tournament_lifecycle_receipts
  from public, anon, authenticated, service_role;
grant select, insert on table public.tournament_lifecycle_receipts to service_role;

comment on table public.tournament_lifecycle_receipts is
  'FastAPI-private immutable proof committed atomically with tournament terminal status.';

create schema if not exists private;
revoke all on schema private from public, anon, authenticated;
grant usage on schema private to service_role;

create or replace function private.tournament_completion_snapshot_fingerprint(
  p_club_id text,
  p_tournament_id text,
  p_ignore_operation_key text default null
)
returns text
language sql
stable
security invoker
set search_path = ''
as $function$
  select pg_catalog.md5(
    pg_catalog.jsonb_build_object(
      'tournament', coalesce((
        select pg_catalog.to_jsonb(t)
          from public.tournaments as t
         where t.id::text = p_tournament_id and t.club_id = p_club_id
      ), '{}'::jsonb),
      'event_options', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(e) order by e.id::text)
          from public.tournament_event_options as e
         where e.tournament_id::text = p_tournament_id
      ), '[]'::jsonb),
      'draws', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(d) order by d.id::text)
          from public.tournament_event_draws as d
         where d.tournament_id::text = p_tournament_id
      ), '[]'::jsonb),
      'teams', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(team) order by team.id::text)
          from public.tournament_teams as team
         where team.tournament_id::text = p_tournament_id
      ), '[]'::jsonb),
      'games', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(game) order by game.id::text)
          from public.tournament_games as game
         where game.tournament_id::text = p_tournament_id
      ), '[]'::jsonb),
      'podium', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(podium) order by podium.id::text)
          from public.tournament_podium as podium
         where podium.tournament_id::text = p_tournament_id
      ), '[]'::jsonb),
      'matches', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(match) order by match.id::text)
          from public.matches as match
         where match.club_id = p_club_id
           and match.tournament_id::text = p_tournament_id
      ), '[]'::jsonb),
      'awards', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(badge) order by badge.id::text)
          from public.player_badges as badge
         where badge.club_id = p_club_id
           and badge.context_type = 'tournament'
           and badge.context_id::text like p_tournament_id || ':%'
      ), '[]'::jsonb),
      'podium_reviews', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(activity) order by activity.id)
          from public.admin_activity_log as activity
         where activity.club_id = p_club_id
           and activity.entity_type = 'tournament_event_draw'
           and activity.action_type = 'review_tournament_draw_podium_admin'
           and exists (
             select 1 from public.tournament_event_draws as draw
              where draw.tournament_id::text = p_tournament_id
                and draw.id::text = activity.entity_id
           )
      ), '[]'::jsonb),
      'operations', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(operation) order by operation.operation_key)
          from public.tournament_admin_operations as operation
         where operation.club_id = p_club_id
           and (
             operation.lock_scope = p_tournament_id
             or operation.lock_scope like 'tournament:' || p_tournament_id || ':%'
           )
           and operation.operation_key <> coalesce(p_ignore_operation_key, '')
      ), '[]'::jsonb),
      'unsettled_exclusions', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(operation) order by operation.id::text)
          from public.match_exclusion_operations as operation
         where operation.club_id = p_club_id
           and operation.status in ('pending_replay', 'pending_badge_reconcile', 'recovery_required')
           and exists (
             select 1
               from public.matches as tournament_match
              where tournament_match.club_id = p_club_id
                and tournament_match.tournament_id::text = p_tournament_id
                and (
                  tournament_match.id = any(
                    coalesce(operation.excluded_match_ids, '{}'::bigint[])
                  )
                  or exists (
                    select 1
                      from pg_catalog.jsonb_array_elements(
                        case
                          when pg_catalog.jsonb_typeof(operation.targets_json) = 'array'
                            then operation.targets_json
                          else '[]'::jsonb
                        end
                      ) as target(value)
                     where target.value ->> 'match_id' = tournament_match.id::text
                  )
                )
           )
      ), '[]'::jsonb),
      'unsettled_replays', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(job) order by job.id::text)
          from public.replay_jobs as job
         where job.club_id = p_club_id
           and job.status in ('pending', 'running')
           and exists (
             select 1
              from public.match_exclusion_operations as operation
              where operation.club_id = p_club_id
                and operation.status in ('pending_replay', 'pending_badge_reconcile', 'recovery_required')
                and operation.replay_job_id = job.id
                and exists (
                  select 1
                    from public.matches as tournament_match
                   where tournament_match.club_id = p_club_id
                     and tournament_match.tournament_id::text = p_tournament_id
                     and (
                       tournament_match.id = any(
                         coalesce(operation.excluded_match_ids, '{}'::bigint[])
                       )
                       or exists (
                         select 1
                           from pg_catalog.jsonb_array_elements(
                             case
                               when pg_catalog.jsonb_typeof(operation.targets_json) = 'array'
                                 then operation.targets_json
                               else '[]'::jsonb
                             end
                           ) as target(value)
                          where target.value ->> 'match_id' = tournament_match.id::text
                       )
                     )
                )
           )
      ), '[]'::jsonb),
      'day_live', coalesce((
        select pg_catalog.jsonb_agg(pg_catalog.to_jsonb(run) order by run.id::text)
          from public.tournament_day_live_runs as run
         where run.tournament_id::text = p_tournament_id
      ), '[]'::jsonb)
    )::text
  )
$function$;

revoke all on function private.tournament_completion_snapshot_fingerprint(text, text, text)
  from public, anon, authenticated;
grant execute on function private.tournament_completion_snapshot_fingerprint(text, text, text)
  to service_role;

create or replace function public.admin_tournament_completion_snapshot(
  p_club_id text,
  p_tournament_id text,
  p_ignore_operation_key text default null
)
returns jsonb
language plpgsql
stable
security invoker
set search_path = ''
as $function$
declare
  v_fingerprint text;
begin
  if not exists (
    select 1 from public.tournaments as tournament
     where tournament.club_id = p_club_id
       and tournament.id::text = p_tournament_id
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_STALE';
  end if;
  v_fingerprint := private.tournament_completion_snapshot_fingerprint(
    p_club_id, p_tournament_id, p_ignore_operation_key
  );
  return pg_catalog.jsonb_build_object(
    'ok', true,
    'snapshot', pg_catalog.jsonb_build_object(
      'contract', 'jupr:tournament-completion-snapshot:v1',
      'club_id', p_club_id,
      'tournament_id', p_tournament_id,
      'snapshot_fingerprint', v_fingerprint
    )
  );
end
$function$;

create or replace function public.admin_reconcile_tournament_round_robin_games_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_expected_teams jsonb,
  p_expected_source_games jsonb,
  p_games jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_team_count integer;
  v_existing_count integer;
  v_expected_game_count integer;
  v_saved jsonb;
begin
  lock table public.tournament_teams in share mode;
  -- This RPC writes tournament_games. A self-conflicting table lock prevents
  -- two recovery calls from each holding SHARE and deadlocking on lock upgrade.
  lock table public.tournament_games in share row exclusive mode;
  lock table public.tournament_podium in share mode;
  lock table public.matches in share mode;
  lock table public.player_badges in share mode;
  lock table public.tournament_day_live_draws in share mode;
  lock table public.tournament_day_live_queue in share mode;

  select draw.* into v_draw
    from public.tournament_event_draws as draw
    join public.tournaments as tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id = p_club_id
     and pg_catalog.upper(coalesce(tournament.status, ''))
         not in ('COMPLETED', 'ARCHIVED')
     and draw.updated_at = p_expected_draw_updated_at
   for update of draw;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;
  perform private.assert_tournament_draw_recovery_snapshot(
    p_tournament_id, p_draw_id, p_expected_teams, p_expected_source_games
  );
  perform private.assert_tournament_draw_recovery_dependencies_clear(
    p_club_id, p_tournament_id, p_draw_id
  );

  select pg_catalog.count(*) into v_team_count
    from public.tournament_teams as team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id;
  select pg_catalog.count(*) into v_existing_count
    from public.tournament_games as game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id;
  v_expected_game_count := v_team_count * (v_team_count - 1) / 2;

  -- Existing official-looking results are preserved exactly, but every result
  -- must be internally complete and every pair must belong to the current
  -- roster. One finalized game in a 21-game partial draw therefore survives.
  if v_existing_count = 0
     or exists (
       select 1 from public.tournament_games as game
       left join public.tournament_teams as team_a
         on team_a.id = game.team_a_id
        and team_a.tournament_id = v_draw.tournament_id and team_a.draw_id = v_draw.id
       left join public.tournament_teams as team_b
         on team_b.id = game.team_b_id
        and team_b.tournament_id = v_draw.tournament_id and team_b.draw_id = v_draw.id
       where game.tournament_id = v_draw.tournament_id
         and game.draw_id = v_draw.id
         and (
           pg_catalog.upper(coalesce(game.stage, '')) <> 'ROUND_ROBIN'
           or team_a.id is null or team_b.id is null or team_a.id = team_b.id
           or not (
             (
               game.score_a is null and game.score_b is null
               and game.winner_team_id is null and game.loser_team_id is null
               and game.finalized_at is null
             )
             or (
               game.score_a is not null and game.score_b is not null
               and game.score_a >= 0 and game.score_b >= 0 and game.score_a <> game.score_b
               and game.finalized_at is not null
               and game.winner_team_id = case when game.score_a > game.score_b then game.team_a_id else game.team_b_id end
               and game.loser_team_id = case when game.score_a > game.score_b then game.team_b_id else game.team_a_id end
             )
           )
         )
     )
     or exists (
       select 1 from public.tournament_games as game
        where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
        group by least(game.team_a_id::text, game.team_b_id::text),
                 greatest(game.team_a_id::text, game.team_b_id::text)
       having pg_catalog.count(*) <> 1
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_RECONCILE_BLOCKED';
  end if;

  if pg_catalog.jsonb_typeof(coalesce(p_games, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_games) = 0
     or v_existing_count + pg_catalog.jsonb_array_length(p_games) <> v_expected_game_count
     or (
       select pg_catalog.count(distinct payload.id)
         from pg_catalog.jsonb_to_recordset(p_games) as payload(id text)
     ) <> pg_catalog.jsonb_array_length(p_games)
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(p_games) as payload(
           id text, stage text, rr_round_number integer, rr_slot_number integer,
           team_a_id text, team_b_id text, score_a integer, score_b integer,
           winner_team_id text, loser_team_id text, finalized_at timestamptz
         )
         left join public.tournament_teams as team_a
           on team_a.id::text = payload.team_a_id
          and team_a.tournament_id = v_draw.tournament_id and team_a.draw_id = v_draw.id
         left join public.tournament_teams as team_b
           on team_b.id::text = payload.team_b_id
          and team_b.tournament_id = v_draw.tournament_id and team_b.draw_id = v_draw.id
        where nullif(payload.id, '') is null
           or pg_catalog.upper(coalesce(payload.stage, '')) <> 'ROUND_ROBIN'
           or payload.rr_round_number is null or payload.rr_round_number < 1
           or payload.rr_slot_number is null or payload.rr_slot_number < 1
           or team_a.id is null or team_b.id is null or team_a.id = team_b.id
           or payload.score_a is not null or payload.score_b is not null
           or payload.winner_team_id is not null or payload.loser_team_id is not null
           or payload.finalized_at is not null
           or exists (
             select 1 from public.tournament_games as existing
              where existing.tournament_id = v_draw.tournament_id
                and existing.draw_id = v_draw.id
                and least(existing.team_a_id::text, existing.team_b_id::text) = least(payload.team_a_id, payload.team_b_id)
                and greatest(existing.team_a_id::text, existing.team_b_id::text) = greatest(payload.team_a_id, payload.team_b_id)
           )
     )
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_games)
         as payload(team_a_id text, team_b_id text)
        group by least(payload.team_a_id, payload.team_b_id),
                 greatest(payload.team_a_id, payload.team_b_id)
       having pg_catalog.count(*) <> 1
     )
     or exists (
       select 1 from (
         select game.rr_round_number, game.rr_slot_number
           from public.tournament_games as game
          where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         union all
         select payload.rr_round_number, payload.rr_slot_number
           from pg_catalog.jsonb_to_recordset(p_games)
             as payload(rr_round_number integer, rr_slot_number integer)
       ) as schedule
       group by schedule.rr_round_number, schedule.rr_slot_number
       having pg_catalog.count(*) <> 1
     )
     or exists (
       select 1 from (
         select game.rr_round_number, game.team_a_id::text as team_id
           from public.tournament_games as game
          where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         union all
         select game.rr_round_number, game.team_b_id::text
           from public.tournament_games as game
          where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         union all
         select payload.rr_round_number, participant.team_id
           from pg_catalog.jsonb_to_recordset(p_games)
             as payload(rr_round_number integer, team_a_id text, team_b_id text)
           cross join lateral (values (payload.team_a_id), (payload.team_b_id)) as participant(team_id)
       ) as appearances
       group by appearances.rr_round_number, appearances.team_id
       having pg_catalog.count(*) > 1
     )
     or exists (
       select 1
         from public.tournament_teams as team_a
         join public.tournament_teams as team_b
           on team_b.tournament_id = team_a.tournament_id
          and team_b.draw_id = team_a.draw_id
          and team_a.id < team_b.id
        where team_a.tournament_id = v_draw.tournament_id
          and team_a.draw_id = v_draw.id
          and not exists (
            select 1 from (
              select game.team_a_id::text as team_a_id, game.team_b_id::text as team_b_id
                from public.tournament_games as game
               where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
              union all
              select payload.team_a_id, payload.team_b_id
                from pg_catalog.jsonb_to_recordset(p_games)
                  as payload(team_a_id text, team_b_id text)
            ) as combined
            where least(combined.team_a_id, combined.team_b_id) = least(team_a.id::text, team_b.id::text)
              and greatest(combined.team_a_id, combined.team_b_id) = greatest(team_a.id::text, team_b.id::text)
          )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_RECONCILE_BLOCKED';
  end if;

  insert into public.tournament_games (
    id, tournament_id, draw_id, registration_day_id, event_option_id, stage,
    rr_round_number, rr_slot_number, team_a_id, team_b_id, score_a, score_b,
    winner_team_id, loser_team_id, finalized_at, created_at, updated_at
  )
  select
    payload.id::uuid, v_draw.tournament_id, v_draw.id,
    v_draw.registration_day_id, v_draw.event_option_id,
    'ROUND_ROBIN', payload.rr_round_number, payload.rr_slot_number,
    payload.team_a_id::uuid, payload.team_b_id::uuid,
    null, null, null, null, null,
    coalesce(payload.created_at, pg_catalog.clock_timestamp()),
    pg_catalog.clock_timestamp()
  from pg_catalog.jsonb_to_recordset(p_games) as payload(
    id text, registration_day_id text, event_option_id text, rr_round_number integer,
    rr_slot_number integer, team_a_id text, team_b_id text, created_at timestamptz
  );

  select coalesce(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(game) order by game.rr_round_number, game.rr_slot_number),
    '[]'::jsonb
  ) into v_saved
    from public.tournament_games as game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id;
  return pg_catalog.jsonb_build_object('ok', true, 'games', v_saved);
exception
  when raise_exception then
    if sqlerrm = 'JUPR_TOURNAMENT_DRAW_RECOVERY_DEPENDENCY_BLOCKED' then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_RECONCILE_BLOCKED';
    end if;
    raise;
end
$function$;

create or replace function public.admin_transition_tournament_terminal_status_cas(
  p_club_id text,
  p_tournament_id text,
  p_action text,
  p_expected_updated_at timestamptz,
  p_operation_key text,
  p_request_fingerprint text,
  p_snapshot_fingerprint text,
  p_evidence_fingerprint text,
  p_evidence_json jsonb,
  p_actor text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_action text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_action, '')));
  v_tournament public.tournaments%rowtype;
  v_operation public.tournament_admin_operations%rowtype;
  v_receipt public.tournament_lifecycle_receipts%rowtype;
  v_from_status text;
  v_to_status text;
  v_current_snapshot text;
begin
  if v_action not in ('complete', 'archive', 'unarchive')
     or p_operation_key !~ '^[0-9a-f]{64}$'
     or p_request_fingerprint !~ '^[0-9a-f]{64}$'
     or p_evidence_fingerprint !~ '^[0-9a-f]{64}$'
     or nullif(pg_catalog.btrim(p_actor), '') is null then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OPERATION_INVALID';
  end if;

  select receipt.* into v_receipt
    from public.tournament_lifecycle_receipts as receipt
   where receipt.operation_key = p_operation_key
     and receipt.club_id = p_club_id
     and receipt.tournament_id::text = p_tournament_id
     and receipt.action = v_action;
  if found then
    select tournament.* into v_tournament
      from public.tournaments as tournament
     where tournament.club_id = p_club_id
       and tournament.id::text = p_tournament_id;
    if found and pg_catalog.upper(v_tournament.status) = v_receipt.to_status then
      return pg_catalog.jsonb_build_object(
        'ok', true,
        'idempotent', true,
        'tournament', pg_catalog.to_jsonb(v_tournament),
        'receipt', pg_catalog.to_jsonb(v_receipt)
      );
    end if;
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OPERATION_INVALID';
  end if;

  select operation.* into v_operation
    from public.tournament_admin_operations as operation
   where operation.operation_key = p_operation_key
     and operation.club_id = p_club_id
     and operation.surface = 'tournament'
     and operation.action = 'tournament_' || v_action
     and operation.entity_type = 'tournament'
     and operation.entity_id = p_tournament_id
     and operation.lock_scope = p_tournament_id
     and operation.request_fingerprint = p_request_fingerprint
     and operation.status in ('intent', 'mutated', 'recovery_required')
   for update;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OPERATION_INVALID';
  end if;

  -- A same-key request may have waited on the durable operation row after its
  -- first receipt read. Re-read under that serialization point so concurrent
  -- idempotent calls return the committed receipt instead of failing stale CAS.
  select receipt.* into v_receipt
    from public.tournament_lifecycle_receipts as receipt
   where receipt.operation_key = p_operation_key
     and receipt.club_id = p_club_id
     and receipt.tournament_id::text = p_tournament_id
     and receipt.action = v_action;
  if found then
    select tournament.* into v_tournament
      from public.tournaments as tournament
     where tournament.club_id = p_club_id
       and tournament.id::text = p_tournament_id;
    if found and pg_catalog.upper(v_tournament.status) = v_receipt.to_status then
      return pg_catalog.jsonb_build_object(
        'ok', true,
        'idempotent', true,
        'tournament', pg_catalog.to_jsonb(v_tournament),
        'receipt', pg_catalog.to_jsonb(v_receipt)
      );
    end if;
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_OPERATION_INVALID';
  end if;

  if v_action = 'complete' then
    -- SHARE conflicts with INSERT/UPDATE/DELETE table locks. This makes the
    -- fingerprint comparison and terminal update one serializable closeout
    -- boundary while retaining caller privileges throughout.
    lock table public.tournament_event_options in share mode;
    lock table public.tournament_event_draws in share mode;
    lock table public.tournament_teams in share mode;
    lock table public.tournament_games in share mode;
    lock table public.tournament_podium in share mode;
    lock table public.matches in share mode;
    lock table public.player_badges in share mode;
    lock table public.admin_activity_log in share mode;
    lock table public.match_exclusion_operations in share mode;
    lock table public.replay_jobs in share mode;
    lock table public.tournament_day_live_runs in share mode;
    lock table public.tournament_admin_operations in share mode;
  end if;

  select tournament.* into v_tournament
    from public.tournaments as tournament
   where tournament.club_id = p_club_id
     and tournament.id::text = p_tournament_id
     and tournament.updated_at = p_expected_updated_at
   for update;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_STALE';
  end if;
  v_from_status := pg_catalog.upper(coalesce(v_tournament.status, ''));

  if v_action = 'complete' then
    if v_from_status <> 'ACTIVE'
       or p_evidence_json #>> '{domain_readiness,completion,ready}' <> 'true'
       or pg_catalog.jsonb_typeof(
            coalesce(
              p_evidence_json #> '{domain_readiness,completion,blockers}',
              'null'::jsonb
            )
          ) <> 'array'
       or pg_catalog.jsonb_array_length(
            p_evidence_json #> '{domain_readiness,completion,blockers}'
          ) <> 0
       or p_evidence_json ->> 'contract' <> 'jupr:tournament-lifecycle:v1'
       or p_evidence_json ->> 'club_id' <> p_club_id
       or p_evidence_json ->> 'tournament_id' <> p_tournament_id
       or p_snapshot_fingerprint !~ '^[0-9a-f]{32}$' then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_CLOSEOUT_NOT_READY';
    end if;
    if exists (
      select 1 from public.tournament_day_live_runs as run
       where run.tournament_id = v_tournament.id
         and run.state in ('ACTIVE', 'PAUSED')
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_CLOSEOUT_NOT_READY';
    end if;
    v_current_snapshot := private.tournament_completion_snapshot_fingerprint(
      p_club_id, p_tournament_id, p_operation_key
    );
    if v_current_snapshot <> p_snapshot_fingerprint then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_CLOSEOUT_SNAPSHOT_STALE';
    end if;
    v_to_status := 'COMPLETED';
  elsif v_action = 'archive' then
    if v_from_status <> 'COMPLETED' or not exists (
      select 1 from public.tournament_lifecycle_receipts as receipt
       where receipt.tournament_id = v_tournament.id
         and receipt.club_id = p_club_id
         and receipt.action = 'complete'
         and receipt.to_status = 'COMPLETED'
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_NOT_COMPLETED';
    end if;
    v_to_status := 'ARCHIVED';
  else
    if v_from_status <> 'ARCHIVED' or not exists (
      select 1 from public.tournament_lifecycle_receipts as receipt
       where receipt.tournament_id = v_tournament.id
         and receipt.club_id = p_club_id
         and receipt.action = 'complete'
         and receipt.to_status = 'COMPLETED'
    ) then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_NOT_ARCHIVED';
    end if;
    v_to_status := 'COMPLETED';
  end if;

  update public.tournaments as tournament
     set status = v_to_status,
         updated_at = pg_catalog.clock_timestamp()
   where tournament.id = v_tournament.id
   returning tournament.* into v_tournament;

  insert into public.tournament_lifecycle_receipts (
    club_id, tournament_id, action, from_status, to_status, operation_key,
    request_fingerprint, snapshot_fingerprint, evidence_fingerprint,
    evidence_json, created_by
  ) values (
    p_club_id, v_tournament.id, v_action, v_from_status, v_to_status,
    p_operation_key, p_request_fingerprint,
    nullif(p_snapshot_fingerprint, ''), p_evidence_fingerprint,
    p_evidence_json, p_actor
  ) returning * into v_receipt;

  return pg_catalog.jsonb_build_object(
    'ok', true,
    'idempotent', false,
    'tournament', pg_catalog.to_jsonb(v_tournament),
    'receipt', pg_catalog.to_jsonb(v_receipt)
  );
end
$function$;

revoke all on function public.admin_tournament_completion_snapshot(text, text, text)
  from public, anon, authenticated;
revoke all on function public.admin_transition_tournament_terminal_status_cas(
  text, text, text, timestamptz, text, text, text, text, jsonb, text
) from public, anon, authenticated;
grant execute on function public.admin_tournament_completion_snapshot(text, text, text)
  to service_role;
grant execute on function public.admin_transition_tournament_terminal_status_cas(
  text, text, text, timestamptz, text, text, text, text, jsonb, text
) to service_role;

create or replace function private.assert_tournament_draw_recovery_snapshot(
  p_tournament_id text,
  p_draw_id text,
  p_expected_teams jsonb,
  p_expected_source_games jsonb
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if pg_catalog.jsonb_typeof(coalesce(p_expected_teams, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_expected_teams) not between 4 and 16
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_expected_teams)
         as expected(id text, updated_at timestamptz)
        where nullif(expected.id, '') is null or expected.updated_at is null
     )
     or (
       select pg_catalog.count(distinct expected.id)
         from pg_catalog.jsonb_to_recordset(p_expected_teams) as expected(id text)
     ) <> pg_catalog.jsonb_array_length(p_expected_teams)
     or (
       select pg_catalog.count(*) from public.tournament_teams as team
        where team.tournament_id::text = p_tournament_id
          and team.draw_id::text = p_draw_id
     ) <> pg_catalog.jsonb_array_length(p_expected_teams)
     or exists (
       select 1 from public.tournament_teams as team
        where team.tournament_id::text = p_tournament_id
          and team.draw_id::text = p_draw_id
          and not exists (
            select 1 from pg_catalog.jsonb_to_recordset(p_expected_teams)
              as expected(id text, updated_at timestamptz)
             where expected.id = team.id::text
               and expected.updated_at = team.updated_at
          )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_TEAM_SNAPSHOT_STALE';
  end if;

  if pg_catalog.jsonb_typeof(coalesce(p_expected_source_games, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_expected_source_games) = 0
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_expected_source_games)
         as expected(id text, updated_at timestamptz)
        where nullif(expected.id, '') is null or expected.updated_at is null
     )
     or (
       select pg_catalog.count(distinct expected.id)
         from pg_catalog.jsonb_to_recordset(p_expected_source_games) as expected(id text)
     ) <> pg_catalog.jsonb_array_length(p_expected_source_games)
     or (
       select pg_catalog.count(*) from public.tournament_games as game
        where game.tournament_id::text = p_tournament_id
          and game.draw_id::text = p_draw_id
     ) <> pg_catalog.jsonb_array_length(p_expected_source_games)
     or exists (
       select 1 from public.tournament_games as game
        where game.tournament_id::text = p_tournament_id
          and game.draw_id::text = p_draw_id
          and not exists (
            select 1 from pg_catalog.jsonb_to_recordset(p_expected_source_games)
              as expected(id text, updated_at timestamptz)
             where expected.id = game.id::text
               and expected.updated_at = game.updated_at
          )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_SOURCE_GAME_SNAPSHOT_STALE';
  end if;
end
$function$;

create or replace function private.assert_tournament_draw_recovery_dependencies_clear(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
begin
  if exists (
       select 1 from public.tournament_podium as podium
        where podium.tournament_id::text = p_tournament_id
          and podium.draw_id::text = p_draw_id
     )
     or exists (
       select 1 from public.matches as match
        where match.club_id = p_club_id
          and match.tournament_id::text = p_tournament_id
          and exists (
            select 1 from public.tournament_games as game
             where game.id::text = match.tournament_game_id::text
               and game.draw_id::text = p_draw_id
          )
     )
     or exists (
       select 1 from public.player_badges as badge
        where badge.club_id = p_club_id
          and badge.context_type = 'tournament'
          and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':podium:%'
     )
     or exists (
       select 1 from public.tournament_day_live_draws as day_draw
        where day_draw.tournament_id::text = p_tournament_id
          and day_draw.draw_id::text = p_draw_id
     )
     or exists (
       select 1 from public.tournament_day_live_queue as queue
        where queue.tournament_id::text = p_tournament_id
          and queue.draw_id::text = p_draw_id
     ) then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DRAW_RECOVERY_DEPENDENCY_BLOCKED';
  end if;
end
$function$;

create or replace function public.admin_rebuild_tournament_round_robin_games_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz,
  p_expected_teams jsonb,
  p_expected_source_games jsonb,
  p_games jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw public.tournament_event_draws%rowtype;
  v_team_count integer;
  v_expected_game_count integer;
  v_saved jsonb;
begin
  lock table public.tournament_teams in share mode;
  -- This RPC writes tournament_games. A self-conflicting table lock prevents
  -- two recovery calls from each holding SHARE and deadlocking on lock upgrade.
  lock table public.tournament_games in share row exclusive mode;
  lock table public.tournament_podium in share mode;
  lock table public.matches in share mode;
  lock table public.player_badges in share mode;
  lock table public.tournament_day_live_draws in share mode;
  lock table public.tournament_day_live_queue in share mode;

  select draw.* into v_draw
    from public.tournament_event_draws as draw
    join public.tournaments as tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id = p_club_id
     and pg_catalog.upper(coalesce(tournament.status, ''))
         not in ('COMPLETED', 'ARCHIVED')
     and draw.updated_at = p_expected_draw_updated_at
   for update of draw;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  perform private.assert_tournament_draw_recovery_snapshot(
    p_tournament_id, p_draw_id, p_expected_teams, p_expected_source_games
  );
  perform private.assert_tournament_draw_recovery_dependencies_clear(
    p_club_id, p_tournament_id, p_draw_id
  );

  if exists (
    select 1 from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         pg_catalog.upper(coalesce(game.stage, '')) <> 'ROUND_ROBIN'
         or game.score_a is not null or game.score_b is not null
         or game.winner_team_id is not null or game.loser_team_id is not null
         or game.finalized_at is not null
       )
  ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_REBUILD_BLOCKED';
  end if;

  select pg_catalog.count(*) into v_team_count
    from public.tournament_teams as team
   where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id;
  v_expected_game_count := v_team_count * (v_team_count - 1) / 2;
  if pg_catalog.jsonb_typeof(coalesce(p_games, 'null'::jsonb)) <> 'array'
     or pg_catalog.jsonb_array_length(p_games) <> v_expected_game_count
     or (
       select pg_catalog.count(distinct payload.id)
         from pg_catalog.jsonb_to_recordset(p_games) as payload(id text)
     ) <> v_expected_game_count
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(p_games) as payload(
           id text, stage text, rr_round_number integer, rr_slot_number integer,
           team_a_id text, team_b_id text, score_a integer, score_b integer,
           winner_team_id text, loser_team_id text, finalized_at timestamptz
         )
         left join public.tournament_teams as team_a
           on team_a.id::text = payload.team_a_id
          and team_a.tournament_id = v_draw.tournament_id and team_a.draw_id = v_draw.id
         left join public.tournament_teams as team_b
           on team_b.id::text = payload.team_b_id
          and team_b.tournament_id = v_draw.tournament_id and team_b.draw_id = v_draw.id
        where nullif(payload.id, '') is null
           or pg_catalog.upper(coalesce(payload.stage, '')) <> 'ROUND_ROBIN'
           or payload.rr_round_number is null or payload.rr_round_number < 1
           or payload.rr_slot_number is null or payload.rr_slot_number < 1
           or team_a.id is null or team_b.id is null or team_a.id = team_b.id
           or payload.score_a is not null or payload.score_b is not null
           or payload.winner_team_id is not null or payload.loser_team_id is not null
           or payload.finalized_at is not null
     )
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_games)
         as payload(rr_round_number integer, rr_slot_number integer)
        group by payload.rr_round_number, payload.rr_slot_number having pg_catalog.count(*) <> 1
     )
     or exists (
       select 1 from pg_catalog.jsonb_to_recordset(p_games)
         as payload(team_a_id text, team_b_id text)
        group by least(payload.team_a_id, payload.team_b_id),
                 greatest(payload.team_a_id, payload.team_b_id)
       having pg_catalog.count(*) <> 1
     )
     or exists (
       select 1
         from pg_catalog.jsonb_to_recordset(p_games)
           as payload(rr_round_number integer, team_a_id text, team_b_id text)
         cross join lateral (values (payload.team_a_id), (payload.team_b_id)) as participant(team_id)
        group by payload.rr_round_number, participant.team_id
       having pg_catalog.count(*) > 1
     )
     or exists (
       select 1
         from public.tournament_teams as team_a
         join public.tournament_teams as team_b
           on team_b.tournament_id = team_a.tournament_id
          and team_b.draw_id = team_a.draw_id
          and team_a.id < team_b.id
        where team_a.tournament_id = v_draw.tournament_id
          and team_a.draw_id = v_draw.id
          and not exists (
            select 1 from pg_catalog.jsonb_to_recordset(p_games)
              as payload(team_a_id text, team_b_id text)
             where least(payload.team_a_id, payload.team_b_id) = least(team_a.id::text, team_b.id::text)
               and greatest(payload.team_a_id, payload.team_b_id) = greatest(team_a.id::text, team_b.id::text)
          )
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_REBUILD_BLOCKED';
  end if;

  delete from public.tournament_games as game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id;

  insert into public.tournament_games (
    id, tournament_id, draw_id, registration_day_id, event_option_id, stage,
    rr_round_number, rr_slot_number, team_a_id, team_b_id, score_a, score_b,
    winner_team_id, loser_team_id, finalized_at, created_at, updated_at
  )
  select
    payload.id::uuid, v_draw.tournament_id, v_draw.id,
    v_draw.registration_day_id, v_draw.event_option_id,
    'ROUND_ROBIN', payload.rr_round_number, payload.rr_slot_number,
    payload.team_a_id::uuid, payload.team_b_id::uuid,
    null, null, null, null, null,
    coalesce(payload.created_at, pg_catalog.clock_timestamp()),
    pg_catalog.clock_timestamp()
  from pg_catalog.jsonb_to_recordset(p_games) as payload(
    id text, registration_day_id text, event_option_id text, rr_round_number integer,
    rr_slot_number integer, team_a_id text, team_b_id text, created_at timestamptz
  );

  select coalesce(
    pg_catalog.jsonb_agg(pg_catalog.to_jsonb(game) order by game.rr_round_number, game.rr_slot_number),
    '[]'::jsonb
  ) into v_saved
    from public.tournament_games as game
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id;
  return pg_catalog.jsonb_build_object('ok', true, 'games', v_saved);
exception
  when raise_exception then
    if sqlerrm = 'JUPR_TOURNAMENT_DRAW_RECOVERY_DEPENDENCY_BLOCKED' then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_REBUILD_BLOCKED';
    end if;
    raise;
end
$function$;

create or replace function public.admin_cancel_empty_tournament_draw_cas(
  p_club_id text,
  p_tournament_id text,
  p_draw_id text,
  p_expected_draw_updated_at timestamptz
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw public.tournament_event_draws%rowtype;
begin
  -- Table locks close the empty-check/disable insert race. This is a rare
  -- recovery operation and retains the draw row as audit history.
  lock table public.tournament_teams in share mode;
  lock table public.tournament_games in share mode;
  lock table public.tournament_podium in share mode;
  lock table public.matches in share mode;
  lock table public.player_badges in share mode;
  lock table public.tournament_day_live_draws in share mode;
  lock table public.tournament_day_live_queue in share mode;

  select draw.* into v_draw
    from public.tournament_event_draws as draw
    join public.tournaments as tournament on tournament.id = draw.tournament_id
   where draw.id::text = p_draw_id
     and draw.tournament_id::text = p_tournament_id
     and tournament.club_id = p_club_id
     and pg_catalog.upper(coalesce(tournament.status, ''))
         not in ('COMPLETED', 'ARCHIVED')
     and draw.updated_at = p_expected_draw_updated_at
     and pg_catalog.lower(coalesce(draw.status, 'draft'))
         not in ('cancelled', 'canceled', 'inactive', 'disabled', 'archived')
   for update of draw;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_STALE';
  end if;

  if exists (
       select 1 from public.tournament_teams as team
        where team.tournament_id = v_draw.tournament_id and team.draw_id = v_draw.id
     )
     or exists (
       select 1 from public.tournament_games as game
        where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
     )
     or exists (
       select 1 from public.tournament_podium as podium
        where podium.tournament_id = v_draw.tournament_id and podium.draw_id = v_draw.id
     )
     or exists (
       select 1 from public.matches as match
        where match.club_id = p_club_id
          and match.tournament_id = v_draw.tournament_id
          and match.context_type = 'tournament_game'
          and exists (
            select 1 from public.tournament_games as game
             where game.id = match.tournament_game_id and game.draw_id = v_draw.id
          )
     )
     or exists (
       select 1 from public.player_badges as badge
        where badge.club_id = p_club_id
          and badge.context_type = 'tournament'
          and badge.context_id::text like p_tournament_id || ':draw:' || p_draw_id || ':%'
     )
     or exists (
       select 1 from public.tournament_day_live_draws as day_draw
        where day_draw.tournament_id = v_draw.tournament_id and day_draw.draw_id = v_draw.id
     )
     or exists (
       select 1 from public.tournament_day_live_queue as queue
        where queue.tournament_id = v_draw.tournament_id and queue.draw_id = v_draw.id
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_DRAW_NOT_EMPTY';
  end if;

  update public.tournament_event_draws as draw
     set status = 'cancelled', updated_at = pg_catalog.clock_timestamp()
   where draw.id = v_draw.id
   returning draw.* into v_draw;
  return pg_catalog.jsonb_build_object(
    'ok', true, 'draw', pg_catalog.to_jsonb(v_draw)
  );
end
$function$;

create or replace function public.admin_cancel_empty_tournament_event_cas(
  p_club_id text,
  p_tournament_id text,
  p_event_option_id text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_event public.tournament_event_options%rowtype;
begin
  -- Prevent a concurrent registration/team/draw insert from landing between
  -- the emptiness proof and the durable enabled=false transition.
  lock table public.tournament_registration_selections in share mode;
  lock table public.tournament_registration_team_links in share mode;
  lock table public.tournament_registration_team_members in share mode;
  lock table public.tournament_event_draws in share mode;
  lock table public.tournament_teams in share mode;
  lock table public.tournament_games in share mode;

  select event.* into v_event
    from public.tournament_event_options as event
    join public.tournaments as tournament
      on tournament.id::text = event.tournament_id::text
   where event.id = p_event_option_id
     and event.tournament_id::text = p_tournament_id
     and tournament.club_id = p_club_id
     and pg_catalog.upper(coalesce(tournament.status, ''))
         not in ('COMPLETED', 'ARCHIVED')
     and coalesce(event.enabled, true)
   for update of event;
  if not found then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_EVENT_STALE';
  end if;

  if exists (
       select 1 from public.tournament_registration_selections as selection
        where selection.tournament_id::text = p_tournament_id
          and selection.event_option_id = p_event_option_id
     )
     or exists (
       select 1 from public.tournament_registration_team_links as link
        where link.tournament_id::text = p_tournament_id
          and link.event_option_id = p_event_option_id
     )
     or exists (
       select 1 from public.tournament_registration_team_members as member
        where member.tournament_id::text = p_tournament_id
          and member.event_option_id = p_event_option_id
     )
     or exists (
       select 1 from public.tournament_event_draws as draw
        where draw.tournament_id::text = p_tournament_id
          and draw.event_option_id = p_event_option_id
     )
     or exists (
       select 1 from public.tournament_teams as team
        where team.tournament_id::text = p_tournament_id
          and team.event_option_id = p_event_option_id
     )
     or exists (
       select 1 from public.tournament_games as game
        where game.tournament_id::text = p_tournament_id
          and game.event_option_id = p_event_option_id
     ) then
    raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_EVENT_NOT_EMPTY';
  end if;

  update public.tournament_event_options as event
     set enabled = false, status = 'cancelled'
   where event.id = v_event.id
     and event.tournament_id::text = p_tournament_id
   returning event.* into v_event;
  return pg_catalog.jsonb_build_object(
    'ok', true, 'event_option', pg_catalog.to_jsonb(v_event)
  );
end
$function$;

revoke all on function private.assert_tournament_draw_recovery_snapshot(text, text, jsonb, jsonb)
  from public, anon, authenticated;
revoke all on function private.assert_tournament_draw_recovery_dependencies_clear(text, text, text)
  from public, anon, authenticated;
grant execute on function private.assert_tournament_draw_recovery_snapshot(text, text, jsonb, jsonb)
  to service_role;
grant execute on function private.assert_tournament_draw_recovery_dependencies_clear(text, text, text)
  to service_role;

revoke all on function public.admin_rebuild_tournament_round_robin_games_cas(
  text, text, text, timestamptz, jsonb, jsonb, jsonb
) from public, anon, authenticated;
revoke all on function public.admin_reconcile_tournament_round_robin_games_cas(
  text, text, text, timestamptz, jsonb, jsonb, jsonb
) from public, anon, authenticated;
revoke all on function public.admin_cancel_empty_tournament_draw_cas(
  text, text, text, timestamptz
) from public, anon, authenticated;
revoke all on function public.admin_cancel_empty_tournament_event_cas(text, text, text)
  from public, anon, authenticated;
grant execute on function public.admin_rebuild_tournament_round_robin_games_cas(
  text, text, text, timestamptz, jsonb, jsonb, jsonb
) to service_role;
grant execute on function public.admin_reconcile_tournament_round_robin_games_cas(
  text, text, text, timestamptz, jsonb, jsonb, jsonb
) to service_role;
grant execute on function public.admin_cancel_empty_tournament_draw_cas(
  text, text, text, timestamptz
) to service_role;
grant execute on function public.admin_cancel_empty_tournament_event_cas(text, text, text)
  to service_role;

notify pgrst, 'reload schema';
