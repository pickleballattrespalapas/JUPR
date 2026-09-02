-- Preserve every played game in a best-two-of-three tournament matchup.
--
-- The scheduled ROUND_ROBIN/PLAYOFF row remains the aggregate parent (2-0 or
-- 2-1) used by standings and bracket progression.  Each real game to 11 is a
-- finalized SERIES_GAME child and therefore remains a one-to-one canonical
-- Match Log/rating source under the existing tournament_game_id invariant.

alter table public.tournament_games
  add column if not exists series_parent_game_id uuid null,
  add column if not exists series_game_number smallint null;

-- Individual game scores cannot be reconstructed from a legacy aggregate
-- 2-0/2-1 result.  Refuse to silently grandfather such rows: an operator must
-- inspect and normalize them before this migration can establish its durable
-- parent/leaf invariant.
do $do$
declare
  v_legacy_parent_count bigint;
begin
  select count(*)
    into v_legacy_parent_count
    from public.tournament_games as game
    left join public.tournament_event_draws as draw
      on draw.id = game.draw_id
     and draw.tournament_id = game.tournament_id
    left join public.tournament_event_options as event
      on event.id::text = coalesce(
           nullif(game.event_option_id::text, ''),
           draw.event_option_id::text
         )
     and event.tournament_id = game.tournament_id::text
   where game.series_parent_game_id is null
     and pg_catalog.upper(coalesce(game.stage, '')) <> 'SERIES_GAME'
     and game.finalized_at is not null
     and pg_catalog.upper(coalesce(game.result_type, 'PLAYED')) = 'PLAYED'
     and pg_catalog.upper(
           coalesce(
             nullif(pg_catalog.btrim(game.scoring_format), ''),
             nullif(pg_catalog.btrim(event.scoring_override), ''),
             nullif(pg_catalog.btrim(event.scoring_default), ''),
             ''
           )
         ) = 'BEST_2_OF_3';
  if v_legacy_parent_count > 0 then
    raise exception using
      errcode = 'P0001',
      message = pg_catalog.format(
        'JUPR_TOURNAMENT_BEST_OF_THREE_LEGACY_DATA_REQUIRES_REVIEW: count=%s',
        v_legacy_parent_count
      );
  end if;
end
$do$;

do $do$
begin
  if not exists (
    select 1
      from pg_catalog.pg_constraint
     where conrelid = 'public.tournament_games'::regclass
       and conname = 'tournament_games_series_parent_fk'
  ) then
    alter table public.tournament_games
      add constraint tournament_games_series_parent_fk
      foreign key (series_parent_game_id)
      references public.tournament_games(id)
      on delete cascade;
  end if;
  if not exists (
    select 1
      from pg_catalog.pg_constraint
     where conrelid = 'public.tournament_games'::regclass
       and conname = 'tournament_games_series_identity_chk'
  ) then
    alter table public.tournament_games
      add constraint tournament_games_series_identity_chk
      check (
        (
          series_parent_game_id is null
          and series_game_number is null
          and pg_catalog.upper(coalesce(stage, '')) <> 'SERIES_GAME'
        )
        or
        (
          series_parent_game_id is not null
          and series_parent_game_id <> id
          and series_game_number is not null
          and series_game_number between 1 and 3
          and pg_catalog.upper(coalesce(stage, '')) = 'SERIES_GAME'
          and not parent_result_only
        )
      ) not valid;
  end if;
  if not exists (
    select 1
      from pg_catalog.pg_constraint
     where conrelid = 'public.matches'::regclass
       and conname = 'matches_tournament_game_fk'
  ) then
    alter table public.matches
      add constraint matches_tournament_game_fk
      foreign key (tournament_game_id)
      references public.tournament_games(id)
      on delete restrict
      not valid;
  end if;
end
$do$;

-- Older production snapshots can contain two kinds of durable history that
-- predate this parent/leaf model:
--
-- * playoff games stored directly as numbered PLAYOFF rows; and
-- * Match Log rows whose source tournament game was removed by a legacy
--   schedule rebuild.
--
-- Keep both constraints NOT VALID when that history is present. PostgreSQL
-- still enforces a NOT VALID CHECK/FK for every new or changed row, so current
-- writes fail closed while the historical rows remain untouched for an
-- owner-reviewed reconciliation. Validate immediately on clean databases.
do $do$
declare
  v_series_identity_violation_count bigint;
  v_match_tournament_game_orphan_count bigint;
begin
  select pg_catalog.count(*)
    into v_series_identity_violation_count
    from public.tournament_games as game
   where (
     (
       game.series_parent_game_id is null
       and game.series_game_number is null
       and pg_catalog.upper(coalesce(game.stage, '')) <> 'SERIES_GAME'
     )
     or
     (
       game.series_parent_game_id is not null
       and game.series_parent_game_id <> game.id
       and game.series_game_number is not null
       and game.series_game_number between 1 and 3
       and pg_catalog.upper(coalesce(game.stage, '')) = 'SERIES_GAME'
       and not game.parent_result_only
     )
   ) is false;

  if v_series_identity_violation_count = 0 then
    alter table public.tournament_games
      validate constraint tournament_games_series_identity_chk;
  else
    raise notice
      'JUPR_TOURNAMENT_SERIES_IDENTITY_LEGACY_ROWS_RETAINED: constraint=tournament_games_series_identity_chk count=% state=NOT_VALID owner_review_required=true',
      v_series_identity_violation_count;
  end if;

  select pg_catalog.count(*)
    into v_match_tournament_game_orphan_count
    from public.matches as match_row
    left join public.tournament_games as game
      on game.id = match_row.tournament_game_id
   where match_row.tournament_game_id is not null
     and game.id is null;

  if v_match_tournament_game_orphan_count = 0 then
    alter table public.matches
      validate constraint matches_tournament_game_fk;
  else
    raise notice
      'JUPR_TOURNAMENT_MATCH_GAME_ORPHANS_RETAINED: constraint=matches_tournament_game_fk count=% state=NOT_VALID owner_review_required=true',
      v_match_tournament_game_orphan_count;
  end if;
end
$do$;

comment on column public.tournament_games.series_parent_game_id is
  'Aggregate BEST_2_OF_3 tournament game whose real rating game this row represents.';
comment on column public.tournament_games.series_game_number is
  'One-based order of a real game inside a BEST_2_OF_3 aggregate parent.';

create unique index if not exists uq_tournament_games_series_parent_number
  on public.tournament_games (series_parent_game_id, series_game_number)
  where series_parent_game_id is not null;

create index if not exists idx_tournament_games_series_parent
  on public.tournament_games (series_parent_game_id)
  where series_parent_game_id is not null;

-- A child belongs to its parent's queue/court authority.  Resolving the
-- effective game id here closes the service-role escape hatch that would
-- otherwise allow a SERIES_GAME correction to bypass the active-day fence.
create or replace function public.guard_tournament_game_day_live_mutation()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_draw_id uuid := case when tg_op = 'DELETE' then old.draw_id else new.draw_id end;
  v_game_id uuid := case
    when tg_op = 'INSERT' then coalesce(new.series_parent_game_id, new.id)
    else coalesce(old.series_parent_game_id, old.id)
  end;
  v_operation_key text := nullif(
    pg_catalog.current_setting('jupr.day_live_operation_key', true),
    ''
  );
  v_fenced boolean := false;
  v_authorized boolean := false;
begin
  if v_draw_id is null then
    return case when tg_op = 'DELETE' then old else new end;
  end if;

  if tg_op = 'INSERT' then
    select exists (
      select 1
        from public.tournament_day_live_draws as day_draw
        join public.tournament_day_live_runs as run on run.id = day_draw.run_id
       where day_draw.draw_id = v_draw_id
         and day_draw.state in ('ACTIVE', 'PAUSED')
         and run.state in ('ACTIVE', 'PAUSED')
    ) into v_fenced;
  else
    select exists (
      select 1
        from public.tournament_day_live_queue as queue
        join public.tournament_day_live_runs as run on run.id = queue.run_id
       where queue.game_id = v_game_id
         and run.state in ('ACTIVE', 'PAUSED')
    ) into v_fenced;
  end if;

  if not v_fenced then
    return case when tg_op = 'DELETE' then old else new end;
  end if;

  if v_operation_key is not null then
    select exists (
      select 1
        from public.tournament_admin_operations as operation
        join public.tournament_day_live_runs as run
          on run.club_id = operation.club_id
         and operation.entity_type = 'tournament_registration_day'
         and operation.entity_id = pg_catalog.concat_ws(
           ':', run.tournament_id::text, run.registration_day_id
         )
         and operation.lock_scope = pg_catalog.concat_ws(
           ':', 'tournament', run.tournament_id::text,
           'day', run.registration_day_id
         )
        join public.tournament_day_live_draws as day_draw
          on day_draw.run_id = run.id and day_draw.draw_id = v_draw_id
       where operation.operation_key = v_operation_key
         and operation.surface = 'tournament_live'
         and operation.status = 'intent'
         and operation.action in (
           'tournament_day_live_score_and_release',
           'tournament_day_live_correct_completed_score',
           'tournament_day_live_record_non_played_result',
           'tournament_day_live_generate_playoffs'
         )
    ) into v_authorized;
  end if;

  if not v_authorized then
    raise exception using
      errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_SCORE_PATH_REQUIRED: use score_and_release for an actively queued game; day-fenced game generation must use its day command.';
  end if;
  return case when tg_op = 'DELETE' then old else new end;
end;
$function$;

revoke all on function public.guard_tournament_game_day_live_mutation()
  from public, anon, authenticated;
grant execute on function public.guard_tournament_game_day_live_mutation()
  to service_role;

-- Validate the normalized score-review envelope independently of the API and
-- mark only the aggregate parent as non-rating evidence.
create or replace function public.validate_tournament_best_of_three_result()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_format text;
  v_games jsonb;
  v_entry jsonb;
  v_review jsonb;
  v_parent public.tournament_games%rowtype;
  v_parent_entry jsonb;
  v_expected_number integer := 1;
  v_game_number integer;
  v_score_a integer;
  v_score_b integer;
  v_wins_a integer := 0;
  v_wins_b integer := 0;
  v_expected_winner uuid;
  v_expected_loser uuid;
  v_parent_allows_rating_children boolean := false;
begin
  if new.series_parent_game_id is not null then
    select parent.*
      into v_parent
      from public.tournament_games as parent
     where parent.id = new.series_parent_game_id
     for key share;
    v_parent_allows_rating_children := (
      pg_catalog.upper(coalesce(v_parent.result_type, 'PLAYED')) = 'PLAYED'
      and v_parent.parent_result_only
    ) or (
      pg_catalog.upper(coalesce(v_parent.result_type, '')) = 'RETIREMENT'
      and not v_parent.parent_result_only
      and v_parent.score_review_json
            ->>'retirement_completed_games_preserved' = 'true'
      and pg_catalog.jsonb_typeof(
            v_parent.score_review_json->'game_scores'
          ) = 'array'
    );
    v_parent_entry := v_parent.score_review_json
      -> 'game_scores'
      -> (new.series_game_number::integer - 1);
    v_expected_winner := case
      when new.score_a > new.score_b then new.team_a_id
      else new.team_b_id
    end;
    v_expected_loser := case
      when new.score_a > new.score_b then new.team_b_id
      else new.team_a_id
    end;
    if new.series_parent_game_id = new.id
       or v_parent.id is null
       or v_parent.series_parent_game_id is not null
       or v_parent.tournament_id is distinct from new.tournament_id
       or v_parent.draw_id is distinct from new.draw_id
       or v_parent.registration_day_id is distinct from new.registration_day_id
       or v_parent.event_option_id is distinct from new.event_option_id
       or v_parent.team_a_id is distinct from new.team_a_id
       or v_parent.team_b_id is distinct from new.team_b_id
       or v_parent.finalized_at is null
       or not v_parent_allows_rating_children
       or pg_catalog.upper(
            coalesce(
              v_parent.score_review_json->>'scoring_format',
              v_parent.scoring_format,
              ''
            )
          ) <> 'BEST_2_OF_3'
       or pg_catalog.upper(coalesce(new.stage, '')) <> 'SERIES_GAME'
       or pg_catalog.upper(coalesce(new.scoring_format, '')) <> 'GAME_TO_11'
       or pg_catalog.upper(coalesce(new.result_type, 'PLAYED')) <> 'PLAYED'
       or new.finalized_at is null
       or new.parent_result_only
       or new.series_game_number not between 1 and 3
       or new.score_a is null
       or new.score_b is null
       or new.score_a < 0
       or new.score_b < 0
       or new.score_a = new.score_b
       or greatest(new.score_a, new.score_b) < 11
       or pg_catalog.abs(new.score_a - new.score_b) < 2
       or pg_catalog.jsonb_typeof(v_parent_entry) is distinct from 'object'
       or (v_parent_entry->>'game_number')::integer
            is distinct from new.series_game_number
       or (v_parent_entry->>'score_a')::integer is distinct from new.score_a
       or (v_parent_entry->>'score_b')::integer is distinct from new.score_b
       or v_parent_entry->'score_review' is distinct from new.score_review_json
       or new.winner_team_id is distinct from v_expected_winner
       or new.loser_team_id is distinct from v_expected_loser then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_BEST_OF_THREE_CHILD_INVALID';
    end if;
    return new;
  end if;

  if tg_op = 'UPDATE'
     and old.series_parent_game_id is null
     and old.score_review_json
           ->>'retirement_completed_games_preserved' = 'true'
     and (
       pg_catalog.upper(coalesce(new.result_type, '')) <> 'RETIREMENT'
       or new.score_review_json
            ->>'retirement_completed_games_preserved' is distinct from 'true'
     ) then
    raise exception using errcode = '23514',
      message = 'JUPR_TOURNAMENT_RETIREMENT_RATING_EVIDENCE_IMMUTABLE';
  end if;

  v_format := pg_catalog.upper(
    coalesce(nullif(pg_catalog.btrim(new.score_review_json->>'scoring_format'), ''), '')
  );
  v_games := new.score_review_json->'game_scores';

  if v_format = 'BEST_2_OF_3'
     and pg_catalog.upper(coalesce(new.result_type, 'PLAYED')) = 'PLAYED'
     and new.finalized_at is not null then
    if new.score_review_json->>'accepted' is distinct from 'true'
       or pg_catalog.jsonb_typeof(v_games) is distinct from 'array'
       or pg_catalog.jsonb_array_length(v_games) not in (2, 3)
       or new.team_a_id is null
       or new.team_b_id is null
       or new.team_a_id = new.team_b_id then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_BEST_OF_THREE_GAMES_INVALID';
    end if;

    for v_entry in
      select row.value
        from pg_catalog.jsonb_array_elements(v_games) with ordinality as row(value, ordinal)
       order by row.ordinal
    loop
      if pg_catalog.jsonb_typeof(v_entry) is distinct from 'object'
         or (v_entry->>'game_number')::integer is distinct from v_expected_number
         or v_wins_a = 2
         or v_wins_b = 2 then
        raise exception using errcode = '22023',
          message = 'JUPR_TOURNAMENT_BEST_OF_THREE_SEQUENCE_INVALID';
      end if;
      v_game_number := (v_entry->>'game_number')::integer;
      v_score_a := (v_entry->>'score_a')::integer;
      v_score_b := (v_entry->>'score_b')::integer;
      v_review := v_entry->'score_review';
      if v_game_number not between 1 and 3
         or v_score_a is null
         or v_score_b is null
         or v_score_a < 0
         or v_score_b < 0
         or v_score_a = v_score_b
         or greatest(v_score_a, v_score_b) < 11
         or pg_catalog.abs(v_score_a - v_score_b) < 2
         or pg_catalog.jsonb_typeof(v_review) is distinct from 'object'
         or v_review->>'accepted' is distinct from 'true'
         or v_review->>'scoring_format' is distinct from 'GAME_TO_11'
         or (v_review->>'score_a')::integer is distinct from v_score_a
         or (v_review->>'score_b')::integer is distinct from v_score_b then
        raise exception using errcode = '22023',
          message = 'JUPR_TOURNAMENT_BEST_OF_THREE_GAME_SCORE_INVALID';
      end if;
      if v_score_a > v_score_b then
        v_wins_a := v_wins_a + 1;
      else
        v_wins_b := v_wins_b + 1;
      end if;
      v_expected_number := v_expected_number + 1;
    end loop;

    if greatest(v_wins_a, v_wins_b) <> 2
       or least(v_wins_a, v_wins_b) not in (0, 1)
       or new.score_a is distinct from v_wins_a
       or new.score_b is distinct from v_wins_b then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_BEST_OF_THREE_RESULT_INVALID';
    end if;
    v_expected_winner := case when v_wins_a > v_wins_b then new.team_a_id else new.team_b_id end;
    v_expected_loser := case when v_expected_winner = new.team_a_id then new.team_b_id else new.team_a_id end;
    if new.winner_team_id is distinct from v_expected_winner
       or new.loser_team_id is distinct from v_expected_loser then
      raise exception using errcode = '22023',
        message = 'JUPR_TOURNAMENT_BEST_OF_THREE_WINNER_INVALID';
    end if;
    new.parent_result_only := true;
  elsif tg_op = 'UPDATE'
        and old.series_parent_game_id is null
        and pg_catalog.upper(
          coalesce(old.score_review_json->>'scoring_format', old.scoring_format, '')
        ) = 'BEST_2_OF_3'
        and (
          new.finalized_at is null
          or pg_catalog.upper(coalesce(new.result_type, 'PLAYED')) <> 'PLAYED'
        ) then
    new.parent_result_only := false;
  end if;
  return new;
exception
  when invalid_text_representation or numeric_value_out_of_range then
    raise exception using errcode = '22023',
      message = 'JUPR_TOURNAMENT_BEST_OF_THREE_GAMES_INVALID';
end;
$function$;

revoke all on function public.validate_tournament_best_of_three_result()
  from public, anon, authenticated;
grant execute on function public.validate_tournament_best_of_three_result()
  to service_role;

drop trigger if exists trg_02_tournament_games_best_of_three_validate
  on public.tournament_games;
create trigger trg_02_tournament_games_best_of_three_validate
before insert or update on public.tournament_games
for each row execute function public.validate_tournament_best_of_three_result();

-- Materialize or correct the rating children in the same transaction as the
-- aggregate parent score.  Existing child identities remain stable across a
-- pre-publication correction.
create or replace function public.sync_tournament_best_of_three_rating_games()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_format text;
  v_games jsonb;
  v_entry jsonb;
  v_game_number integer;
  v_score_a integer;
  v_score_b integer;
  v_winner uuid;
  v_loser uuid;
  v_preserve_retirement_games boolean := false;
begin
  if new.series_parent_game_id is not null then
    return null;
  end if;
  v_format := pg_catalog.upper(
    coalesce(new.score_review_json->>'scoring_format', new.scoring_format, '')
  );
  v_games := new.score_review_json->'game_scores';
  v_preserve_retirement_games := (
    v_format = 'BEST_2_OF_3'
    and pg_catalog.upper(coalesce(new.result_type, '')) = 'RETIREMENT'
    and new.finalized_at is not null
    and not new.parent_result_only
    and new.score_review_json
          ->>'retirement_completed_games_preserved' = 'true'
    and case
      when pg_catalog.jsonb_typeof(v_games) = 'array'
        then pg_catalog.jsonb_array_length(v_games) in (1, 2)
      else false
    end
  );

  if (
       v_format = 'BEST_2_OF_3'
       and pg_catalog.upper(coalesce(new.result_type, 'PLAYED')) = 'PLAYED'
       and new.finalized_at is not null
       and new.parent_result_only
     ) or v_preserve_retirement_games then
    if exists (
      select 1
        from public.matches as official
       where official.tournament_game_id = new.id
          or official.tournament_game_id in (
            select child.id
              from public.tournament_games as child
             where child.series_parent_game_id = new.id
          )
    ) then
      raise exception using errcode = '40001',
        message = 'JUPR_TOURNAMENT_SCORE_PUBLISHED_LOCK';
    end if;

    for v_entry in
      select row.value
        from pg_catalog.jsonb_array_elements(v_games) with ordinality as row(value, ordinal)
       order by row.ordinal
    loop
      v_game_number := (v_entry->>'game_number')::integer;
      v_score_a := (v_entry->>'score_a')::integer;
      v_score_b := (v_entry->>'score_b')::integer;
      v_winner := case when v_score_a > v_score_b then new.team_a_id else new.team_b_id end;
      v_loser := case when v_winner = new.team_a_id then new.team_b_id else new.team_a_id end;

      insert into public.tournament_games (
        tournament_id,
        draw_id,
        registration_day_id,
        event_option_id,
        stage,
        team_a_id,
        team_b_id,
        score_a,
        score_b,
        winner_team_id,
        loser_team_id,
        finalized_at,
        scoring_format,
        result_type,
        result_note,
        result_recorded_by,
        score_review_json,
        parent_result_only,
        series_parent_game_id,
        series_game_number,
        created_at,
        updated_at
      ) values (
        new.tournament_id,
        new.draw_id,
        new.registration_day_id,
        new.event_option_id,
        'SERIES_GAME',
        new.team_a_id,
        new.team_b_id,
        v_score_a,
        v_score_b,
        v_winner,
        v_loser,
        new.finalized_at - (
          (pg_catalog.jsonb_array_length(v_games) - v_game_number)
          * interval '1 microsecond'
        ),
        'GAME_TO_11',
        'PLAYED',
        null,
        new.result_recorded_by,
        v_entry->'score_review',
        false,
        new.id,
        v_game_number,
        pg_catalog.clock_timestamp(),
        pg_catalog.clock_timestamp()
      )
      on conflict (series_parent_game_id, series_game_number)
        where series_parent_game_id is not null
      do update set
        tournament_id = excluded.tournament_id,
        draw_id = excluded.draw_id,
        registration_day_id = excluded.registration_day_id,
        event_option_id = excluded.event_option_id,
        stage = excluded.stage,
        team_a_id = excluded.team_a_id,
        team_b_id = excluded.team_b_id,
        score_a = excluded.score_a,
        score_b = excluded.score_b,
        winner_team_id = excluded.winner_team_id,
        loser_team_id = excluded.loser_team_id,
        finalized_at = excluded.finalized_at,
        scoring_format = excluded.scoring_format,
        result_type = excluded.result_type,
        result_note = excluded.result_note,
        result_recorded_by = excluded.result_recorded_by,
        score_review_json = excluded.score_review_json,
        parent_result_only = false,
        updated_at = pg_catalog.clock_timestamp();
    end loop;

    delete from public.tournament_games as child
     where child.series_parent_game_id = new.id
       and child.series_game_number > pg_catalog.jsonb_array_length(v_games);
  elsif tg_op = 'UPDATE'
        and pg_catalog.upper(
          coalesce(old.score_review_json->>'scoring_format', old.scoring_format, '')
        ) = 'BEST_2_OF_3'
        and (
          new.finalized_at is null
          or pg_catalog.upper(coalesce(new.result_type, 'PLAYED')) <> 'PLAYED'
        ) then
    delete from public.tournament_games as child
     where child.series_parent_game_id = new.id;
  end if;
  return null;
end;
$function$;

revoke all on function public.sync_tournament_best_of_three_rating_games()
  from public, anon, authenticated;
grant execute on function public.sync_tournament_best_of_three_rating_games()
  to service_role;

drop trigger if exists trg_10_tournament_games_best_of_three_sync
  on public.tournament_games;
create trigger trg_10_tournament_games_best_of_three_sync
after insert or update on public.tournament_games
for each row execute function public.sync_tournament_best_of_three_rating_games();

-- The ordinary score wrapper intentionally writes the aggregate score first
-- and its reviewed individual games second.  A deferred constraint therefore
-- validates the committed shape instead of rejecting that safe transient
-- state.  It also closes direct calls to the legacy/core score RPC and direct
-- pre-publication child edits: every finalized played BEST_2_OF_3 parent must
-- finish the transaction with exactly the rating leaves described by its
-- accepted review, and no other parent may retain rating leaves.
create or replace function public.assert_tournament_best_of_three_final_state()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_parent_ids uuid[];
  v_parent_id uuid;
  v_parent public.tournament_games%rowtype;
  v_child public.tournament_games%rowtype;
  v_stored_format text;
  v_effective_format text;
  v_review_format text;
  v_games jsonb;
  v_entry jsonb;
  v_review jsonb;
  v_game_count integer;
  v_child_count integer;
  v_expected_number integer;
  v_game_number integer;
  v_score_a integer;
  v_score_b integer;
  v_wins_a integer;
  v_wins_b integer;
  v_expected_winner uuid;
  v_expected_loser uuid;
  v_expected_finalized_at timestamptz;
  v_result_type text;
  v_is_played_series boolean;
  v_is_retirement_series boolean;
begin
  if tg_op = 'INSERT' then
    v_parent_ids := array[coalesce(new.series_parent_game_id, new.id)];
  elsif tg_op = 'DELETE' then
    v_parent_ids := array[coalesce(old.series_parent_game_id, old.id)];
  else
    v_parent_ids := array[
      coalesce(old.series_parent_game_id, old.id),
      coalesce(new.series_parent_game_id, new.id)
    ];
  end if;

  for v_parent_id in
    select distinct candidate.parent_id
      from pg_catalog.unnest(v_parent_ids) as candidate(parent_id)
     where candidate.parent_id is not null
     order by candidate.parent_id
  loop
    select parent.*
      into v_parent
      from public.tournament_games as parent
     where parent.id = v_parent_id;
    if not found then
      -- Deleting an aggregate parent cascades its rating leaves.  Once both
      -- are gone there is no surviving series state to validate.
      continue;
    end if;

    if v_parent.series_parent_game_id is not null then
      raise exception using errcode = '23514',
        message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
    end if;

    v_stored_format := pg_catalog.upper(
      coalesce(nullif(pg_catalog.btrim(v_parent.scoring_format), ''), '')
    );
    select pg_catalog.upper(
             coalesce(
               nullif(pg_catalog.btrim(v_parent.scoring_format), ''),
               nullif(pg_catalog.btrim(event.scoring_override), ''),
               nullif(pg_catalog.btrim(event.scoring_default), ''),
               ''
             )
           )
      into v_effective_format
      from (select 1) as anchor
      left join public.tournament_event_draws as draw
        on draw.id = v_parent.draw_id
       and draw.tournament_id = v_parent.tournament_id
      left join public.tournament_event_options as event
        on event.id::text = coalesce(
             nullif(v_parent.event_option_id::text, ''),
             draw.event_option_id::text
           )
       and event.tournament_id = v_parent.tournament_id::text;
    v_review_format := pg_catalog.upper(
      coalesce(
        nullif(
          pg_catalog.btrim(v_parent.score_review_json->>'scoring_format'),
          ''
        ),
        ''
      )
    );
    v_games := v_parent.score_review_json->'game_scores';
    select count(*)
      into v_child_count
      from public.tournament_games as child
     where child.series_parent_game_id = v_parent.id;

    v_result_type := pg_catalog.upper(
      coalesce(v_parent.result_type, 'PLAYED')
    );
    v_is_played_series := (
      v_result_type = 'PLAYED'
      and v_parent.finalized_at is not null
      and (
        v_effective_format = 'BEST_2_OF_3'
        or v_review_format = 'BEST_2_OF_3'
        or v_parent.parent_result_only
        or v_child_count > 0
      )
    );
    v_is_retirement_series := (
      v_result_type = 'RETIREMENT'
      and v_parent.finalized_at is not null
      and (
        v_parent.score_review_json
          ->>'retirement_completed_games_preserved' = 'true'
        or v_games is not null
        or v_child_count > 0
      )
    );

    if v_is_played_series or v_is_retirement_series then
      if v_review_format <> 'BEST_2_OF_3'
         or v_effective_format <> 'BEST_2_OF_3'
         or v_stored_format not in ('', 'BEST_2_OF_3')
         or pg_catalog.jsonb_typeof(v_games) is distinct from 'array'
         or v_parent.team_a_id is null
         or v_parent.team_b_id is null
         or v_parent.team_a_id = v_parent.team_b_id then
        raise exception using errcode = '23514',
          message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
      end if;

      v_game_count := pg_catalog.jsonb_array_length(v_games);
      if v_is_played_series and (
           not v_parent.parent_result_only
           or v_parent.score_review_json->>'accepted' is distinct from 'true'
           or v_game_count not in (2, 3)
           or (v_parent.score_review_json->>'score_a')::integer
                is distinct from v_parent.score_a
           or (v_parent.score_review_json->>'score_b')::integer
                is distinct from v_parent.score_b
         ) then
        raise exception using errcode = '23514',
          message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
      end if;
      if v_is_retirement_series and (
           v_parent.parent_result_only
           or v_parent.score_review_json
                ->>'retirement_completed_games_preserved'
                is distinct from 'true'
           or v_parent.score_review_json
                ->>'synthetic_progression_score' is distinct from 'true'
           or v_parent.score_review_json
                ->>'rating_publish_eligible' is distinct from 'false'
           or v_parent.score_review_json->>'accepted' is distinct from 'true'
           or v_game_count not in (1, 2)
           or (v_parent.score_review_json->>'score_a')::integer
                is distinct from v_parent.score_a
           or (v_parent.score_review_json->>'score_b')::integer
                is distinct from v_parent.score_b
           or v_parent.score_review_json->>'non_playing_team_id'
                is distinct from v_parent.loser_team_id::text
           or v_parent.score_a is null
           or v_parent.score_b is null
           or v_parent.score_a < 0
           or v_parent.score_b < 0
           or v_parent.score_a = v_parent.score_b
           or (
             v_parent.winner_team_id is distinct from v_parent.team_a_id
             and v_parent.winner_team_id is distinct from v_parent.team_b_id
           )
           or (
             v_parent.loser_team_id is distinct from v_parent.team_a_id
             and v_parent.loser_team_id is distinct from v_parent.team_b_id
           )
           or v_parent.winner_team_id is not distinct from v_parent.loser_team_id
           or (
             v_parent.score_a > v_parent.score_b
             and v_parent.winner_team_id is distinct from v_parent.team_a_id
           )
           or (
             v_parent.score_b > v_parent.score_a
             and v_parent.winner_team_id is distinct from v_parent.team_b_id
           )
         ) then
        raise exception using errcode = '23514',
          message = 'JUPR_TOURNAMENT_RETIREMENT_RATING_EVIDENCE_INVALID';
      end if;
      if v_child_count <> v_game_count then
        raise exception using errcode = '23514',
          message = 'JUPR_TOURNAMENT_BEST_OF_THREE_CHILD_SET_INVALID';
      end if;

      v_expected_number := 1;
      v_wins_a := 0;
      v_wins_b := 0;
      for v_entry in
        select row.value
          from pg_catalog.jsonb_array_elements(v_games) with ordinality
            as row(value, ordinal)
         order by row.ordinal
      loop
        if pg_catalog.jsonb_typeof(v_entry) is distinct from 'object'
           or (v_entry->>'game_number')::integer
                is distinct from v_expected_number
           or v_wins_a = 2
           or v_wins_b = 2 then
          raise exception using errcode = '23514',
            message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
        end if;

        v_game_number := (v_entry->>'game_number')::integer;
        v_score_a := (v_entry->>'score_a')::integer;
        v_score_b := (v_entry->>'score_b')::integer;
        v_review := v_entry->'score_review';
        if v_game_number not between 1 and 3
           or v_score_a is null
           or v_score_b is null
           or v_score_a < 0
           or v_score_b < 0
           or v_score_a = v_score_b
           or greatest(v_score_a, v_score_b) < 11
           or pg_catalog.abs(v_score_a - v_score_b) < 2
           or pg_catalog.jsonb_typeof(v_review) is distinct from 'object'
           or v_review->>'accepted' is distinct from 'true'
           or v_review->>'scoring_format' is distinct from 'GAME_TO_11'
           or (v_review->>'score_a')::integer is distinct from v_score_a
           or (v_review->>'score_b')::integer is distinct from v_score_b then
          raise exception using errcode = '23514',
            message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
        end if;

        if v_score_a > v_score_b then
          v_wins_a := v_wins_a + 1;
          v_expected_winner := v_parent.team_a_id;
          v_expected_loser := v_parent.team_b_id;
        else
          v_wins_b := v_wins_b + 1;
          v_expected_winner := v_parent.team_b_id;
          v_expected_loser := v_parent.team_a_id;
        end if;
        v_expected_finalized_at := v_parent.finalized_at - (
          (v_game_count - v_game_number) * interval '1 microsecond'
        );

        select child.*
          into v_child
          from public.tournament_games as child
         where child.series_parent_game_id = v_parent.id
           and child.series_game_number = v_game_number;
        if not found then
          raise exception using errcode = '23514',
            message = 'JUPR_TOURNAMENT_BEST_OF_THREE_CHILD_SET_INVALID';
        end if;
        if v_child.tournament_id is distinct from v_parent.tournament_id
           or v_child.draw_id is distinct from v_parent.draw_id
           or v_child.registration_day_id
                is distinct from v_parent.registration_day_id
           or v_child.event_option_id is distinct from v_parent.event_option_id
           or v_child.stage is distinct from 'SERIES_GAME'
           or v_child.team_a_id is distinct from v_parent.team_a_id
           or v_child.team_b_id is distinct from v_parent.team_b_id
           or v_child.score_a is distinct from v_score_a
           or v_child.score_b is distinct from v_score_b
           or v_child.winner_team_id is distinct from v_expected_winner
           or v_child.loser_team_id is distinct from v_expected_loser
           or v_child.finalized_at is distinct from v_expected_finalized_at
           or v_child.scoring_format is distinct from 'GAME_TO_11'
           or v_child.result_type is distinct from 'PLAYED'
           or v_child.result_note is not null
           or v_child.result_recorded_by
                is distinct from v_parent.result_recorded_by
           or v_child.score_review_json is distinct from v_review
           or v_child.parent_result_only is distinct from false
           or v_child.series_parent_game_id is distinct from v_parent.id
           or v_child.series_game_number is distinct from v_game_number then
          raise exception using errcode = '23514',
            message = 'JUPR_TOURNAMENT_BEST_OF_THREE_CHILD_SET_INVALID';
        end if;
        v_expected_number := v_expected_number + 1;
      end loop;

      if v_is_played_series then
        if greatest(v_wins_a, v_wins_b) <> 2
           or least(v_wins_a, v_wins_b) not in (0, 1)
           or v_parent.score_a is distinct from v_wins_a
           or v_parent.score_b is distinct from v_wins_b then
          raise exception using errcode = '23514',
            message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
        end if;
        v_expected_winner := case
          when v_wins_a > v_wins_b then v_parent.team_a_id
          else v_parent.team_b_id
        end;
        v_expected_loser := case
          when v_expected_winner = v_parent.team_a_id then v_parent.team_b_id
          else v_parent.team_a_id
        end;
        if v_parent.winner_team_id is distinct from v_expected_winner
           or v_parent.loser_team_id is distinct from v_expected_loser then
          raise exception using errcode = '23514',
            message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
        end if;
      elsif greatest(v_wins_a, v_wins_b) >= 2 then
        raise exception using errcode = '23514',
          message = 'JUPR_TOURNAMENT_RETIREMENT_RATING_EVIDENCE_INVALID';
      end if;
    elsif v_parent.parent_result_only or v_child_count <> 0 then
      raise exception using errcode = '23514',
        message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
    end if;
  end loop;
  return null;
exception
  when invalid_text_representation or numeric_value_out_of_range then
    raise exception using errcode = '23514',
      message = 'JUPR_TOURNAMENT_BEST_OF_THREE_FINAL_STATE_INVALID';
end;
$function$;

revoke all on function public.assert_tournament_best_of_three_final_state()
  from public, anon, authenticated;
grant execute on function public.assert_tournament_best_of_three_final_state()
  to service_role;

drop trigger if exists trg_90_tournament_games_best_of_three_final_state
  on public.tournament_games;
create constraint trigger trg_90_tournament_games_best_of_three_final_state
after insert or update or delete on public.tournament_games
deferrable initially deferred
for each row
execute function public.assert_tournament_best_of_three_final_state();

-- Official tournament matches are projections of tournament_games. Generic
-- Match Log edits/exclusions must not be able to change their source facts or
-- active state independently of standings and bracket authority. Rating replay
-- is intentionally still allowed to refresh only its snapshot/rating columns.
create or replace function public.protect_official_tournament_match_source()
returns trigger
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation_key text;
  v_authorized boolean := false;
begin
  if tg_op = 'INSERT' then
    if new.tournament_game_id is null then
      return new;
    end if;
    v_operation_key := nullif(
      pg_catalog.btrim(
        pg_catalog.current_setting(
          'jupr.official_publish_operation_key',
          true
        )
      ),
      ''
    );
    if v_operation_key is not null then
      select exists (
        select 1
          from public.tournament_admin_operations as operation
          join public.tournament_games as game
            on game.id = new.tournament_game_id
           and game.tournament_id = new.tournament_id
         where operation.club_id = new.club_id
           and operation.operation_key = v_operation_key
           and operation.entity_type = 'tournament_event_draw'
           and operation.entity_id = game.draw_id::text
           and operation.action in (
             'ops_official_publish',
             'tournament_live_official_publish'
           )
           and operation.status = 'intent'
           and new.context_type = 'tournament_game'
           and new.context_id = new.tournament_game_id::text
           and exists (
             select 1
               from pg_catalog.jsonb_array_elements_text(
                 coalesce(
                   operation.request_json
                     #> '{payload,publish_plan,tournament_game_ids}',
                   '[]'::jsonb
                 )
               ) as planned(game_id)
              where planned.game_id = new.tournament_game_id::text
           )
      ) into v_authorized;
    end if;
    if not v_authorized then
      raise exception using errcode = 'P0001',
        message = 'JUPR_TOURNAMENT_OFFICIAL_MATCH_ASSOCIATION_REQUIRES_PUBLISH';
    end if;
    return new;
  end if;

  if tg_op = 'DELETE' then
    if old.tournament_game_id is null then
      return old;
    end if;
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_OFFICIAL_MATCH_HARD_DELETE_FORBIDDEN';
  end if;

  if old.tournament_game_id is null and new.tournament_game_id is null then
    return new;
  end if;

  if old.tournament_game_id is null and new.tournament_game_id is not null then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_OFFICIAL_MATCH_ASSOCIATION_REQUIRES_PUBLISH';
  end if;

  if new.id is distinct from old.id
     or new.club_id is distinct from old.club_id
     or new.tournament_id is distinct from old.tournament_id
     or new.tournament_game_id is distinct from old.tournament_game_id
     or new.context_type is distinct from old.context_type
     or new.context_id is distinct from old.context_id
     or new.date is distinct from old.date
     or new.league is distinct from old.league
     or new.week_tag is distinct from old.week_tag
     or new.match_type is distinct from old.match_type
     or new.match_format is distinct from old.match_format
     or new.rating_scope is distinct from old.rating_scope
     or new.t1_p1 is distinct from old.t1_p1
     or new.t1_p2 is distinct from old.t1_p2
     or new.t2_p1 is distinct from old.t2_p1
     or new.t2_p2 is distinct from old.t2_p2
     or new.score_t1 is distinct from old.score_t1
     or new.score_t2 is distinct from old.score_t2
     or new.rating_bonus_elo is distinct from old.rating_bonus_elo
     or new.rating_bonus_reason is distinct from old.rating_bonus_reason
     or new.notes is distinct from old.notes
     or new.deleted_at is distinct from old.deleted_at
     or new.deleted_by is distinct from old.deleted_by
     or new.deleted_source is distinct from old.deleted_source
     or new.delete_note is distinct from old.delete_note then
    raise exception using errcode = 'P0001',
      message = 'JUPR_TOURNAMENT_OFFICIAL_MATCH_SOURCE_IMMUTABLE';
  end if;
  return new;
end;
$function$;

revoke all on function public.protect_official_tournament_match_source()
  from public, anon, authenticated;
grant execute on function public.protect_official_tournament_match_source()
  to service_role;

drop trigger if exists trg_03_matches_protect_official_tournament_source
  on public.matches;
create trigger trg_03_matches_protect_official_tournament_source
before insert or update or delete on public.matches
for each row execute function public.protect_official_tournament_match_source();

-- Day activation continues to validate and queue only scheduled matchups. Rating
-- children carry the same scope and roster evidence, but their SERIES_GAME stage
-- is intentionally not an operator-facing day-run stage.
create or replace function public.assert_tournament_day_live_draw_ready(
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_draw_id uuid,
  p_expected_draw_updated_at timestamptz
)
returns void
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_day public.tournament_registration_days%rowtype;
  v_draw public.tournament_event_draws%rowtype;
  v_event public.tournament_event_options%rowtype;
  v_event_option_id text;
  v_player_id integer;
  v_registration_count integer;
begin
  -- Read only the setup identity first.  The final draw row is deliberately
  -- locked after its team/game children because their version triggers touch
  -- the draw and use that same child -> parent order.
  select draw.event_option_id
    into v_event_option_id
    from public.tournament_event_draws as draw
    join public.tournaments as tournament
      on tournament.id = draw.tournament_id
     and tournament.club_id::text = p_club_id
   where draw.id = p_draw_id
     and draw.tournament_id::text = p_tournament_id;
  if v_event_option_id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: reviewed draw identity is no longer available.';
  end if;

  select day.* into v_day
    from public.tournament_registration_days as day
   where day.id = p_registration_day_id
     and day.tournament_id::text = p_tournament_id
     and day.enabled is true
   for share;
  if v_day.id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: selected tournament day changed or is disabled.';
  end if;

  select event.* into v_event
    from public.tournament_event_options as event
   where event.id = v_event_option_id
     and event.tournament_id::text = p_tournament_id
   for share;
  if v_event.id is null then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_SCOPE: reviewed event identity changed.';
  end if;

  perform team.id
    from public.tournament_teams as team
   where team.tournament_id::text = p_tournament_id
     and team.draw_id = p_draw_id
   order by team.id
   for share;
  perform game.id
    from public.tournament_games as game
   where game.tournament_id::text = p_tournament_id
     and game.draw_id = p_draw_id
   order by game.id
   for share;

  select draw.*
    into v_draw
    from public.tournament_event_draws as draw
    join public.tournaments as tournament
      on tournament.id = draw.tournament_id
     and tournament.club_id::text = p_club_id
   where draw.id = p_draw_id
     and draw.tournament_id::text = p_tournament_id
   for share;

  if v_draw.id is null
     or v_draw.updated_at is distinct from p_expected_draw_updated_at
     or v_draw.event_option_id is distinct from v_event.id
     or pg_catalog.upper(coalesce(v_draw.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or coalesce(v_draw.hidden_from_primary_ops, false) is true
     or pg_catalog.upper(coalesce(v_draw.draw_kind, 'STANDARD')) <> 'STANDARD' then
    raise exception using
      errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_STALE: draw changed, is inactive, or is not supported by the day runner.';
  end if;

  if v_event.enabled is not true
     or pg_catalog.upper(coalesce(v_event.status, 'DRAFT')) in
       ('CANCELLED', 'CANCELED', 'ARCHIVED', 'DISABLED')
     or (case
       when v_draw.registration_day_id is not null
         then v_draw.registration_day_id = p_registration_day_id
          and case
            when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
              then case
                when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) > 0
                  then v_event.scheduled_day_ids ? v_draw.registration_day_id
                else v_event.registration_day_id is null
                  or v_event.registration_day_id = v_draw.registration_day_id
              end
            else false
          end
       else case
         when pg_catalog.jsonb_typeof(v_event.scheduled_day_ids) = 'array'
           then case
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 1
               then v_event.scheduled_day_ids ? p_registration_day_id
             when pg_catalog.jsonb_array_length(v_event.scheduled_day_ids) = 0
               then v_event.registration_day_id = p_registration_day_id
             else false
           end
         else false
       end
     end) is not true then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_DRAW_UNSCHEDULED: draw is not scheduled on this enabled tournament day.';
  end if;

  if not exists (
    select 1 from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAMES_REQUIRED: generate games before activating this draw.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         game.registration_day_id is distinct from p_registration_day_id
         or game.event_option_id is distinct from v_draw.event_option_id
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_DAY: every draw game must belong to the selected tournament day and event.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and game.series_parent_game_id is null
       and pg_catalog.upper(coalesce(game.stage, '')) not in ('ROUND_ROBIN', 'PLAYOFF')
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_STAGE: every draw game must use a supported ROUND_ROBIN or PLAYOFF stage.';
  end if;

  if (
    select pg_catalog.count(*)
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and (
         team.registration_day_id is distinct from p_registration_day_id
         or team.event_option_id is distinct from v_draw.event_option_id
       )
  ) > 0 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_TEAM_DAY_SCOPE: every draw team must belong to the selected tournament day and draw event.';
  end if;

  if exists (
    select participant.player_id
      from public.tournament_teams as team
      cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
       and participant.player_id is not null
     group by participant.player_id
    having pg_catalog.count(*) > 1
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROSTER_PLAYER_DUPLICATE: each player may belong to only one exact team in this draw.';
  end if;

  if (
    select coalesce(
             pg_catalog.array_agg(team.id order by team.id),
             '{}'::uuid[]
           )
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
  ) is distinct from (
    select coalesce(
             pg_catalog.array_agg(distinct side.team_id order by side.team_id)
               filter (where side.team_id is not null),
             '{}'::uuid[]
           )
      from public.tournament_games as game
      cross join lateral (values (game.team_a_id), (game.team_b_id)) as side(team_id)
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'ROUND_ROBIN'
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROUND_ROBIN_ROSTER: every exact draw team must appear in the reviewed round-robin schedule.';
  end if;

  if (
    select pg_catalog.count(*)
      from public.tournament_teams as team
     where team.tournament_id = v_draw.tournament_id
       and team.draw_id = v_draw.id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
  ) < 4 then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFF_FORMAT: activate at least four exact in-draw teams so a supported playoff format can complete day closeout.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      left join public.tournament_teams as team_a
        on team_a.id = game.team_a_id
       and team_a.tournament_id = game.tournament_id
       and team_a.draw_id = game.draw_id
       and team_a.registration_day_id = p_registration_day_id
       and team_a.event_option_id = v_draw.event_option_id
      left join public.tournament_teams as team_b
        on team_b.id = game.team_b_id
       and team_b.tournament_id = game.tournament_id
       and team_b.draw_id = game.draw_id
       and team_b.registration_day_id = p_registration_day_id
       and team_b.event_option_id = v_draw.event_option_id
      cross join lateral (
        select
          pg_catalog.count(participant.player_id) as player_count,
          pg_catalog.count(distinct participant.player_id) as distinct_player_count
        from (values
          (team_a.player1_id), (team_a.player2_id),
          (team_b.player1_id), (team_b.player2_id)
        ) as participant(player_id)
        where participant.player_id is not null
      ) as participant_counts
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'ROUND_ROBIN'
       and (
         game.team_a_id is null
         or game.team_b_id is null
         or game.team_a_id = game.team_b_id
         or team_a.id is null
         or team_b.id is null
         or team_a.player1_id is null
         or team_b.player1_id is null
         or participant_counts.player_count not in (2, 4)
         or participant_counts.player_count <> participant_counts.distinct_player_count
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_ROUND_ROBIN: every round-robin game requires two distinct in-draw teams with two or four distinct effective players.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and pg_catalog.upper(coalesce(game.stage, '')) = 'PLAYOFF'
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYOFFS_ALREADY_GENERATED: activate reviewed round-robin games first, then generate playoffs through the guarded day operation.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         (game.finalized_at is null and (
           game.score_a is not null or game.score_b is not null
           or game.winner_team_id is not null or game.loser_team_id is not null
         ))
         or
         (game.finalized_at is not null and (
           game.score_a is null or game.score_b is null
           or game.score_a = game.score_b
           or game.winner_team_id is null or game.loser_team_id is null
           or game.team_a_id is null or game.team_b_id is null
           or game.team_a_id = game.team_b_id
           or game.winner_team_id = game.loser_team_id
           or game.winner_team_id not in (game.team_a_id, game.team_b_id)
           or game.loser_team_id not in (game.team_a_id, game.team_b_id)
           or (game.score_a > game.score_b and game.winner_team_id <> game.team_a_id)
           or (game.score_b > game.score_a and game.winner_team_id <> game.team_b_id)
         ))
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_STATE: partially scored or tied games require reconciliation before activation.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      cross join lateral (
        select
          pg_catalog.count(participant.player_id) as player_count,
          pg_catalog.count(distinct participant.player_id) as distinct_player_count
        from public.tournament_teams as team
        cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
        where team.id in (game.team_a_id, game.team_b_id)
          and team.tournament_id = game.tournament_id
          and team.draw_id = game.draw_id
          and team.registration_day_id = p_registration_day_id
          and team.event_option_id = v_draw.event_option_id
          and participant.player_id is not null
      ) as participant_counts
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and game.team_a_id is not null
       and game.team_b_id is not null
       and (
         participant_counts.player_count not in (2, 4)
         or participant_counts.player_count <> participant_counts.distinct_player_count
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PARTICIPANTS: every playable game needs two or four distinct effective players.';
  end if;

  if exists (
    select 1
      from public.tournament_games as game
      left join public.tournament_teams as team_a
        on team_a.id = game.team_a_id
       and team_a.tournament_id = game.tournament_id
       and team_a.draw_id = game.draw_id
       and team_a.registration_day_id = p_registration_day_id
       and team_a.event_option_id = v_draw.event_option_id
      left join public.tournament_teams as team_b
        on team_b.id = game.team_b_id
       and team_b.tournament_id = game.tournament_id
       and team_b.draw_id = game.draw_id
       and team_b.registration_day_id = p_registration_day_id
       and team_b.event_option_id = v_draw.event_option_id
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and (
         (game.team_a_id is not null and (
           team_a.id is null or team_a.player1_id is null
         ))
         or (game.team_b_id is not null and (
           team_b.id is null or team_b.player1_id is null
         ))
         or (
           game.team_a_id is not null and game.team_b_id is not null
           and team_a.id = team_b.id
         )
       )
  ) then
    raise exception using
      errcode = '22023',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_PARTICIPANTS: a playable game has incomplete or foreign team evidence.';
  end if;

  for v_player_id in
    select distinct participant.player_id
      from public.tournament_games as game
      join public.tournament_teams as team
        on team.id in (game.team_a_id, game.team_b_id)
       and team.tournament_id = game.tournament_id
       and team.draw_id = game.draw_id
       and team.registration_day_id = p_registration_day_id
       and team.event_option_id = v_draw.event_option_id
      cross join lateral (values (team.player1_id), (team.player2_id)) as participant(player_id)
     where game.tournament_id = v_draw.tournament_id
       and game.draw_id = v_draw.id
       and participant.player_id is not null
     order by participant.player_id
  loop
    perform player.id
      from public.players as player
     where player.id = v_player_id
       and player.club_id::text = p_club_id
     for share;
    if not found then
      raise exception using
        errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_PLAYER_SCOPE: every participant must belong to the tournament club.';
    end if;
    perform registration.id
      from public.tournament_registrations as registration
     where registration.tournament_id::text = p_tournament_id
       and registration.player_id = v_player_id
       and pg_catalog.upper(coalesce(registration.status, '')) in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED')
     order by registration.id
     for share;
    select pg_catalog.count(*)
      into v_registration_count
      from public.tournament_registrations as registration
     where registration.tournament_id::text = p_tournament_id
       and registration.player_id = v_player_id
       and pg_catalog.upper(coalesce(registration.status, '')) in
         ('ACTIVE', 'APPROVED', 'CONFIRMED', 'REGISTERED');
    if v_registration_count <> 1 then
      raise exception using
        errcode = '22023',
        message = 'JUPR_TOURNAMENT_DAY_LIVE_REGISTRATION: every player must resolve to exactly one active tournament registration.';
    end if;
    -- The exact draw roster plus canonical active registration is the
    -- live-play authority. Check-in, waiver, and payment remain visible in
    -- preflight but never block draw activation or court assignment.
  end loop;
end;
$function$;


-- Queue seeding counts and inserts aggregate matchups only. This keeps a
-- recovered/re-created day run from materializing rating children as courts.
create or replace function public.seed_tournament_day_live_draw(
  p_run_id uuid,
  p_club_id text,
  p_tournament_id text,
  p_registration_day_id text,
  p_draw_id uuid,
  p_expected_draw_updated_at timestamptz,
  p_priority integer,
  p_operation_key text,
  p_actor text
)
returns uuid
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_day_draw_id uuid;
  v_priority_base bigint;
  v_source_game_count integer;
  v_inserted_game_count integer;
begin
  perform public.assert_tournament_day_live_draw_ready(
    p_club_id,
    p_tournament_id,
    p_registration_day_id,
    p_draw_id,
    p_expected_draw_updated_at
  );

  insert into public.tournament_day_live_draws (
    run_id, tournament_id, registration_day_id, draw_id, state, priority,
    source_draw_updated_at, version, last_operation_key,
    activated_by, updated_by
  ) values (
    p_run_id, p_tournament_id::uuid, p_registration_day_id, p_draw_id,
    'ACTIVE', p_priority, p_expected_draw_updated_at, 1, p_operation_key,
    p_actor, p_actor
  )
  returning id into v_day_draw_id;

  select coalesce(pg_catalog.max(queue.priority), 0)
    into v_priority_base
    from public.tournament_day_live_queue as queue
   where queue.run_id = p_run_id;

  select pg_catalog.count(*)
    into v_source_game_count
    from public.tournament_games as game
   where game.tournament_id::text = p_tournament_id
     and game.draw_id = p_draw_id
     and game.series_parent_game_id is null;

  insert into public.tournament_day_live_queue (
    run_id, tournament_id, registration_day_id, day_draw_id, draw_id, game_id,
    team_a_id, team_b_id, state, priority, court_id,
    blocker_code, blocker_detail, version, last_operation_key,
    eligible_since, released_at, completed_at, updated_by
  )
  select
    p_run_id,
    game.tournament_id,
    p_registration_day_id,
    v_day_draw_id,
    game.draw_id,
    game.id,
    game.team_a_id,
    game.team_b_id,
    case
      when game.finalized_at is not null then 'COMPLETED'
      when game.team_a_id is not null and game.team_b_id is not null then 'WAITING'
      else 'BLOCKED'
    end,
    v_priority_base + pg_catalog.row_number() over (
      order by
        case when game.stage = 'ROUND_ROBIN' then 0 else 1 end,
        game.rr_round_number nulls last,
        game.rr_slot_number nulls last,
        game.playoff_round nulls last,
        game.playoff_game_code nulls last,
        game.id
    ),
    null,
    case
      when game.finalized_at is not null then null
      when game.team_a_id is null or game.team_b_id is null then 'DEPENDENCY_PENDING'
      else null
    end,
    case
      when game.team_a_id is null or game.team_b_id is null
        then 'Playoff source teams are not resolved yet.'
      else null
    end,
    1,
    p_operation_key,
    case
      when game.finalized_at is null and game.team_a_id is not null and game.team_b_id is not null
        then pg_catalog.clock_timestamp()
      else null
    end,
    case when game.finalized_at is not null then game.finalized_at else null end,
    game.finalized_at,
    p_actor
  from public.tournament_games as game
  where game.tournament_id::text = p_tournament_id
    and game.draw_id = p_draw_id
    and game.series_parent_game_id is null
    and game.registration_day_id = p_registration_day_id
    and game.event_option_id is not distinct from (
      select draw.event_option_id
        from public.tournament_event_draws as draw
       where draw.id = p_draw_id
    )
  order by
    case when game.stage = 'ROUND_ROBIN' then 0 else 1 end,
    game.rr_round_number nulls last,
    game.rr_slot_number nulls last,
    game.playoff_round nulls last,
    game.playoff_game_code nulls last,
    game.id
  on conflict (run_id, game_id) do nothing;
  get diagnostics v_inserted_game_count = row_count;
  if v_inserted_game_count <> v_source_game_count then
    raise exception using errcode = '40001',
      message = 'JUPR_TOURNAMENT_DAY_LIVE_GAME_DAY: exact draw game materialization failed.';
  end if;

  return v_day_draw_id;
end;
$function$;


-- Round-robin recovery compares and repairs the aggregate schedule. Series
-- rating children are immutable derived evidence, not missing schedule pairs.
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
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null;
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
         and game.series_parent_game_id is null
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
         and game.series_parent_game_id is null
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
         and game.series_parent_game_id is null
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
         and game.series_parent_game_id is null
         union all
         select game.rr_round_number, game.team_b_id::text
           from public.tournament_games as game
          where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null
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
         and game.series_parent_game_id is null
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
   where game.tournament_id = v_draw.tournament_id and game.draw_id = v_draw.id
         and game.series_parent_game_id is null;
  return pg_catalog.jsonb_build_object('ok', true, 'games', v_saved);
exception
  when raise_exception then
    if sqlerrm = 'JUPR_TOURNAMENT_DRAW_RECOVERY_DEPENDENCY_BLOCKED' then
      raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_ROUND_ROBIN_RECONCILE_BLOCKED';
    end if;
    raise;
end
$function$;

notify pgrst, 'reload schema';
