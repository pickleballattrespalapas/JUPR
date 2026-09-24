-- Team League finalize/reconcile used a column absent from the canonical
-- matches table. Match Log exclusions use deleted_at. Preserve that authority
-- and all roster, score reservation, idempotency and standings checks.
-- Apply after the normalized roster migration in the deployment contract.

create or replace function public.team_league_reconcile_fixture_v2(
  p_operation_id uuid,
  p_club_id text,
  p_fixture_id uuid,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_operation public.team_league_operations%rowtype;
  v_before public.team_league_fixtures%rowtype;
  v_updated public.team_league_fixtures%rowtype;
  v_match public.matches%rowtype;
  v_match_found boolean := false;
  v_normal_sides boolean := false;
  v_swapped_sides boolean := false;
  v_new_winner uuid;
  v_dependency_changed boolean := false;
  v_resolved_count integer := 0;
  v_actor_email text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_email)),
        320
      ),
      ''
    ),
    'unknown'
  );
  v_actor_role text := coalesce(
    nullif(
      pg_catalog.left(
        pg_catalog.lower(pg_catalog.btrim(p_actor_role)),
        80
      ),
      ''
    ),
    'admin'
  );
  v_source text := coalesce(
    nullif(pg_catalog.left(pg_catalog.btrim(p_source), 160), ''),
    'next_team_league_reconcile'
  );
  v_result jsonb;
begin
  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
   for update;
  if not found or v_operation.operation_type <> 'admin_reconcile_fixture' then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_RECONCILE_OPERATION_NOT_FOUND';
  end if;
  if v_operation.status = 'complete'
     and v_operation.result_json is not null then
    return v_operation.result_json || '{"idempotent": true}'::jsonb;
  end if;

  perform 1
    from public.team_league_settings as settings
   where settings.club_id = v_operation.club_id
     and settings.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;

  select fixture.*
    into v_before
    from public.team_league_fixtures as fixture
   where fixture.id = p_fixture_id
     and fixture.club_id = v_operation.club_id
     and fixture.league_name = v_operation.league_name
   for update;
  if not found or v_before.official_match_id is null then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_CANONICAL_MATCH_NOT_LINKED';
  end if;
  if v_before.phase = 'regular' and exists (
    select 1
      from public.team_league_fixtures as playoff
     where playoff.club_id = v_before.club_id
       and playoff.league_name = v_before.league_name
       and playoff.phase = 'playoff'
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_REGULAR_RESULT_LOCKED_AFTER_PLAYOFF_SEEDING';
  end if;

  select match_row.*
    into v_match
    from public.matches as match_row
   where match_row.club_id = v_before.club_id
     and match_row.id = v_before.official_match_id
   for share;
  v_match_found := found;
  if v_match_found then
    if v_match.league is distinct from v_before.league_name
       or v_match.match_type is distinct from 'Team League' then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_CANONICAL_MATCH_CONTEXT_INVALID';
    end if;
    if v_before.team_a_player_1_id is null
       or v_before.team_a_player_2_id is null
       or v_before.team_b_player_1_id is null
       or v_before.team_b_player_2_id is null then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_FIXTURE_PLAYER_EVIDENCE_MISSING';
    end if;
    v_normal_sides := (
      array[v_match.t1_p1, v_match.t1_p2]
        @> array[
          v_before.team_a_player_1_id,
          v_before.team_a_player_2_id
        ]
      and array[v_match.t1_p1, v_match.t1_p2]
        <@ array[
          v_before.team_a_player_1_id,
          v_before.team_a_player_2_id
        ]
      and array[v_match.t2_p1, v_match.t2_p2]
        @> array[
          v_before.team_b_player_1_id,
          v_before.team_b_player_2_id
        ]
      and array[v_match.t2_p1, v_match.t2_p2]
        <@ array[
          v_before.team_b_player_1_id,
          v_before.team_b_player_2_id
        ]
    );
    v_swapped_sides := (
      array[v_match.t1_p1, v_match.t1_p2]
        @> array[
          v_before.team_b_player_1_id,
          v_before.team_b_player_2_id
        ]
      and array[v_match.t1_p1, v_match.t1_p2]
        <@ array[
          v_before.team_b_player_1_id,
          v_before.team_b_player_2_id
        ]
      and array[v_match.t2_p1, v_match.t2_p2]
        @> array[
          v_before.team_a_player_1_id,
          v_before.team_a_player_2_id
        ]
      and array[v_match.t2_p1, v_match.t2_p2]
        <@ array[
          v_before.team_a_player_1_id,
          v_before.team_a_player_2_id
        ]
    );
    if not v_normal_sides and not v_swapped_sides then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_CANONICAL_MATCH_PLAYER_SET_INVALID';
    end if;
  end if;

  if not v_match_found
     or v_match.deleted_at is not null then
    v_new_winner := null;
  else
    if v_match.score_t1 is null
       or v_match.score_t2 is null
       or v_match.score_t1 = v_match.score_t2 then
      raise exception using
        errcode = '22023',
        message = 'TEAM_LEAGUE_CANONICAL_MATCH_SCORE_INVALID';
    end if;
    v_new_winner := case
      when v_normal_sides and v_match.score_t1 > v_match.score_t2
        then v_before.team_a_id
      when v_normal_sides
        then v_before.team_b_id
      when v_match.score_t1 > v_match.score_t2
        then v_before.team_b_id
      else v_before.team_a_id
    end;
  end if;
  v_dependency_changed :=
    v_before.phase = 'playoff'
    and v_before.winner_team_id is distinct from v_new_winner;

  if v_dependency_changed and exists (
    with recursive affected as (
      select
        source.round_number,
        source.bracket_slot,
        0 as depth
      from public.team_league_fixtures as source
      where source.id = p_fixture_id
      union all
      select
        target.round_number,
        target.bracket_slot,
        affected.depth + 1
      from affected
      join public.team_league_fixtures as target
        on target.club_id = v_before.club_id
       and target.league_name = v_before.league_name
       and target.phase = 'playoff'
       and (
         target.team_a_source =
           'winner:' || affected.round_number::text || ':' ||
           affected.bracket_slot::text
         or target.team_b_source =
           'winner:' || affected.round_number::text || ':' ||
           affected.bracket_slot::text
       )
    )
    select 1
      from affected
      join public.team_league_fixtures as fixture
        on fixture.club_id = v_before.club_id
       and fixture.league_name = v_before.league_name
       and fixture.phase = 'playoff'
       and fixture.round_number = affected.round_number
       and fixture.bracket_slot = affected.bracket_slot
     where affected.depth > 0
       and (
         fixture.status in ('complete', 'forfeit')
         or fixture.official_match_id is not null
         or fixture.score_operation_id is not null
       )
  ) then
    raise exception using
      errcode = '55000',
      message = 'TEAM_LEAGUE_PLAYOFF_DEPENDENT_RESULT_LOCKED';
  end if;

  if v_dependency_changed then
    with recursive affected as (
      select
        source.round_number,
        source.bracket_slot,
        0 as depth
      from public.team_league_fixtures as source
      where source.id = p_fixture_id
      union all
      select
        target.round_number,
        target.bracket_slot,
        affected.depth + 1
      from affected
      join public.team_league_fixtures as target
        on target.club_id = v_before.club_id
       and target.league_name = v_before.league_name
       and target.phase = 'playoff'
       and (
         target.team_a_source =
           'winner:' || affected.round_number::text || ':' ||
           affected.bracket_slot::text
         or target.team_b_source =
           'winner:' || affected.round_number::text || ':' ||
           affected.bracket_slot::text
       )
    ),
    descendant as (
      select distinct round_number, bracket_slot
        from affected
       where depth > 0
    ),
    source_key as (
      select distinct
        'winner:' || round_number::text || ':' ||
        bracket_slot::text as value
      from affected
    )
    update public.team_league_fixtures as fixture
       set team_a_id = case
             when fixture.team_a_source in (
               select value from source_key
             ) then null
             else fixture.team_a_id
           end,
           team_b_id = case
             when fixture.team_b_source in (
               select value from source_key
             ) then null
             else fixture.team_b_id
           end,
           status = 'scheduled',
           resolution = null,
           team_a_score = null,
           team_b_score = null,
           winner_team_id = null,
           official_match_id = null,
           team_a_player_1_id = null,
           team_a_player_2_id = null,
           team_b_player_1_id = null,
           team_b_player_2_id = null,
           substitutions_json = '[]'::jsonb,
           score_note = null,
           score_operation_id = null,
           score_reserved_at = null,
           scored_by = null,
           scored_at = null,
           updated_at = pg_catalog.clock_timestamp()
      from descendant
     where fixture.club_id = v_before.club_id
       and fixture.league_name = v_before.league_name
       and fixture.phase = 'playoff'
       and fixture.round_number = descendant.round_number
       and fixture.bracket_slot = descendant.bracket_slot;
  end if;

  if not v_match_found
     or v_match.deleted_at is not null then
    update public.team_league_fixtures
       set status = 'cancelled',
           resolution = 'cancelled',
           winner_team_id = null,
           team_a_score = null,
           team_b_score = null,
           score_note =
             'Canonical match was excluded or removed through Match Log.',
           updated_at = pg_catalog.clock_timestamp()
     where id = p_fixture_id
    returning * into v_updated;
  else
    update public.team_league_fixtures
       set status = 'complete',
           resolution = 'played',
           team_a_score = case
             when v_normal_sides then v_match.score_t1
             else v_match.score_t2
           end,
           team_b_score = case
             when v_normal_sides then v_match.score_t2
             else v_match.score_t1
           end,
           winner_team_id = v_new_winner,
           score_note = null,
           updated_at = pg_catalog.clock_timestamp()
     where id = p_fixture_id
    returning * into v_updated;
  end if;

  if v_dependency_changed and v_updated.winner_team_id is not null then
    loop
      update public.team_league_fixtures as target
         set team_a_id = case
               when target.team_a_id is null
                and target.team_a_source =
                  'winner:' || source.round_number::text || ':' ||
                  source.bracket_slot::text
                 then source.winner_team_id
               else target.team_a_id
             end,
             team_b_id = case
               when target.team_b_id is null
                and target.team_b_source =
                  'winner:' || source.round_number::text || ':' ||
                  source.bracket_slot::text
                 then source.winner_team_id
               else target.team_b_id
             end,
             updated_at = pg_catalog.clock_timestamp()
        from public.team_league_fixtures as source
       where target.club_id = v_before.club_id
         and target.league_name = v_before.league_name
         and target.phase = 'playoff'
         and source.club_id = target.club_id
         and source.league_name = target.league_name
         and source.phase = 'playoff'
         and source.winner_team_id is not null
         and (
           (
             target.team_a_id is null
             and target.team_a_source =
               'winner:' || source.round_number::text || ':' ||
               source.bracket_slot::text
           )
           or (
             target.team_b_id is null
             and target.team_b_source =
               'winner:' || source.round_number::text || ':' ||
               source.bracket_slot::text
           )
         );
      get diagnostics v_resolved_count = row_count;
      exit when v_resolved_count = 0;
    end loop;
  end if;

  update public.team_league_settings
     set standings_version = standings_version + 1,
         updated_at = pg_catalog.clock_timestamp()
   where club_id = v_before.club_id
     and league_name = v_before.league_name;

  v_result := pg_catalog.jsonb_build_object(
    'ok', true,
    'committed', true,
    'operation_id', p_operation_id,
    'fixture_id', p_fixture_id,
    'status', v_updated.status,
    'official_match_id', v_updated.official_match_id,
    'sides_swapped', v_swapped_sides,
    'downstream_invalidated', v_dependency_changed,
    'message', 'Fixture refreshed from the canonical match.',
    'idempotent', false
  );
  insert into public.admin_activity_log (
    club_id,
    actor_email,
    actor_role,
    action_type,
    entity_type,
    entity_id,
    before_json,
    after_json,
    note,
    source_page,
    flagged_for_review
  ) values (
    v_updated.club_id,
    v_actor_email,
    v_actor_role,
    'team_league_fixture_reconciled',
    'team_league_fixture',
    p_fixture_id::text,
    pg_catalog.to_jsonb(v_before),
    pg_catalog.to_jsonb(v_updated),
    case
      when v_dependency_changed
        then 'Unplayed dependent playoff slots were rebuilt.'
      else null
    end,
    v_source,
    false
  );
  update public.team_league_operations
     set status = 'complete',
         result_json = v_result,
         recovery_note = null,
         completed_at = pg_catalog.clock_timestamp(),
         updated_at = pg_catalog.clock_timestamp()
   where id = p_operation_id;
  return v_result;
end
$function$;

create or replace function public.team_league_finalize_fixture_v2(
  p_operation_id uuid,
  p_club_id text,
  p_fixture_id uuid,
  p_status text,
  p_team_a_score integer,
  p_team_b_score integer,
  p_winner_team_id uuid,
  p_official_match_id bigint,
  p_team_a_player_1_id bigint,
  p_team_a_player_2_id bigint,
  p_team_b_player_1_id bigint,
  p_team_b_player_2_id bigint,
  p_substitutions jsonb,
  p_score_note text,
  p_actor_email text,
  p_actor_role text,
  p_source text
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_status text := pg_catalog.lower(pg_catalog.btrim(coalesce(p_status, '')));
  v_operation public.team_league_operations%rowtype;
  v_fixture public.team_league_fixtures%rowtype;
  v_settings public.team_league_settings%rowtype;
  v_pool_substitution_count integer;
  v_expected_substitutions jsonb := '[]'::jsonb;
  v_result jsonb;
begin
  if v_status not in ('complete', 'forfeit')
     or p_substitutions is null
     or pg_catalog.jsonb_typeof(p_substitutions) <> 'array' then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_FIXTURE_RESULT_INVALID';
  end if;
  select operation.*
    into v_operation
    from public.team_league_operations as operation
   where operation.id = p_operation_id
     and operation.club_id = pg_catalog.btrim(p_club_id)
   for update;
  if not found or v_operation.operation_type <> 'admin_score_fixture' then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SCORE_OPERATION_NOT_FOUND';
  end if;
  if nullif(v_operation.request_json ->> 'fixture_id', '') is distinct from
     p_fixture_id::text then
    raise exception using
      errcode = '22023',
      message = 'TEAM_LEAGUE_SCORE_OPERATION_FIXTURE_MISMATCH';
  end if;
  if v_operation.status = 'complete'
     and v_operation.result_json is not null then
    return v_operation.result_json || '{"idempotent": true}'::jsonb;
  end if;

  select settings.*
    into v_settings
    from public.team_league_settings as settings
   where settings.club_id = v_operation.club_id
     and settings.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_SETTINGS_NOT_FOUND';
  end if;
  select fixture.*
    into v_fixture
    from public.team_league_fixtures as fixture
   where fixture.id = p_fixture_id
     and fixture.club_id = v_operation.club_id
     and fixture.league_name = v_operation.league_name
   for update;
  if not found then
    raise exception using
      errcode = 'P0002',
      message = 'TEAM_LEAGUE_FIXTURE_NOT_FOUND';
  end if;
  if v_fixture.score_operation_id = p_operation_id
     and v_fixture.status in ('complete', 'forfeit') then
    null;
  elsif v_fixture.status <> 'scheduled'
        or v_fixture.score_operation_id is distinct from p_operation_id
        or v_fixture.score_reserved_at is null then
    raise exception using
      errcode = '40001',
      message = 'TEAM_LEAGUE_FIXTURE_SCORE_RESERVATION_CONFLICT';
  else
    if v_status = 'complete' then
      if not exists (
        select 1
          from public.matches as match_row
         where match_row.club_id = v_fixture.club_id
           and match_row.id = p_official_match_id
           and match_row.league = v_fixture.league_name
           and match_row.match_type = 'Team League'
           and match_row.deleted_at is null
           and match_row.score_t1 = p_team_a_score
           and match_row.score_t2 = p_team_b_score
           and (
             case
               when match_row.score_t1 > match_row.score_t2
                 then v_fixture.team_a_id
               else v_fixture.team_b_id
             end
           ) = p_winner_team_id
           and array[match_row.t1_p1, match_row.t1_p2] @>
               array[p_team_a_player_1_id, p_team_a_player_2_id]
           and array[match_row.t1_p1, match_row.t1_p2] <@
               array[p_team_a_player_1_id, p_team_a_player_2_id]
           and array[match_row.t2_p1, match_row.t2_p2] @>
               array[p_team_b_player_1_id, p_team_b_player_2_id]
           and array[match_row.t2_p1, match_row.t2_p2] <@
               array[p_team_b_player_1_id, p_team_b_player_2_id]
        for share
      ) then
        raise exception using
          errcode = '22023',
          message = 'TEAM_LEAGUE_CANONICAL_MATCH_INVALID';
      end if;

      select coalesce(
               pg_catalog.jsonb_agg(
                 pg_catalog.jsonb_build_object(
                   'incoming_player_id', lineup.player_id,
                   'team_id', lineup.expected_team_id,
                   'source', 'substitute_pool'
                 )
                 order by lineup.expected_team_id::text, lineup.player_id
               ),
               '[]'::jsonb
             )
        into v_expected_substitutions
        from (values
          (p_team_a_player_1_id, v_fixture.team_a_id),
          (p_team_a_player_2_id, v_fixture.team_a_id),
          (p_team_b_player_1_id, v_fixture.team_b_id),
          (p_team_b_player_2_id, v_fixture.team_b_id)
        ) as lineup(player_id, expected_team_id)
       where not exists (
         select 1
           from public.team_league_team_members as member
          where member.player_id = lineup.player_id
            and member.status = 'active'
            and member.team_id = lineup.expected_team_id
       );
      v_pool_substitution_count :=
        pg_catalog.jsonb_array_length(v_expected_substitutions);
      if v_pool_substitution_count > 0 and (
        not v_settings.allow_substitutes
        or not v_settings.substitute_pool_enabled
        or exists (
          select 1
            from (values
              (p_team_a_player_1_id, v_fixture.team_a_id),
              (p_team_a_player_2_id, v_fixture.team_a_id),
              (p_team_b_player_1_id, v_fixture.team_b_id),
              (p_team_b_player_2_id, v_fixture.team_b_id)
            ) as lineup(player_id, expected_team_id)
           where not exists (
             select 1
               from public.team_league_team_members as member
              where member.player_id = lineup.player_id
                and member.status = 'active'
                and member.team_id = lineup.expected_team_id
           )
             and not exists (
               select 1
                 from public.team_league_substitute_pool as pool
                where pool.club_id = v_fixture.club_id
                  and pool.league_name = v_fixture.league_name
                  and pool.player_id = lineup.player_id
                  and pool.status = 'available'
             )
        )
      ) then
        raise exception using
          errcode = '55000',
          message = 'TEAM_LEAGUE_SUBSTITUTE_POOL_INVALID';
      end if;
      if pg_catalog.jsonb_array_length(p_substitutions)
           <> v_pool_substitution_count
         or not (p_substitutions @> v_expected_substitutions)
         or not (p_substitutions <@ v_expected_substitutions) then
        raise exception using
          errcode = '22023',
          message = 'TEAM_LEAGUE_SUBSTITUTION_AUDIT_INVALID';
      end if;

      if v_settings.team_category in ('mens', 'womens', 'mixed') and exists (
        select 1
          from (values
            (p_team_a_player_1_id, 'a'),
            (p_team_a_player_2_id, 'a'),
            (p_team_b_player_1_id, 'b'),
            (p_team_b_player_2_id, 'b')
          ) as lineup(player_id, side)
          left join public.players as player
            on player.id = lineup.player_id
           and player.club_id = v_fixture.club_id
         group by lineup.side
        having count(*) filter (
          where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
            in ('m', 'man', 'men', 'male', 'mens', 'men''s')
        ) <> case when v_settings.team_category = 'mens' then 2
                  when v_settings.team_category = 'mixed' then 1
                  else 0 end
            or count(*) filter (
          where pg_catalog.lower(pg_catalog.btrim(coalesce(player.gender, '')))
            in ('f', 'w', 'woman', 'women', 'female', 'womens', 'women''s')
        ) <> case when v_settings.team_category = 'womens' then 2
                  when v_settings.team_category = 'mixed' then 1
                  else 0 end
      ) then
        raise exception using
          errcode = '23514',
          message = 'TEAM_LEAGUE_LINEUP_CATEGORY_INVALID';
      end if;
    end if;

    update public.team_league_fixtures
       set score_operation_id = null
     where id = p_fixture_id;
  end if;

  select public.team_league_finalize_fixture_v1(
    p_operation_id,
    p_club_id,
    p_fixture_id,
    v_status,
    p_team_a_score,
    p_team_b_score,
    p_winner_team_id,
    p_official_match_id,
    p_team_a_player_1_id,
    p_team_a_player_2_id,
    p_team_b_player_1_id,
    p_team_b_player_2_id,
    p_substitutions,
    p_score_note,
    p_actor_email,
    p_actor_role,
    p_source
  ) into v_result;
  return v_result;
end
$function$;

notify pgrst, 'reload schema';
