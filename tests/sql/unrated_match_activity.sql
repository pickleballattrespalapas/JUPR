-- Staging/local fixture only. Every inserted profile, match, receipt and audit
-- rolls back. Apply 20261109004104_unrated_match_activity.sql first.
begin;
do $test$
declare
  club text := 'unrated-activity-' || gen_random_uuid()::text;
  ids bigint[] := '{}';
  pid bigint;
  i integer;
  format text;
  operation uuid;
  fingerprint text := repeat('a', 64);
  request_key text;
  rows jsonb;
  updates jsonb;
  invalid_updates jsonb;
  before_players jsonb;
  after_players jsonb;
  result jsonb;
  replay jsonb;
  match_count integer;
  fields text[];
  game_time timestamptz := now() - interval '1 hour';
begin
  insert into public.clubs(id,slug,name,is_active,status,onboarding_status)
  values(club,club,'Unrated activity fixture',false,'draft','draft');
  for i in 1..5 loop
    pid := 1800000000 + (random()*1000000)::bigint; -- Positive int32 for installed triggers.
    ids := array_append(ids,pid);
    insert into public.players(id,club_id,name,normalized_name,rating,wins,losses,matches_played,
      active,inactive_at,last_game_at,singles_rating,singles_wins,singles_losses,singles_matches_played)
    values(pid,club,'Activity '||i,'unrated activity '||i,1400,3,2,5,
      false,now()-interval '30 days',null,null,0,0,0);
  end loop;
  foreach format in array array['doubles','singles'] loop
    operation := gen_random_uuid();
    request_key := 'activity-' || format || '-' || operation::text;
    fields := case when format = 'singles'
      then array['singles_rating','singles_wins','singles_losses','singles_matches_played','singles_last_game_at']
      else array['rating','wins','losses','matches_played'] end;
    rows := jsonb_build_array(jsonb_build_object(
      'club_id',club,'date',game_time,'league','Overall','match_format',format,
      'rating_scope','unrated','score_t1',11,'score_t2',7,'t1_p1',ids[1],'t2_p1',ids[3],
      't1_p2',case when format='doubles' then ids[2] else null end,
      't2_p2',case when format='doubles' then ids[4] else null end,
      'singles_replay_managed',format='singles'));
    select jsonb_agg(jsonb_build_object('player_id',p.id,'rating_mode',format,
      'activity_only',true,'expected',projection,'after',projection || jsonb_build_object(
        'active',true,'inactive_at',null,'last_game_at',game_time)) order by p.id)
      into updates
      from public.players p
      cross join lateral (select jsonb_object_agg(k,to_jsonb(p)->k) as projection
        from unnest(fields || array['active','inactive_at','last_game_at']) k) snapshot
     where p.id = any(case when format='doubles' then ids[1:4] else array[ids[1],ids[3]] end);
    select jsonb_agg(to_jsonb(p) - 'active' - 'inactive_at' - 'last_game_at' order by p.id)
      into before_players from public.players p where p.club_id=club;

    -- An unrated projection cannot smuggle a rating/counter change or an
    -- unrelated player through the relaxed participant-coverage check.
    invalid_updates := jsonb_set(updates,array['0','after',fields[1]],'1600'::jsonb);
    begin
      perform public.admin_apply_direct_match_entry_atomic_v1(operation,club,request_key,fingerprint,
        format,'activity_fixture','fixture@example.test','administrator','{}','{"inserted":1}',rows,invalid_updates,'[]','[]');
      raise exception 'Unrated projection changed a rating';
    exception when invalid_parameter_value then
      if sqlerrm not like '%PLAYER_PLAN_INVALID%' then raise; end if;
    end;
    invalid_updates := jsonb_set(updates,'{0,player_id}',to_jsonb(ids[5]));
    begin
      perform public.admin_apply_direct_match_entry_atomic_v1(operation,club,request_key,fingerprint,
        format,'activity_fixture','fixture@example.test','administrator','{}','{"inserted":1}',rows,invalid_updates,'[]','[]');
      raise exception 'Unrelated player projection accepted';
    exception when invalid_parameter_value then
      if sqlerrm not like '%PLAYER_PLAN_INVALID%' then raise; end if;
    end;
    begin
      perform public.admin_apply_direct_match_entry_atomic_v1(operation,club,request_key,fingerprint,
        format,'activity_fixture','fixture@example.test','administrator','{}','{"inserted":1}',
        jsonb_set(rows,'{0,rating_scope}','"overall_only"'),'[]','[]','[]');
      raise exception 'Rated participants did not require projections';
    exception when invalid_parameter_value then
      if sqlerrm not like '%PLAYER_PLAN_INVALID%' then raise; end if;
    end;
    invalid_updates := jsonb_set(updates,'{0,expected,active}',to_jsonb(format='doubles'));
    begin
      perform public.admin_apply_direct_match_entry_atomic_v1(operation,club,request_key,fingerprint,
        format,'activity_fixture','fixture@example.test','administrator','{}','{"inserted":1}',rows,invalid_updates,'[]','[]');
      raise exception 'Stale activity snapshot accepted';
    exception when raise_exception then
      if sqlerrm not like '%PLAYER_STALE%' then raise; end if;
    end;
    if exists(select 1 from public.admin_direct_match_entry_operations where id=operation) then
      raise exception 'Rejected plans left a receipt';
    end if;

    select count(*) into match_count from public.matches where club_id=club;
    result := public.admin_apply_direct_match_entry_atomic_v1(operation,club,request_key,fingerprint,
      format,'activity_fixture','fixture@example.test','administrator','{}','{"inserted":1}',rows,updates,'[]','[]');
    replay := public.admin_apply_direct_match_entry_atomic_v1(operation,club,request_key,fingerprint,
      format,'activity_fixture','fixture@example.test','administrator','{}','{"inserted":1}',rows,updates,'[]','[]');
    if not (result->>'committed')::boolean or not (replay->>'idempotent')::boolean
      or result->'match_ids' <> replay->'match_ids'
      or (select count(*) from public.matches where club_id=club) <> match_count+1 then
      raise exception 'Activity commit/receipt replay was not atomic and idempotent';
    end if;
    select jsonb_agg(to_jsonb(p) - 'active' - 'inactive_at' - 'last_game_at' order by p.id)
      into after_players from public.players p where p.club_id=club;
    -- Ignore a schema-level updated_at maintenance column if installed.
    if (select jsonb_agg(value - 'updated_at') from jsonb_array_elements(before_players))
       is distinct from
       (select jsonb_agg(value - 'updated_at') from jsonb_array_elements(after_players)) then
      raise exception 'Unrated activity changed profile rating/counters/baselines';
    end if;
    if exists(select 1 from public.players where id=any(ids[1:4])
        and (active is not true or inactive_at is not null or last_game_at <> game_time))
      or (select last_game_at is not null from public.players where id=ids[5]) then
      raise exception 'Activity did not update exactly the recorded participants';
    end if;

    -- Deployment order is safe: the previous writer can still omit unrated
    -- activity projections during a rolling release.
    perform public.admin_apply_direct_match_entry_atomic_v1(gen_random_uuid(),club,request_key||'-old',fingerprint,
      format,'activity_fixture','fixture@example.test','administrator','{}','{"inserted":1}',rows,'[]','[]','[]');
  end loop;
end
$test$;
select 'PASS: unrated doubles/singles activity, unchanged ratings and nullable seeds, CAS, coverage, atomic replay, old writers' as result;
rollback;
