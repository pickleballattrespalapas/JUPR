-- Durable, idempotent Replay History requests for Next/FastAPI.

alter table public.matches add column if not exists notes text null;

alter table public.replay_jobs
  add column if not exists idempotency_key text null,
  add column if not exists source text null,
  add column if not exists attempt_count integer not null default 0;

create unique index if not exists uq_replay_jobs_club_idempotency
  on public.replay_jobs (club_id, idempotency_key)
  where idempotency_key is not null;

create index if not exists idx_replay_jobs_club_created
  on public.replay_jobs (club_id, created_at desc);

alter table public.replay_jobs enable row level security;
revoke all on table public.replay_jobs from public, anon, authenticated;
grant all privileges on table public.replay_jobs to service_role;

create table if not exists public.match_edit_operations (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  idempotency_key text not null,
  status text not null,
  patches_json jsonb not null default '[]'::jsonb,
  before_json jsonb not null default '[]'::jsonb,
  after_json jsonb not null default '[]'::jsonb,
  recompute_scope jsonb not null default '{"standings": false, "ratings": false}'::jsonb,
  replay_target text,
  replay_job_id uuid references public.replay_jobs(id) on delete set null,
  result_json jsonb not null default '{}'::jsonb,
  error_text text,
  actor_email text not null,
  actor_role text not null,
  source text not null,
  correction_note text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  finished_at timestamptz,
  constraint match_edit_operations_status_check check (
    status in ('pending_replay', 'succeeded', 'recovery_required')
  ),
  constraint match_edit_operations_club_idempotency_unique unique (club_id, idempotency_key)
);

create index if not exists idx_match_edit_operations_club_created
  on public.match_edit_operations (club_id, created_at desc);

alter table public.match_edit_operations enable row level security;
revoke all on table public.match_edit_operations from public, anon, authenticated;
grant all privileges on table public.match_edit_operations to service_role;

create or replace function public.apply_match_log_patches_atomic(
  p_club_id text,
  p_patches jsonb,
  p_actor_email text,
  p_actor_role text,
  p_source text,
  p_correction_note text,
  p_idempotency_key text,
  p_replay_target text default 'ALL (Full System Reset)'
)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_operation_id uuid := gen_random_uuid();
  v_existing public.match_edit_operations%rowtype;
  v_patch jsonb;
  v_before jsonb;
  v_after jsonb;
  v_before_rows jsonb := '[]'::jsonb;
  v_after_rows jsonb := '[]'::jsonb;
  v_ids bigint[] := '{}';
  v_match_id bigint;
  v_updated_ids bigint[] := '{}';
  v_standings boolean := false;
  v_ratings boolean := false;
  v_replay_job_id uuid;
  v_status text;
  v_clean_key text := nullif(left(btrim(coalesce(p_idempotency_key, '')), 160), '');
  v_clean_source text := left(btrim(coalesce(p_source, 'next_match_log')), 120);
  v_clean_note text := nullif(left(btrim(coalesce(p_correction_note, '')), 2000), '');
  v_replay_key text;
begin
  if v_clean_key is null then
    raise exception 'A stable idempotency key is required.' using errcode = '22023';
  end if;
  if jsonb_typeof(p_patches) <> 'array' or jsonb_array_length(p_patches) < 1 then
    raise exception 'At least one match patch is required.' using errcode = '22023';
  end if;
  if jsonb_array_length(p_patches) > 100 then
    raise exception 'No more than 100 match patches may be applied at once.' using errcode = '22023';
  end if;

  perform pg_advisory_xact_lock(hashtextextended(p_club_id || ':' || v_clean_key, 0));
  select * into v_existing
  from public.match_edit_operations
  where club_id = p_club_id and idempotency_key = v_clean_key;
  if found then
    return jsonb_build_object(
      'operation_id', v_existing.id,
      'status', v_existing.status,
      'updated_ids', coalesce(v_existing.result_json->'updated_ids', '[]'::jsonb),
      'updated_count', coalesce(jsonb_array_length(coalesce(v_existing.result_json->'updated_ids', '[]'::jsonb)), 0),
      'recompute_scope', v_existing.recompute_scope,
      'replay_target', v_existing.replay_target,
      'replay_job_id', v_existing.replay_job_id,
      'result_json', v_existing.result_json,
      'idempotent', true,
      'error_text', v_existing.error_text
    );
  end if;

  for v_patch in select value from jsonb_array_elements(p_patches)
  loop
    if jsonb_typeof(v_patch) <> 'object' or not (v_patch ? 'id') then
      raise exception 'Every patch must be an object with an id.' using errcode = '22023';
    end if;
    if exists (
      select 1 from jsonb_object_keys(v_patch) as key
      where key not in ('id','league','date','week_tag','match_type','notes','t1_p1','t1_p2','t2_p1','t2_p2','score_t1','score_t2')
    ) then
      raise exception 'A patch contains an unsupported field.' using errcode = '22023';
    end if;
    if exists (
      select 1
      from unnest(array['t1_p1','t1_p2','t2_p1','t2_p2']) player_key
      where v_patch ? player_key and nullif(btrim(v_patch->>player_key), '') is null
    ) then
      raise exception 'Player replacements cannot be blank.' using errcode = '22023';
    end if;
    v_match_id := (v_patch->>'id')::bigint;
    if v_match_id = any(v_ids) then
      raise exception 'Duplicate match ids are not allowed in one operation.' using errcode = '22023';
    end if;
    v_ids := array_append(v_ids, v_match_id);

    select to_jsonb(m) into v_before
    from public.matches m
    where m.club_id::text = p_club_id and m.id = v_match_id
    for update;
    if v_before is null then
      raise exception 'Match % was not found in this club.', v_match_id using errcode = 'P0002';
    end if;
    if coalesce(v_before->>'deleted_at', '') <> '' then
      raise exception 'Match % is excluded and cannot be edited here.', v_match_id using errcode = '22023';
    end if;
    v_before_rows := v_before_rows || jsonb_build_array(v_before);

    v_standings := v_standings or v_patch ?| array['league','date','week_tag','t1_p1','t1_p2','t2_p1','t2_p2','score_t1','score_t2'];
    v_ratings := v_ratings or v_patch ?| array['league','date','match_type','t1_p1','t1_p2','t2_p1','t2_p2','score_t1','score_t2'];

    update public.matches m
    set
      league = case when v_patch ? 'league' then nullif(btrim(v_patch->>'league'), '') else m.league end,
      date = case when v_patch ? 'date' then (v_patch->>'date')::timestamptz else m.date end,
      week_tag = case
        when v_patch ? 'week_tag' then nullif(btrim(v_patch->>'week_tag'), '')
        when v_patch ? 'league' or v_patch ? 'date' then null
        else m.week_tag
      end,
      match_type = case when v_patch ? 'match_type' then nullif(btrim(v_patch->>'match_type'), '') else m.match_type end,
      notes = case when v_patch ? 'notes' then nullif(left(btrim(v_patch->>'notes'), 2000), '') else m.notes end,
      t1_p1 = case when v_patch ? 't1_p1' then (v_patch->>'t1_p1')::bigint else m.t1_p1 end,
      t1_p2 = case when v_patch ? 't1_p2' then (v_patch->>'t1_p2')::bigint else m.t1_p2 end,
      t2_p1 = case when v_patch ? 't2_p1' then (v_patch->>'t2_p1')::bigint else m.t2_p1 end,
      t2_p2 = case when v_patch ? 't2_p2' then (v_patch->>'t2_p2')::bigint else m.t2_p2 end,
      score_t1 = case when v_patch ? 'score_t1' then (v_patch->>'score_t1')::integer else m.score_t1 end,
      score_t2 = case when v_patch ? 'score_t2' then (v_patch->>'score_t2')::integer else m.score_t2 end,
      updated_at = now(),
      updated_by = lower(btrim(p_actor_email)),
      correction_note = coalesce(v_clean_note, m.correction_note)
    where m.club_id::text = p_club_id and m.id = v_match_id;

    select to_jsonb(m) into v_after
    from public.matches m
    where m.club_id::text = p_club_id and m.id = v_match_id;
    if v_patch ?| array['score_t1','score_t2'] then
      if (v_after->>'score_t1') is null or (v_after->>'score_t2') is null
         or (v_after->>'score_t1')::integer < 0 or (v_after->>'score_t2')::integer < 0 then
        raise exception 'Match % scores must be present and non-negative.', v_match_id using errcode = '22023';
      end if;
    end if;
    if v_patch ?| array['t1_p1','t1_p2','t2_p1','t2_p2'] then
      if (v_after->>'t1_p1') is null or (v_after->>'t2_p1') is null
         or ((v_after->>'t1_p2') is null) <> ((v_after->>'t2_p2') is null)
         or (select count(value) from jsonb_array_elements_text(jsonb_build_array(v_after->'t1_p1', v_after->'t1_p2', v_after->'t2_p1', v_after->'t2_p2')) where value is not null) not in (2, 4)
         or (select count(distinct value) from jsonb_array_elements_text(jsonb_build_array(v_after->'t1_p1', v_after->'t1_p2', v_after->'t2_p1', v_after->'t2_p2')) where value is not null)
            <> (select count(value) from jsonb_array_elements_text(jsonb_build_array(v_after->'t1_p1', v_after->'t1_p2', v_after->'t2_p1', v_after->'t2_p2')) where value is not null) then
        raise exception 'Match % must contain balanced teams with two or four distinct players.', v_match_id using errcode = '22023';
      end if;
      if exists (
        select 1
        from jsonb_array_elements_text(jsonb_build_array(v_after->'t1_p1', v_after->'t1_p2', v_after->'t2_p1', v_after->'t2_p2')) player_id
        where player_id is not null and not exists (
          select 1 from public.players p
          where p.club_id::text = p_club_id and p.id = player_id::bigint
        )
      ) then
        raise exception 'Match % contains a player outside this club.', v_match_id using errcode = '22023';
      end if;
    end if;
    v_after_rows := v_after_rows || jsonb_build_array(v_after);
    v_updated_ids := array_append(v_updated_ids, v_match_id);
  end loop;

  if v_ratings then
    v_replay_key := left('match-edit:' || v_clean_key, 160);
    insert into public.replay_jobs (
      club_id, target_reset, status, actor_email, actor_role, idempotency_key, source
    ) values (
      p_club_id,
      coalesce(nullif(btrim(p_replay_target), ''), 'ALL (Full System Reset)'),
      'pending', lower(btrim(p_actor_email)), p_actor_role, v_replay_key, v_clean_source
    )
    on conflict (club_id, idempotency_key) where idempotency_key is not null
    do update set updated_at = now()
    returning id into v_replay_job_id;
    v_status := 'pending_replay';
  else
    v_status := 'succeeded';
  end if;

  insert into public.match_edit_operations (
    id, club_id, idempotency_key, status, patches_json, before_json, after_json,
    recompute_scope, replay_target, replay_job_id, result_json, actor_email,
    actor_role, source, correction_note, finished_at
  ) values (
    v_operation_id, p_club_id, v_clean_key, v_status, p_patches, v_before_rows, v_after_rows,
    jsonb_build_object('standings', v_standings, 'ratings', v_ratings),
    case when v_ratings then coalesce(nullif(btrim(p_replay_target), ''), 'ALL (Full System Reset)') else null end,
    v_replay_job_id,
    jsonb_build_object('updated_ids', to_jsonb(v_updated_ids)),
    lower(btrim(p_actor_email)), p_actor_role, v_clean_source, v_clean_note,
    case when v_status = 'succeeded' then now() else null end
  );

  insert into public.admin_activity_log (
    club_id, actor_email, actor_role, action_type, entity_type, entity_id,
    before_json, after_json, note, source_page, flagged_for_review
  ) values (
    p_club_id, lower(btrim(p_actor_email)), p_actor_role, 'match_edit', 'match_edit_operation',
    v_operation_id::text, v_before_rows,
    jsonb_build_object(
      'rows', v_after_rows,
      'recompute_scope', jsonb_build_object('standings', v_standings, 'ratings', v_ratings),
      'replay_job_id', v_replay_job_id,
      'idempotency_key', v_clean_key
    ),
    v_clean_note, v_clean_source, v_ratings
  );

  return jsonb_build_object(
    'operation_id', v_operation_id,
    'status', v_status,
    'updated_ids', to_jsonb(v_updated_ids),
    'updated_count', cardinality(v_updated_ids),
    'recompute_scope', jsonb_build_object('standings', v_standings, 'ratings', v_ratings),
    'replay_target', case when v_ratings then coalesce(nullif(btrim(p_replay_target), ''), 'ALL (Full System Reset)') else null end,
    'replay_job_id', v_replay_job_id,
    'result_json', jsonb_build_object('updated_ids', to_jsonb(v_updated_ids)),
    'idempotent', false
  );
end;
$$;

revoke all on function public.apply_match_log_patches_atomic(text, jsonb, text, text, text, text, text, text)
  from public, anon, authenticated;
grant execute on function public.apply_match_log_patches_atomic(text, jsonb, text, text, text, text, text, text)
  to service_role;
