-- Durable, server-only coordination for the staging Admin Diagnostics slice.
-- Browser roles are deliberately excluded; FastAPI uses the service role after
-- JWT permission checks and records audit intent before calling any write RPC.

create table if not exists public.admin_guarded_operations (
  id uuid primary key default gen_random_uuid(),
  club_id text not null,
  workflow text not null,
  action text not null,
  operation_key text not null,
  request_fingerprint text not null,
  status text not null default 'intent_recorded',
  before_json jsonb null,
  after_json jsonb null,
  result_json jsonb null,
  actor_email text not null,
  actor_role text not null,
  source text not null,
  error_text text null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  finished_at timestamptz null,
  constraint admin_guarded_operations_key_unique unique (club_id, workflow, operation_key),
  constraint admin_guarded_operations_status_check check (
    status in ('intent_recorded','running','completed','recovery_required','compensated','failed')
  )
);

create index if not exists idx_admin_guarded_operations_recovery
  on public.admin_guarded_operations (club_id, workflow, status, updated_at desc);

alter table public.admin_guarded_operations enable row level security;
alter table public.admin_guarded_operations force row level security;
revoke all on table public.admin_guarded_operations from public, anon, authenticated;
grant select, insert, update on table public.admin_guarded_operations to service_role;

create or replace function public.apply_match_canonical_patches_atomic(
  p_club_id text,
  p_operation_key text,
  p_request_fingerprint text,
  p_patches jsonb
)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
  v_operation public.admin_guarded_operations%rowtype;
  v_patch jsonb;
  v_expected jsonb;
  v_changes jsonb;
  v_before jsonb;
  v_after jsonb;
  v_before_rows jsonb := '[]'::jsonb;
  v_after_rows jsonb := '[]'::jsonb;
  v_match_id bigint;
  v_ids bigint[] := '{}';
  v_updated_ids bigint[] := '{}';
begin
  if nullif(btrim(coalesce(p_operation_key, '')), '') is null
     or nullif(btrim(coalesce(p_request_fingerprint, '')), '') is null then
    raise exception 'Operation key and request fingerprint are required.' using errcode = '22023';
  end if;
  if jsonb_typeof(p_patches) <> 'array'
     or jsonb_array_length(p_patches) < 1
     or jsonb_array_length(p_patches) > 100 then
    raise exception 'Canonical apply requires 1-100 reviewed patches.' using errcode = '22023';
  end if;

  perform pg_advisory_xact_lock(hashtextextended(p_club_id || ':match-canonical:' || p_operation_key, 0));
  select * into v_operation
    from public.admin_guarded_operations
   where club_id = p_club_id
     and workflow = 'match_canonical_normalize'
     and operation_key = p_operation_key
   for update;
  if not found then
    raise exception 'Guarded operation intent is missing.' using errcode = 'P0002';
  end if;
  if v_operation.request_fingerprint <> p_request_fingerprint then
    raise exception 'Operation request fingerprint does not match.' using errcode = '40001';
  end if;
  if v_operation.status = 'completed' then
    return coalesce(v_operation.result_json, '{}'::jsonb) || jsonb_build_object('idempotent', true);
  end if;
  if v_operation.status <> 'intent_recorded' then
    raise exception 'Operation requires recovery before retry.' using errcode = '55000';
  end if;

  update public.admin_guarded_operations
     set status = 'running', updated_at = now()
   where id = v_operation.id;

  for v_patch in select value from jsonb_array_elements(p_patches)
  loop
    if jsonb_typeof(v_patch) <> 'object'
       or not (v_patch ? 'id')
       or jsonb_typeof(v_patch->'expected') <> 'object'
       or jsonb_typeof(v_patch->'changes') <> 'object' then
      raise exception 'Every canonical patch needs id, expected, and changes.' using errcode = '22023';
    end if;
    v_match_id := (v_patch->>'id')::bigint;
    if v_match_id = any(v_ids) then
      raise exception 'Duplicate match IDs are not allowed.' using errcode = '22023';
    end if;
    v_ids := array_append(v_ids, v_match_id);
    v_expected := v_patch->'expected';
    v_changes := v_patch->'changes';
    if v_changes = '{}'::jsonb or exists (
      select 1 from jsonb_object_keys(v_changes) as key
       where key not in ('league','match_type','context_type','context_id','t1_p1','t1_p2','t2_p1','t2_p2','score_t1','score_t2')
    ) then
      raise exception 'Canonical patch contains no changes or an unsupported field.' using errcode = '22023';
    end if;
    if exists (
      select 1 from jsonb_object_keys(v_expected) as key
       where key not in ('league','match_type','context_type','context_id','t1_p1','t1_p2','t2_p1','t2_p2','score_t1','score_t2','updated_at')
    ) then
      raise exception 'Canonical expected state contains an unsupported field.' using errcode = '22023';
    end if;

    select to_jsonb(m) into v_before
      from public.matches m
     where m.club_id::text = p_club_id and m.id = v_match_id
     for update;
    if v_before is null then
      raise exception 'Match % was not found in this club.', v_match_id using errcode = 'P0002';
    end if;
    if coalesce(v_before->>'deleted_at', '') <> '' then
      raise exception 'Match % is excluded and cannot be normalized.', v_match_id using errcode = '22023';
    end if;
    if exists (
      select 1 from jsonb_object_keys(v_expected) as key
       where (v_before->key) is distinct from (v_expected->key)
    ) then
      raise exception 'Match % changed after the reviewed dry run.', v_match_id using errcode = '40001';
    end if;

    update public.matches m
       set league = case when v_changes ? 'league' then nullif(btrim(v_changes->>'league'), '') else m.league end,
           match_type = case when v_changes ? 'match_type' then nullif(btrim(v_changes->>'match_type'), '') else m.match_type end,
           context_type = case when v_changes ? 'context_type' then nullif(btrim(v_changes->>'context_type'), '') else m.context_type end,
           context_id = case when v_changes ? 'context_id' then nullif(btrim(v_changes->>'context_id'), '') else m.context_id end,
           t1_p1 = case when v_changes ? 't1_p1' then (v_changes->>'t1_p1')::bigint else m.t1_p1 end,
           t1_p2 = case when v_changes ? 't1_p2' then (v_changes->>'t1_p2')::bigint else m.t1_p2 end,
           t2_p1 = case when v_changes ? 't2_p1' then (v_changes->>'t2_p1')::bigint else m.t2_p1 end,
           t2_p2 = case when v_changes ? 't2_p2' then (v_changes->>'t2_p2')::bigint else m.t2_p2 end,
           score_t1 = case when v_changes ? 'score_t1' then (v_changes->>'score_t1')::integer else m.score_t1 end,
           score_t2 = case when v_changes ? 'score_t2' then (v_changes->>'score_t2')::integer else m.score_t2 end,
           updated_at = now(),
           updated_by = lower(btrim(v_operation.actor_email)),
           correction_note = 'Canonical normalization ' || p_operation_key
     where m.club_id::text = p_club_id and m.id = v_match_id;

    select to_jsonb(m) into v_after
      from public.matches m
     where m.club_id::text = p_club_id and m.id = v_match_id;
    if (v_after->>'t1_p1') is null or (v_after->>'t2_p1') is null
       or (v_after->>'score_t1') is null or (v_after->>'score_t2') is null
       or (v_after->>'score_t1')::integer < 0 or (v_after->>'score_t2')::integer < 0 then
      raise exception 'Canonical readback validation failed for match %.', v_match_id using errcode = '22023';
    end if;
    v_before_rows := v_before_rows || jsonb_build_array(v_before);
    v_after_rows := v_after_rows || jsonb_build_array(v_after);
    v_updated_ids := array_append(v_updated_ids, v_match_id);
  end loop;

  update public.admin_guarded_operations
     set status = 'completed',
         before_json = v_before_rows,
         after_json = v_after_rows,
         result_json = jsonb_build_object(
           'ok', true,
           'mode', 'match_canonical_normalize',
           'atomic', true,
           'operation_key', p_operation_key,
           'updated_ids', to_jsonb(v_updated_ids),
           'updated_count', cardinality(v_updated_ids),
           'recovery', jsonb_build_object(
             'match_log', '/admin/match-log',
             'replay_history', '/admin/replay-history'
           )
         ),
         updated_at = now(), finished_at = now()
   where id = v_operation.id;

  insert into public.admin_activity_log (
    club_id, actor_email, actor_role, action_type, entity_type, entity_id,
    before_json, after_json, note, source_page, flagged_for_review
  ) values (
    p_club_id, v_operation.actor_email, v_operation.actor_role,
    'match_canonical_normalize_completed', 'admin_guarded_operation', p_operation_key,
    v_before_rows,
    jsonb_build_object(
      'rows', v_after_rows,
      'request_fingerprint', p_request_fingerprint,
      'updated_ids', to_jsonb(v_updated_ids),
      'source_client', 'fastapi/nextjs'
    ),
    'Atomic canonical normalization completed', v_operation.source, true
  );

  return jsonb_build_object(
    'ok', true,
    'mode', 'match_canonical_normalize',
    'atomic', true,
    'operation_key', p_operation_key,
    'updated_ids', to_jsonb(v_updated_ids),
    'updated_count', cardinality(v_updated_ids),
    'idempotent', false,
    'recovery', jsonb_build_object(
      'match_log', '/admin/match-log',
      'replay_history', '/admin/replay-history'
    )
  );
end;
$$;

revoke all on function public.apply_match_canonical_patches_atomic(text, text, text, jsonb)
  from public, anon, authenticated;
grant execute on function public.apply_match_canonical_patches_atomic(text, text, text, jsonb)
  to service_role;

notify pgrst, 'reload schema';
