-- Guard direct Match Uploader writes against a league lifecycle change that
-- races the application preflight. Keep the already-applied 20260726222339
-- migration immutable by wrapping its RPC in this forward-only migration.

alter function public.admin_apply_direct_match_entry_atomic_v1(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) rename to admin_apply_direct_match_entry_atomic_v1_base_20260727;

revoke all on function public.admin_apply_direct_match_entry_atomic_v1_base_20260727(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) from public, anon, authenticated;

grant execute on function public.admin_apply_direct_match_entry_atomic_v1_base_20260727(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) to service_role;

create function public.admin_apply_direct_match_entry_atomic_v1(
  p_operation_id uuid,
  p_club_id text,
  p_idempotency_key text,
  p_request_fingerprint text,
  p_match_format text,
  p_source text,
  p_actor_email text,
  p_actor_role text,
  p_request_json jsonb,
  p_result_summary jsonb,
  p_match_rows jsonb,
  p_player_updates jsonb,
  p_league_rating_updates jsonb,
  p_league_metadata_expectations jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = ''
as $function$
declare
  v_club_id text := nullif(pg_catalog.btrim(p_club_id), '');
  v_idempotency_key text :=
    nullif(pg_catalog.left(pg_catalog.btrim(p_idempotency_key), 160), '');
  v_match_format text :=
    pg_catalog.lower(pg_catalog.btrim(coalesce(p_match_format, '')));
  v_expectations jsonb :=
    coalesce(p_league_metadata_expectations, '[]'::jsonb);
  v_operation public.admin_direct_match_entry_operations%rowtype;
  v_metadata public.leagues_metadata%rowtype;
  v_item record;
  v_required_league_name text;
  v_expected_k_factor_text text;
  v_expected_k_factor integer;
  v_expected_is_active_json jsonb;
  v_expected_is_active boolean;
begin
  -- Use the same club lock as the base RPC. This makes the receipt check below
  -- deterministic and keeps every metadata row lock through the entire write.
  if v_club_id is not null and v_idempotency_key is not null then
    perform pg_catalog.pg_advisory_xact_lock(
      pg_catalog.hashtextextended(
        'jupr:direct-match-entry:' || v_club_id,
        0
      )
    );

    select operation.*
      into v_operation
      from public.admin_direct_match_entry_operations as operation
     where operation.club_id = v_club_id
       and operation.idempotency_key = v_idempotency_key;

    -- Exact retries must remain recoverable even after a league later ends.
    -- The base RPC still verifies the fingerprint and format before returning.
    if found then
      return public.admin_apply_direct_match_entry_atomic_v1_base_20260727(
        p_operation_id,
        p_club_id,
        p_idempotency_key,
        p_request_fingerprint,
        p_match_format,
        p_source,
        p_actor_email,
        p_actor_role,
        p_request_json,
        p_result_summary,
        p_match_rows,
        p_player_updates,
        p_league_rating_updates,
        p_league_metadata_expectations
      );
    end if;
  end if;

  -- Let the base RPC report its established validation error for malformed
  -- plans. Only well-formed expectation arrays are inspected here.
  if pg_catalog.jsonb_typeof(v_expectations) <> 'array' then
    return public.admin_apply_direct_match_entry_atomic_v1_base_20260727(
      p_operation_id,
      p_club_id,
      p_idempotency_key,
      p_request_fingerprint,
      p_match_format,
      p_source,
      p_actor_email,
      p_actor_role,
      p_request_json,
      p_result_summary,
      p_match_rows,
      p_player_updates,
      p_league_rating_updates,
      p_league_metadata_expectations
    );
  end if;

  -- The durable match rows are the write source of truth. Any non-reserved
  -- league name is an official-league context, so it must have an exact
  -- lifecycle snapshot even if a caller omits the expectation array.
  if pg_catalog.jsonb_typeof(coalesce(p_match_rows, '[]'::jsonb)) = 'array' then
    for v_required_league_name in
      select distinct pg_catalog.btrim(match_row.value->>'league')
        from pg_catalog.jsonb_array_elements(
          coalesce(p_match_rows, '[]'::jsonb)
        ) as match_row(value)
       where nullif(pg_catalog.btrim(match_row.value->>'league'), '') is not null
         and pg_catalog.lower(
               pg_catalog.btrim(match_row.value->>'league')
             ) not in ('overall', 'popup', 'singles')
       order by pg_catalog.btrim(match_row.value->>'league')
    loop
      if not exists (
        select 1
          from pg_catalog.jsonb_to_recordset(v_expectations)
            as required_expectation(
              league_name text,
              expected jsonb
            )
         where pg_catalog.lower(
                 pg_catalog.btrim(required_expectation.league_name)
               ) = pg_catalog.lower(v_required_league_name)
           and required_expectation.expected is not null
           and pg_catalog.jsonb_typeof(required_expectation.expected) = 'object'
      ) then
        raise exception using
          errcode = 'P0001',
          message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: an exact active official-league snapshot is required.';
      end if;
    end loop;
  end if;

  for v_item in
    select *
      from pg_catalog.jsonb_to_recordset(v_expectations)
        as metadata_expectation(
          league_name text,
          expected jsonb
        )
     order by metadata_expectation.league_name
  loop
    if v_item.expected is null
       or pg_catalog.jsonb_typeof(v_item.expected) <> 'object' then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: an active official league is required.';
    end if;

    v_expected_k_factor_text := v_item.expected->>'k_factor';
    if v_expected_k_factor_text is null
       or v_expected_k_factor_text !~ '^[+-]?[0-9]+$' then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: the official league snapshot is invalid.';
    end if;
    begin
      v_expected_k_factor := v_expected_k_factor_text::integer;
    exception
      when invalid_text_representation or numeric_value_out_of_range then
        raise exception using
          errcode = 'P0001',
          message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: the official league snapshot is invalid.';
    end;

    if not (v_item.expected ? 'is_active') then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: the official league snapshot is invalid.';
    end if;
    v_expected_is_active_json := v_item.expected->'is_active';
    if pg_catalog.jsonb_typeof(v_expected_is_active_json) = 'null' then
      v_expected_is_active := null;
    elsif pg_catalog.jsonb_typeof(v_expected_is_active_json) = 'boolean' then
      v_expected_is_active :=
        pg_catalog.lower(v_item.expected->>'is_active') = 'true';
    else
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: the official league snapshot is invalid.';
    end if;

    select metadata.*
      into v_metadata
      from public.leagues_metadata as metadata
     where metadata.club_id = v_club_id
       and pg_catalog.lower(metadata.league_name) =
            pg_catalog.lower(v_item.league_name)
     for update;

    if not found
       or v_metadata.id::text is distinct from v_item.expected->>'id'
       or v_metadata.club_id is distinct from v_item.expected->>'club_id'
       or v_metadata.league_name is distinct from
            v_item.expected->>'league_name'
       or v_metadata.k_factor is distinct from
            v_expected_k_factor
       or v_metadata.status is distinct from
            v_item.expected->>'status'
       or v_metadata.is_active is distinct from
            v_expected_is_active
       or (v_metadata.ended_at is null) is distinct from
            (v_item.expected->>'ended_at' is null)
       or v_metadata.ended_at is not null
       or not (
         case
           when pg_catalog.lower(
             pg_catalog.btrim(coalesce(v_metadata.status, ''))
           ) in ('active', 'running', 'live')
             then coalesce(v_metadata.is_active, true)
           when pg_catalog.btrim(coalesce(v_metadata.status, '')) = ''
             then coalesce(v_metadata.is_active, false)
           else false
         end
       ) then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: the official league changed or is no longer active.';
    end if;
  end loop;

  -- Singles use league metadata only as an active context guard. They do not
  -- mutate doubles league-rating rows, so the base RPC receives no league
  -- metadata CAS plan after this wrapper has locked and validated it.
  return public.admin_apply_direct_match_entry_atomic_v1_base_20260727(
    p_operation_id,
    p_club_id,
    p_idempotency_key,
    p_request_fingerprint,
    p_match_format,
    p_source,
    p_actor_email,
    p_actor_role,
    p_request_json,
    p_result_summary,
    p_match_rows,
    p_player_updates,
    p_league_rating_updates,
    case
      when v_match_format = 'singles' then '[]'::jsonb
      else p_league_metadata_expectations
    end
  );
end
$function$;

revoke all on function public.admin_apply_direct_match_entry_atomic_v1(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) from public, anon, authenticated;

grant execute on function public.admin_apply_direct_match_entry_atomic_v1(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) to service_role;

comment on function public.admin_apply_direct_match_entry_atomic_v1(
  uuid,
  text,
  text,
  text,
  text,
  text,
  text,
  text,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb,
  jsonb
) is
  'Applies idempotent direct match plans while locking and validating active official league metadata.';

notify pgrst, 'reload schema';
