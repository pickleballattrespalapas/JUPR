-- Separate official leagues by match format so singles matches cannot be
-- entered into doubles leagues, and vice versa.

alter table public.leagues_metadata
  add column if not exists match_format text;

update public.leagues_metadata
   set match_format = 'doubles'
 where match_format is null
    or pg_catalog.lower(pg_catalog.btrim(match_format)) not in ('doubles', 'singles');

alter table public.leagues_metadata
  alter column match_format set default 'doubles';

alter table public.leagues_metadata
  alter column match_format set not null;

alter table public.leagues_metadata
  drop constraint if exists leagues_metadata_match_format_check;

alter table public.leagues_metadata
  add constraint leagues_metadata_match_format_check
  check (match_format in ('doubles', 'singles'));

create index if not exists leagues_metadata_active_match_format_idx
  on public.leagues_metadata (club_id, match_format, is_active, status)
  where ended_at is null;

comment on column public.leagues_metadata.match_format is
  'Official league match format. Doubles and singles leagues are intentionally separate.';

-- Keep the match-format decision under the same club transaction lock as the
-- existing lifecycle and player-state compare-and-swap guards. Exact retries
-- still return their original durable receipt even if a league later changes.
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
) rename to admin_apply_direct_match_entry_atomic_v1_base_20260731_format;

revoke all on function public.admin_apply_direct_match_entry_atomic_v1_base_20260731_format(
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

grant execute on function public.admin_apply_direct_match_entry_atomic_v1_base_20260731_format(
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
  v_required_league_name text;
begin
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

    if found then
      return public.admin_apply_direct_match_entry_atomic_v1_base_20260731_format(
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

  -- Preserve the established validation errors for malformed plans.
  if v_match_format not in ('doubles', 'singles')
     or pg_catalog.jsonb_typeof(v_expectations) <> 'array'
     or pg_catalog.jsonb_typeof(
          coalesce(p_match_rows, '[]'::jsonb)
        ) <> 'array' then
    return public.admin_apply_direct_match_entry_atomic_v1_base_20260731_format(
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
    select metadata.*
      into v_metadata
      from public.leagues_metadata as metadata
     where metadata.club_id = v_club_id
       and pg_catalog.lower(metadata.league_name) =
            pg_catalog.lower(v_required_league_name)
     for update;

    if not found
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
        message = 'JUPR_DIRECT_MATCH_LEAGUE_METADATA_STALE: an active official league is required.';
    end if;

    if pg_catalog.lower(
         pg_catalog.btrim(coalesce(v_metadata.match_format, ''))
       ) is distinct from v_match_format then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_FORMAT_MISMATCH: the official league does not accept this match format.';
    end if;

    if not exists (
      select 1
        from pg_catalog.jsonb_to_recordset(v_expectations)
          as format_expectation(
            league_name text,
            expected jsonb
          )
       where pg_catalog.lower(
               pg_catalog.btrim(format_expectation.league_name)
             ) = pg_catalog.lower(v_required_league_name)
         and pg_catalog.jsonb_typeof(format_expectation.expected) = 'object'
         and pg_catalog.lower(
               pg_catalog.btrim(
                 coalesce(
                   format_expectation.expected->>'match_format',
                   ''
                 )
               )
             ) = v_match_format
    ) then
      raise exception using
        errcode = 'P0001',
        message = 'JUPR_DIRECT_MATCH_LEAGUE_FORMAT_STALE: an exact league match-format snapshot is required.';
    end if;
  end loop;

  return public.admin_apply_direct_match_entry_atomic_v1_base_20260731_format(
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
  'Applies idempotent direct match plans while atomically validating active official league lifecycle and singles/doubles format.';

notify pgrst, 'reload schema';
