-- A verified edit link may withdraw only its own unchanged registration.
-- Reuse the cancellation transaction so partner, draw and commerce guards,
-- historical payment records and the private cancellation audit stay intact.
create or replace function public.server_withdraw_public_tournament_registration(
  p_club_id text, p_tournament_id text, p_registration_id text,
  p_email_hash text, p_expected_updated_at timestamptz,
  p_expected_selection_versions jsonb
) returns jsonb language plpgsql security invoker set search_path = '' as $$
declare
  v_registration public.tournament_registrations%rowtype;
  v_id text;
  v_scope text[];
  v_actor text := 'public-withdrawal:' || p_registration_id;
begin
  if nullif(p_email_hash, '') is null or p_expected_updated_at is null
     or not exists (select 1 from public.tournaments
       where id::text = p_tournament_id and club_id = p_club_id) then
    raise exception 'JUPR_WITHDRAW_IDENTITY: Invalid registration edit link.';
  end if;
  -- Match the cancellation/pairing lock order before checking any versions.
  for v_id in select id from public.tournament_registrations
    where tournament_id = p_tournament_id order by id loop
    perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('jupr:tournament-registration:' || v_id, 0));
  end loop;
  perform pg_catalog.pg_advisory_xact_lock(pg_catalog.hashtextextended('jupr:tournament-registration:' || p_registration_id, 0));
  select array_agg(id order by id) into v_scope from public.tournament_registration_selections
    where tournament_id = p_tournament_id;
  if coalesce(cardinality(v_scope), 0) > 0 then
    perform private.lock_tournament_registration_selection_scope(v_scope);
  end if;
  select * into v_registration from public.tournament_registrations
    where id = p_registration_id and tournament_id = p_tournament_id for update;
  if not found then
    -- A lost response can be retried without cancelling a later registration.
    if exists (select 1 from private.tournament_registration_cancellations c
      where c.tournament_id::text = p_tournament_id and c.registration_id = p_registration_id
        and c.actor_email = v_actor
        and (c.before_json->'registration'->>'updated_at')::timestamptz = p_expected_updated_at
        and pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(lower(btrim(c.before_json->'registration'->>'email')), 'UTF8')), 'hex') = p_email_hash) then
      return jsonb_build_object('ok', true, 'withdrawn', true, 'registration_id', p_registration_id, 'idempotent', true);
    end if;
    raise exception 'JUPR_WITHDRAW_IDENTITY: Invalid registration edit link.';
  end if;
  if pg_catalog.encode(pg_catalog.sha256(pg_catalog.convert_to(lower(btrim(v_registration.email)), 'UTF8')), 'hex') is distinct from p_email_hash then
    raise exception 'JUPR_WITHDRAW_IDENTITY: Invalid registration edit link.';
  end if;
  if v_registration.updated_at is distinct from p_expected_updated_at
     or jsonb_typeof(p_expected_selection_versions) is distinct from 'array' then
    raise exception 'JUPR_CANCEL_CONFLICT: Registration changed. Refresh and try again.';
  end if;
  if jsonb_array_length(p_expected_selection_versions) <> (
      select count(*) from public.tournament_registration_selections where registration_id = p_registration_id and tournament_id = p_tournament_id)
     or jsonb_array_length(p_expected_selection_versions) <> (
      select count(distinct e->>'id') from jsonb_array_elements(p_expected_selection_versions) e)
     or exists (select 1 from jsonb_array_elements(p_expected_selection_versions) e
       left join public.tournament_registration_selections s on s.id = e->>'id'
         and s.registration_id = p_registration_id and s.tournament_id = p_tournament_id
       where s.id is null or s.updated_at is distinct from (e->>'updated_at')::timestamptz) then
    raise exception 'JUPR_CANCEL_CONFLICT: Registration changed. Refresh and try again.';
  end if;
  perform public.server_cancel_tournament_registrations(p_tournament_id,
    jsonb_build_array(jsonb_build_object('id', p_registration_id, 'expected_updated_at', p_expected_updated_at)), v_actor);
  return jsonb_build_object('ok', true, 'withdrawn', true, 'registration_id', p_registration_id, 'idempotent', false);
end;
$$;
revoke all on function public.server_withdraw_public_tournament_registration(text,text,text,text,timestamptz,jsonb) from public, anon, authenticated;
grant execute on function public.server_withdraw_public_tournament_registration(text,text,text,text,timestamptz,jsonb) to service_role;
