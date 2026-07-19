-- Order-26 Tournament Admin durable mutation/reconciliation contract.
--
-- FastAPI authenticates and authorizes the Supabase JWT, then uses its
-- server-only service role for this coordination table. Next never receives a
-- service credential and cannot access these rows directly.

create table if not exists public.tournament_admin_operations (
    operation_key text primary key,
    request_fingerprint text not null,
    club_id text not null,
    surface text not null,
    action text not null,
    entity_type text not null,
    entity_id text not null,
    lock_scope text not null,
    expected_state text not null,
    status text not null default 'intent',
    request_json jsonb not null default '{}'::jsonb,
    result_json jsonb not null default '{}'::jsonb,
    error_text text,
    attempt_count integer not null default 1,
    created_by text not null,
    updated_by text not null,
    completion_audited_at timestamptz,
    created_at timestamptz not null default timezone('utc', now()),
    updated_at timestamptz not null default timezone('utc', now()),
    constraint tournament_admin_operation_key_length check (char_length(operation_key) = 64),
    constraint tournament_admin_request_fingerprint_length check (char_length(request_fingerprint) = 64),
    constraint tournament_admin_lock_scope_present check (char_length(trim(lock_scope)) > 0),
    constraint tournament_admin_surface_check check (surface in ('tournament', 'setup', 'registration', 'import_handoff')),
    constraint tournament_admin_status_check check (status in ('intent', 'mutated', 'completed', 'failed', 'recovery_required')),
    constraint tournament_admin_attempt_count_check check (attempt_count >= 1),
    constraint tournament_admin_operation_fingerprint_unique unique (club_id, request_fingerprint)
);

create index if not exists idx_tournament_admin_operations_entity_updated
    on public.tournament_admin_operations (club_id, entity_type, entity_id, updated_at desc);

create index if not exists idx_tournament_admin_operations_recovery
    on public.tournament_admin_operations (club_id, status, updated_at desc)
    where status <> 'completed';

create unique index if not exists idx_tournament_admin_operations_active_lock
    on public.tournament_admin_operations (club_id, lock_scope)
    where status in ('intent', 'mutated', 'recovery_required');

alter table public.tournament_admin_operations enable row level security;
revoke all on table public.tournament_admin_operations from public, anon, authenticated;
grant select, insert, update, delete on table public.tournament_admin_operations to service_role;

comment on table public.tournament_admin_operations is
    'FastAPI-private Tournament Admin mutation intents, results, and response-loss recovery state.';

create or replace function public.admin_delete_empty_tournament_draft_cas(
    p_club_id text,
    p_tournament_id text,
    p_expected_updated_at timestamptz
)
returns jsonb
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_tournament public.tournaments%rowtype;
    v_registrations integer;
    v_selections integer;
    v_draws integer;
    v_teams integer;
    v_games integer;
    v_podium integer;
begin
    select *
     into v_tournament
      from public.tournaments
     where id::text = p_tournament_id
       and club_id = p_club_id
       and updated_at = p_expected_updated_at
     for update;

    if not found then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_STALE';
    end if;
    if upper(coalesce(v_tournament.status, '')) <> 'DRAFT' then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_NOT_DRAFT';
    end if;

    select count(*) into v_registrations from public.tournament_registrations where tournament_id::text = p_tournament_id;
    select count(*) into v_selections from public.tournament_registration_selections where tournament_id::text = p_tournament_id;
    select count(*) into v_draws from public.tournament_event_draws where tournament_id::text = p_tournament_id;
    select count(*) into v_teams from public.tournament_teams where tournament_id::text = p_tournament_id;
    select count(*) into v_games from public.tournament_games where tournament_id::text = p_tournament_id;
    select count(*) into v_podium from public.tournament_podium where tournament_id::text = p_tournament_id;

    if v_registrations > 0 or v_selections > 0 or v_draws > 0 or v_teams > 0 or v_games > 0 or v_podium > 0 then
        raise exception using errcode = 'P0001', message = 'JUPR_TOURNAMENT_NOT_EMPTY';
    end if;

    delete from public.tournament_registration_settings where tournament_id::text = p_tournament_id;
    delete from public.tournament_event_options where tournament_id::text = p_tournament_id;
    delete from public.tournament_registration_days where tournament_id::text = p_tournament_id;
    delete from public.tournaments where id::text = p_tournament_id and club_id = p_club_id;

    return jsonb_build_object(
        'ok', true,
        'tournament_id', p_tournament_id,
        'usage_summary', jsonb_build_object(
            'registrations', v_registrations,
            'registration_selections', v_selections,
            'event_draws', v_draws,
            'teams', v_teams,
            'games', v_games,
            'podium', v_podium
        )
    );
end;
$$;

revoke all on function public.admin_delete_empty_tournament_draft_cas(text, text, timestamptz) from public, anon, authenticated;
grant execute on function public.admin_delete_empty_tournament_draft_cas(text, text, timestamptz) to service_role;

notify pgrst, 'reload schema';
