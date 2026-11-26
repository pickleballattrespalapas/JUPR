begin;

create table public.pcs_managed_interclub_club_entries (
    season_id uuid not null references public.pcs_managed_interclub_seasons(id) on delete cascade,
    club_id text not null,
    token_hash text not null unique check (token_hash ~ '^[a-f0-9]{64}$'),
    registration jsonb check (jsonb_typeof(registration) = 'object'),
    revision integer not null default 0 check (revision >= 0),
    primary key (season_id, club_id)
);
alter table public.pcs_managed_interclub_club_entries enable row level security;
revoke all on public.pcs_managed_interclub_club_entries from public, anon, authenticated;
grant select, insert, update, delete on public.pcs_managed_interclub_club_entries to service_role;

alter table public.pcs_managed_interclub_audit drop constraint pcs_managed_interclub_audit_actor_kind_check;
alter table public.pcs_managed_interclub_audit add check (actor_kind in ('organizer', 'host', 'captain'));

-- Managed clubs live inside the season document. Removing a club must destroy its
-- credential at the same time, including when an organizer later reuses its ID.
create function public.pcs_managed_interclub_remove_orphan_entries()
returns trigger language plpgsql security invoker set search_path = public as $$
begin
    delete from public.pcs_managed_interclub_club_entries e
    where e.season_id = new.id and not exists (
        select from jsonb_array_elements(new.details->'clubs') c where c->>'id' = e.club_id
    );
    return new;
end $$;
revoke all on function public.pcs_managed_interclub_remove_orphan_entries() from public, anon, authenticated;
create trigger pcs_managed_interclub_prune_club_entries after update of details on public.pcs_managed_interclub_seasons
for each row when (old.details->'clubs' is distinct from new.details->'clubs')
execute function public.pcs_managed_interclub_remove_orphan_entries();

create function public.pcs_managed_interclub_club_link(
    p_actor_id uuid, p_actor_email text, p_organizer_club_id text,
    p_season_id uuid, p_entry_club_id text, p_token_hash text
) returns integer language plpgsql security invoker set search_path = public as $$
declare
    s public.pcs_managed_interclub_seasons;
    entry_revision integer;
begin
    if p_actor_id is null or not exists (
        select from public.admin_role_assignments
        where club_id = p_organizer_club_id and email = lower(trim(p_actor_email))
          and (user_id is null or user_id = p_actor_id)
          and role in ('super_admin','administrator','club_owner') and revoked_at is null
          and (expires_at is null or expires_at > now())
    ) then
        raise exception 'Organizer access required' using errcode = '42501';
    end if;

    -- All entry operations acquire season then entry locks in that order.
    select * into s from public.pcs_managed_interclub_seasons
    where id = p_season_id and organizer_club_id = p_organizer_club_id for share;
    if not found or not exists (
        select from jsonb_array_elements(s.details->'clubs') c where c->>'id' = p_entry_club_id
    ) then
        raise exception 'Club unavailable' using errcode = 'P0002';
    end if;
    insert into public.pcs_managed_interclub_club_entries(season_id, club_id, token_hash)
    values (s.id, p_entry_club_id, p_token_hash)
    on conflict (season_id, club_id) do update set token_hash = excluded.token_hash
    returning revision into entry_revision;
    insert into public.pcs_managed_interclub_audit(season_id, actor_id, actor_kind, action, revision, after_data)
    values (s.id, p_actor_id, 'organizer', 'club_entry_link', entry_revision, jsonb_build_object('club_id',p_entry_club_id));
    return entry_revision;
end $$;
revoke all on function public.pcs_managed_interclub_club_link(uuid,text,text,uuid,text,text) from public,anon,authenticated;
grant execute on function public.pcs_managed_interclub_club_link(uuid,text,text,uuid,text,text) to service_role;

-- The API passes a digest, never the bearer token. Read and save share a
-- single-club projection. Each save rechecks the season and token under lock.
create function public.pcs_managed_interclub_club_entry(
    p_token_hash text, p_expected_revision integer default null, p_registration jsonb default null
) returns jsonb language plpgsql security invoker set search_path = public as $$
declare
    s public.pcs_managed_interclub_seasons;
    e public.pcs_managed_interclub_club_entries;
    club_name text;
begin
    select season.* into s from public.pcs_managed_interclub_seasons season
    join public.pcs_managed_interclub_club_entries entry on entry.season_id = season.id
    where entry.token_hash = p_token_hash for share of season;
    select * into e from public.pcs_managed_interclub_club_entries
    where season_id = s.id and token_hash = p_token_hash for update;
    select c->>'name' into club_name from jsonb_array_elements(s.details->'clubs') c where c->>'id' = e.club_id;
    if club_name is null then
        raise exception 'Club link unavailable' using errcode = 'P0002';
    end if;

    if p_registration is not null then
        if e.revision is distinct from p_expected_revision then
            raise exception 'Club entry changed' using errcode = 'PT409';
        end if;
        -- These season-dependent conditions must use the current season, even
        -- when an organizer edits divisions or dates while a captain is typing.
        if jsonb_typeof(p_registration->'divisions') is distinct from 'array'
            or jsonb_array_length(p_registration->'divisions') < 1
            or not (s.details->'divisions' @> (p_registration->'divisions')) then
            raise exception 'Select offered divisions' using errcode = '22023';
        end if;
        if exists (
            select from jsonb_array_elements(p_registration->'host_offers') offer
            where (offer->>'date')::date < (s.details->>'start_date')::date
               or (offer->>'date')::date > (s.details->>'end_date')::date
        ) then
            raise exception 'Hosting date outside season' using errcode = '22023';
        end if;
        update public.pcs_managed_interclub_club_entries
        set registration = p_registration, revision = revision + 1
        where season_id = e.season_id and club_id = e.club_id returning * into e;
        insert into public.pcs_managed_interclub_audit(season_id, actor_kind, action, revision, after_data)
        values (s.id, 'captain', 'club_entry_save', e.revision, jsonb_build_object('club_id',e.club_id));
    end if;

    return jsonb_build_object(
        'season_name', s.details->>'name', 'club_name', club_name, 'divisions', s.details->'divisions',
        'start_date', s.details->'start_date', 'end_date', s.details->'end_date',
        'revision', e.revision, 'registration', e.registration
    );
end $$;
revoke all on function public.pcs_managed_interclub_club_entry(text,integer,jsonb) from public,anon,authenticated;
grant execute on function public.pcs_managed_interclub_club_entry(text,integer,jsonb) to service_role;
commit;
