create or replace function public.set_updated_at_timestamp()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

create table if not exists public.club_people (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    display_name text not null,
    normalized_name text not null,
    linked_player_id bigint null,
    source text not null default 'social',
    first_seen_on date null,
    last_seen_on date null,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists club_people_club_normalized_name_idx
    on public.club_people (club_id, normalized_name);

create unique index if not exists club_people_club_linked_player_uidx
    on public.club_people (club_id, linked_player_id)
    where linked_player_id is not null;

create table if not exists public.live_events (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    source_event_uid text not null,
    name text not null,
    event_type text not null,
    result_mode text not null,
    event_date date not null,
    status text not null default 'saved',
    raw_event_json jsonb not null default '{}'::jsonb,
    summary_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    unique (club_id, source_event_uid)
);

create table if not exists public.live_event_participants (
    id uuid primary key default gen_random_uuid(),
    event_id uuid not null references public.live_events(id) on delete cascade,
    participant_key text not null,
    club_person_id uuid not null references public.club_people(id),
    linked_player_id bigint null,
    display_name_snapshot text not null,
    seed integer null,
    created_at timestamptz not null default now(),
    unique (event_id, participant_key)
);

create table if not exists public.live_event_matches (
    id uuid primary key default gen_random_uuid(),
    event_id uuid not null references public.live_events(id) on delete cascade,
    match_key text not null,
    played_on date not null,
    round_number integer null,
    court_number integer null,
    mini_round_number integer null,
    t1_p1_participant_id uuid not null references public.live_event_participants(id),
    t1_p2_participant_id uuid not null references public.live_event_participants(id),
    t2_p1_participant_id uuid not null references public.live_event_participants(id),
    t2_p2_participant_id uuid not null references public.live_event_participants(id),
    score_t1 integer not null,
    score_t2 integer not null,
    created_at timestamptz not null default now(),
    unique (event_id, match_key)
);

create index if not exists live_events_club_event_date_idx
    on public.live_events (club_id, event_date desc);

create index if not exists live_event_participants_event_idx
    on public.live_event_participants (event_id);

create index if not exists live_event_matches_event_idx
    on public.live_event_matches (event_id);

drop trigger if exists club_people_set_updated_at on public.club_people;
create trigger club_people_set_updated_at
before update on public.club_people
for each row
execute function public.set_updated_at_timestamp();

drop trigger if exists live_events_set_updated_at on public.live_events;
create trigger live_events_set_updated_at
before update on public.live_events
for each row
execute function public.set_updated_at_timestamp();
