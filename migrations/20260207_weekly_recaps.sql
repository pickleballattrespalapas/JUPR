create table if not exists public.weekly_recaps (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    week_start date not null,
    week_end date not null,
    status text not null default 'draft',
    generated_json jsonb not null default '{}'::jsonb,
    edits_json jsonb not null default '{}'::jsonb,
    final_json jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    published_at timestamptz null,
    published_by text null,
    unique (club_id, week_start)
);

create index if not exists weekly_recaps_club_week_idx
    on public.weekly_recaps (club_id, week_start desc);

create index if not exists weekly_recaps_club_status_week_idx
    on public.weekly_recaps (club_id, status, week_start desc);

create or replace function public.set_updated_at_timestamp()
returns trigger as $$
begin
    new.updated_at = now();
    return new;
end;
$$ language plpgsql;

drop trigger if exists weekly_recaps_set_updated_at on public.weekly_recaps;
create trigger weekly_recaps_set_updated_at
before update on public.weekly_recaps
for each row
execute function public.set_updated_at_timestamp();
