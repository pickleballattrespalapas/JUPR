create extension if not exists "pgcrypto";

do $$
begin
    if not exists (select 1 from pg_type where typname = 'badge_state') then
        create type badge_state as enum ('live', 'frozen', 'deprecated');
    end if;
end $$;

create table if not exists badges (
    badge_id text primary key,
    name text not null,
    prestige integer not null,
    category text not null,
    is_stackable boolean not null default false,
    is_active boolean not null default true,
    state badge_state not null default 'live',
    state_changed_at timestamptz not null default now(),
    state_change_reason text,
    created_at timestamptz not null default now()
);

alter table badges
    add column if not exists state badge_state not null default 'live',
    add column if not exists state_changed_at timestamptz not null default now(),
    add column if not exists state_change_reason text,
    add column if not exists created_at timestamptz not null default now();

alter table badges
    alter column is_stackable set default false,
    alter column is_active set default true;

create table if not exists player_badges (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    player_id bigint not null,
    badge_id text not null references badges(badge_id) on update cascade on delete restrict,
    earned_at timestamptz not null default now(),
    context_type text not null default 'overall',
    context_id text,
    match_id text,
    value_num numeric,
    value_json jsonb
);

alter table player_badges
    alter column context_type set default 'overall';

alter table player_badges
    drop constraint if exists player_badges_unique_context,
    drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key,
    add constraint player_badges_unique_context unique (club_id, player_id, badge_id, context_type, context_id);

alter table player_badges
    drop constraint if exists player_badges_badge_id_fkey,
    add constraint player_badges_badge_id_fkey
        foreign key (badge_id) references badges(badge_id) on update cascade on delete restrict;

create index if not exists player_badges_club_player_idx on player_badges (club_id, player_id);
create index if not exists player_badges_club_badge_idx on player_badges (club_id, badge_id);
create index if not exists player_badges_club_earned_at_idx on player_badges (club_id, earned_at desc);

alter table badges enable row level security;
alter table player_badges enable row level security;

revoke all on badges from anon, authenticated;
revoke all on player_badges from anon, authenticated;

grant select on badges to anon, authenticated;

grant select (
    badge_id,
    player_id,
    earned_at,
    context_type,
    context_id,
    match_id,
    value_num,
    value_json
) on player_badges to anon, authenticated;

grant insert, update on player_badges to authenticated;

drop policy if exists public_select_badges on badges;
create policy public_select_badges
    on badges
    for select
    to public
    using (true);

drop policy if exists public_select_player_badges on player_badges;
create policy public_select_player_badges
    on player_badges
    for select
    to public
    using (
        club_id = coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'club_id', '')
    );

drop policy if exists club_insert_player_badges on player_badges;
create policy club_insert_player_badges
    on player_badges
    for insert
    to authenticated
    with check (
        club_id = coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'club_id', '')
    );

drop policy if exists club_update_player_badges on player_badges;
create policy club_update_player_badges
    on player_badges
    for update
    to authenticated
    using (
        club_id = coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'club_id', '')
    )
    with check (
        club_id = coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'club_id', '')
    );

insert into badges (badge_id, name, prestige, category, is_stackable, is_active, state)
values
    ('participant', 'Participant', 10, 'participation', false, true, 'live'),
    ('dedicated_participant_50', 'Dedicated Participant', 25, 'participation', false, true, 'live'),
    ('lifetime_participant_200', 'Lifetime Participant', 50, 'participation', false, true, 'live'),
    ('mountain_climber', 'Mountain Climber', 45, 'momentum', true, true, 'live'),
    ('breakthrough', 'Breakthrough', 55, 'momentum', false, true, 'live'),
    ('above_expectations', 'Above Expectations', 50, 'performance', true, true, 'live'),
    ('clutch_performer', 'Clutch Performer', 60, 'performance', true, true, 'live'),
    ('dominant_run', 'Dominant Run', 45, 'dominance', true, true, 'live'),
    ('high_output', 'High Output', 40, 'dominance', true, true, 'live'),
    ('battle_tested', 'Battle Tested', 50, 'quality', true, true, 'live'),
    ('consistency', 'Consistency', 60, 'quality', true, true, 'live'),
    ('giant_slayer', 'Giant Slayer', 75, 'rarity', true, true, 'live'),
    ('upset_champion', 'Upset Champion', 90, 'rarity', true, true, 'live')
on conflict (badge_id) do update set
    name = excluded.name,
    prestige = excluded.prestige,
    category = excluded.category,
    is_stackable = excluded.is_stackable,
    is_active = excluded.is_active,
    state = excluded.state;
