create extension if not exists "pgcrypto";

create table if not exists badges (
    badge_id text primary key,
    name text not null,
    prestige integer not null,
    category text not null,
    is_stackable boolean not null default false,
    is_active boolean not null default true
);

create table if not exists player_badges (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    player_id bigint not null,
    badge_id text not null references badges(badge_id),
    earned_at timestamptz not null default now(),
    context_type text not null,
    context_id text,
    match_id text,
    value_num numeric,
    value_json jsonb,
    unique (club_id, player_id, badge_id, context_id)
);

create index if not exists player_badges_club_player_idx on player_badges (club_id, player_id);
create index if not exists player_badges_badge_idx on player_badges (badge_id);

insert into badges (badge_id, name, prestige, category, is_stackable, is_active)
values
    ('participant', 'Participant', 10, 'Participation', false, true),
    ('dedicated_participant', 'Dedicated Participant', 25, 'Participation', false, true),
    ('lifetime_participant', 'Lifetime Participant', 50, 'Participation', false, true),
    ('mountain_climber', 'Mountain Climber', 45, 'Momentum & Progress', true, true),
    ('breakthrough', 'Breakthrough', 55, 'Momentum & Progress', false, true),
    ('above_expectations', 'Above Expectations', 50, 'Performance vs Expectation', false, true),
    ('clutch_performer', 'Clutch Performer', 60, 'Performance vs Expectation', false, true),
    ('dominant_run', 'Dominant Run', 45, 'Dominance & Quality', false, true),
    ('high_output', 'High Output', 40, 'Dominance & Quality', false, true),
    ('battle_tested', 'Battle Tested', 50, 'Dominance & Quality', false, true),
    ('consistency', 'Consistency', 60, 'Dominance & Quality', false, true),
    ('giant_slayer', 'Giant Slayer', 75, 'Prestige / Rarity', true, true),
    ('upset_champion', 'Upset Champion', 90, 'Prestige / Rarity', true, true)
on conflict (badge_id) do update set
    name = excluded.name,
    prestige = excluded.prestige,
    category = excluded.category,
    is_stackable = excluded.is_stackable,
    is_active = excluded.is_active;
