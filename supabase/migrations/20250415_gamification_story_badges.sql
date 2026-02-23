alter table badges
    add column if not exists rarity text,
    add column if not exists tier integer,
    add column if not exists icon_key text,
    add column if not exists lore text not null default '',
    add column if not exists hint text not null default '',
    add column if not exists scope text not null default 'overall';

alter table badges
    alter column rarity set default 'common',
    alter column lore set default '',
    alter column hint set default '',
    alter column scope set default 'overall';

update badges
set
    rarity = coalesce(rarity, 'common'),
    lore = coalesce(lore, ''),
    hint = coalesce(hint, ''),
    scope = coalesce(scope, 'overall')
where rarity is null or lore is null or hint is null or scope is null;

alter table player_badges
    alter column value_json type jsonb using value_json::jsonb,
    alter column value_json set default '{}'::jsonb;

alter table player_badges
    drop constraint if exists player_badges_unique_context,
    drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key,
    add constraint player_badges_unique_context unique (club_id, player_id, badge_id, context_id);

create table if not exists player_stories (
    id uuid primary key default gen_random_uuid(),
    club_id text not null,
    player_id bigint not null,
    created_at timestamptz not null default now(),
    story_type text not null,
    context_type text,
    context_id text,
    match_id text,
    title text not null,
    body text not null,
    importance integer not null default 0,
    expires_at timestamptz,
    value_json jsonb not null default '{}'::jsonb
);

update player_stories
set value_json = '{}'::jsonb
where value_json is null;

alter table player_stories
    alter column context_type drop not null,
    alter column context_id drop not null,
    alter column value_json set default '{}'::jsonb,
    alter column value_json set not null;

alter table player_stories
    drop constraint if exists player_stories_unique_context,
    add constraint player_stories_unique_context unique (club_id, player_id, story_type, context_id);
