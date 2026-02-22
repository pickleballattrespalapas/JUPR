alter table badges
    add column if not exists rarity text,
    add column if not exists tier integer,
    add column if not exists icon_key text,
    add column if not exists lore text not null default '',
    add column if not exists hint text not null default '',
    add column if not exists scope text not null default 'overall';

update badges
set
    rarity = coalesce(rarity, 'common'),
    lore = coalesce(lore, ''),
    hint = coalesce(hint, ''),
    scope = coalesce(scope, 'overall')
where rarity is null or lore is null or hint is null or scope is null;

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
    context_type text not null default 'overall',
    context_id text not null,
    match_id text,
    title text not null,
    body text not null,
    importance integer not null default 0,
    expires_at timestamptz,
    value_json jsonb
);

alter table player_stories
    drop constraint if exists player_stories_unique_context,
    add constraint player_stories_unique_context unique (club_id, player_id, story_type, context_id);

create index if not exists player_stories_club_player_idx on player_stories (club_id, player_id);
create index if not exists player_stories_club_created_idx on player_stories (club_id, created_at desc);

alter table player_stories enable row level security;

revoke all on player_stories from anon, authenticated;

grant select (
    id,
    player_id,
    created_at,
    story_type,
    context_type,
    context_id,
    match_id,
    title,
    body,
    importance,
    expires_at,
    value_json
) on player_stories to anon, authenticated;

grant insert, update, delete on player_stories to authenticated;

drop policy if exists public_select_player_stories on player_stories;
create policy public_select_player_stories
    on player_stories
    for select
    to public
    using (
        club_id = coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'club_id', '')
    );

drop policy if exists club_insert_player_stories on player_stories;
create policy club_insert_player_stories
    on player_stories
    for insert
    to authenticated
    with check (
        club_id = coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'club_id', '')
    );

drop policy if exists club_update_player_stories on player_stories;
create policy club_update_player_stories
    on player_stories
    for update
    to authenticated
    using (
        club_id = coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'club_id', '')
    )
    with check (
        club_id = coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'club_id', '')
    );
