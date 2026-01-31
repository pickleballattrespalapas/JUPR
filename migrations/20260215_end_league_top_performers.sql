alter table if exists public.leagues_metadata
    add column if not exists ended_at timestamptz,
    add column if not exists ended_by text,
    add column if not exists status text,
    add column if not exists end_awards jsonb;

create index if not exists leagues_metadata_ended_at_idx
    on public.leagues_metadata (ended_at);

create index if not exists leagues_metadata_status_idx
    on public.leagues_metadata (status);

insert into public.badges (badge_id, name, prestige, category, is_stackable, is_active, rarity, tier, icon_key, lore, hint, scope)
values
    ('top_performer_highest_rating', 'Top Performer: Highest Rating', 130, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The league closes with your rating on the peak.', 'Finish the season with the highest mark.', 'league'),
    ('top_performer_most_improved', 'Top Performer: Most Improved', 125, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The biggest climb shows up in the final tape.', 'Make the largest rating leap in the league.', 'league'),
    ('top_performer_best_win_pct', 'Top Performer: Best Win %', 120, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The league’s cleanest record shines at the top.', 'Finish with the best win percentage.', 'league'),
    ('top_performer_most_wins', 'Top Performer: Most Wins', 115, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'No one stacks wins faster when the season closes.', 'Lead the league in total wins.', 'league')
on conflict (badge_id) do update set
    name = excluded.name,
    prestige = excluded.prestige,
    category = excluded.category,
    is_stackable = excluded.is_stackable,
    is_active = excluded.is_active,
    rarity = excluded.rarity,
    tier = excluded.tier,
    icon_key = excluded.icon_key,
    lore = excluded.lore,
    hint = excluded.hint,
    scope = excluded.scope;
