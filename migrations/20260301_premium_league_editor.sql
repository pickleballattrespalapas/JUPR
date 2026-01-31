alter table if exists public.leagues_metadata
    add column if not exists status text,
    add column if not exists started_at timestamptz,
    add column if not exists schedule_config jsonb,
    add column if not exists court_board_defaults jsonb,
    add column if not exists rules_config jsonb,
    add column if not exists awards_config jsonb;

alter table if exists public.leagues_metadata
    alter column status set default 'draft';

update public.leagues_metadata
set status = case
    when ended_at is not null then 'ended'
    when is_active is true then 'active'
    else 'draft'
end
where status is null;

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

do $$
begin
    if not exists (
        select 1
        from pg_constraint
        where conname = 'player_badges_unique_context'
    ) then
        alter table public.player_badges
            add constraint player_badges_unique_context unique (club_id, player_id, badge_id, context_id);
    end if;
end $$;
