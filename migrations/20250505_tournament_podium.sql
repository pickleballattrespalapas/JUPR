create table if not exists public.tournament_podium (
    id uuid primary key default gen_random_uuid(),
    tournament_id uuid not null references public.tournaments(id) on delete cascade,
    placement integer not null,
    team_id uuid not null references public.tournament_teams(id) on delete restrict,
    source text not null,
    created_at timestamptz not null default now(),
    constraint tournament_podium_unique_placement unique (tournament_id, placement),
    constraint tournament_podium_unique_team unique (tournament_id, team_id),
    constraint tournament_podium_valid_placement check (placement between 1 and 3)
);

create index if not exists idx_tournament_podium_tournament_id on public.tournament_podium (tournament_id);

insert into public.badges (badge_id, name, prestige, category, is_stackable, is_active, rarity, tier, icon_key, lore, hint, scope)
values
    (
        'tournament_champion',
        'Tournament Champion',
        160,
        'Tournament Podium',
        false,
        true,
        'legendary',
        null,
        'podium',
        'The bracket closes with your team on the top step.',
        'Win the tournament to claim gold.',
        'tournament'
    ),
    (
        'tournament_runner_up',
        'Tournament Runner-Up',
        140,
        'Tournament Podium',
        false,
        true,
        'legendary',
        null,
        'podium',
        'Second place still lands on the stage.',
        'Finish as the tournament runner-up.',
        'tournament'
    ),
    (
        'tournament_third_place',
        'Tournament Third Place',
        130,
        'Tournament Podium',
        false,
        true,
        'legendary',
        null,
        'podium',
        'Third place keeps your name on the podium.',
        'Earn the bronze finish.',
        'tournament'
    )
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
