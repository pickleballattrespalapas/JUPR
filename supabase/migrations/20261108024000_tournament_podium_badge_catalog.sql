-- Seed the three catalog definitions required by Tournament Live podium awards.
--
-- The original definitions lived only in the legacy migration archive, while
-- the active Supabase migration chain installed the atomic award RPC that
-- references them. Existing rows are preserved; presence is the foreign-key
-- prerequisite for minting the complete podium badge set.

do $podium_badge_seed$
declare
    compatibility_column text;
    canonical_column text;
    compatibility_insert_columns text := '';
    compatibility_select_columns text := '';
    missing_badge_ids text[];
begin
    if pg_catalog.to_regclass('public.badges') is null then
        raise exception 'Tournament podium badge seed requires public.badges.';
    end if;
    if pg_catalog.to_regtype('public.badge_state') is null then
        raise exception 'Tournament podium badge seed requires public.badge_state.';
    end if;

    for compatibility_column, canonical_column in
        select mapping.compatibility_column, mapping.canonical_column
        from (
            values
                ('name_v2', 'name'),
                ('prestige_v2', 'prestige'),
                ('category_v2', 'category'),
                ('is_active_v2', 'is_active'),
                ('is_stackable_v2', 'is_stackable'),
                ('lore_v2', 'lore'),
                ('hint_v2', 'hint'),
                ('scope_v2', 'scope')
        ) as mapping(compatibility_column, canonical_column)
    loop
        if exists (
            select 1
            from pg_catalog.pg_attribute as attribute
            where attribute.attrelid = 'public.badges'::regclass
              and attribute.attname = compatibility_column
              and attribute.attnum > 0
              and not attribute.attisdropped
        ) then
            compatibility_insert_columns := compatibility_insert_columns
                || pg_catalog.format(', %I', compatibility_column);
            compatibility_select_columns := compatibility_select_columns
                || pg_catalog.format(', canonical.%I', canonical_column);
        end if;
    end loop;

    execute
        $podium_badge_insert$
        insert into public.badges (
            badge_id,
            name,
            prestige,
            category,
            is_stackable,
            is_active,
            rarity,
            tier,
            icon_key,
            lore,
            hint,
            scope,
            state
        $podium_badge_insert$
        || compatibility_insert_columns
        || $podium_badge_insert$
        )
        select
            canonical.badge_id,
            canonical.name,
            canonical.prestige,
            canonical.category,
            canonical.is_stackable,
            canonical.is_active,
            canonical.rarity,
            canonical.tier,
            canonical.icon_key,
            canonical.lore,
            canonical.hint,
            canonical.scope,
            canonical.state
        $podium_badge_insert$
        || compatibility_select_columns
        || $podium_badge_insert$
        from (
            values
                (
                    'tournament_champion',
                    'Tournament Champion',
                    160,
                    'Tournament Podium',
                    false,
                    true,
                    'legendary',
                    null::integer,
                    'podium',
                    'The bracket closes with your team on the top step.',
                    'Win the tournament to claim gold.',
                    'tournament',
                    'live'::public.badge_state
                ),
                (
                    'tournament_runner_up',
                    'Tournament Runner-Up',
                    140,
                    'Tournament Podium',
                    false,
                    true,
                    'legendary',
                    null::integer,
                    'podium',
                    'Second place still lands on the stage.',
                    'Finish as the tournament runner-up.',
                    'tournament',
                    'live'::public.badge_state
                ),
                (
                    'tournament_third_place',
                    'Tournament Third Place',
                    130,
                    'Tournament Podium',
                    false,
                    true,
                    'legendary',
                    null::integer,
                    'podium',
                    'Third place keeps your name on the podium.',
                    'Earn the bronze finish.',
                    'tournament',
                    'live'::public.badge_state
                )
        ) as canonical(
            badge_id,
            name,
            prestige,
            category,
            is_stackable,
            is_active,
            rarity,
            tier,
            icon_key,
            lore,
            hint,
            scope,
            state
        )
        on conflict (badge_id) do nothing;
        $podium_badge_insert$;

    select coalesce(
        pg_catalog.array_agg(expected.badge_id order by expected.badge_id),
        array[]::text[]
    )
    into missing_badge_ids
    from (
        values
            ('tournament_champion'),
            ('tournament_runner_up'),
            ('tournament_third_place')
    ) as expected(badge_id)
    left join public.badges as seeded using (badge_id)
    where seeded.badge_id is null;

    if pg_catalog.cardinality(missing_badge_ids) > 0 then
        raise exception
            'Tournament podium badge seed incomplete; missing badge IDs: %',
            pg_catalog.array_to_string(missing_badge_ids, ', ');
    end if;
end;
$podium_badge_seed$;

comment on table public.badges is
    'Canonical badge catalog, including the definitions required by guarded tournament podium awards.';
