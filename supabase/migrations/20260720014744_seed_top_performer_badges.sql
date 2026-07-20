-- Seed the four definitions required by the League Awards mint workflow.
--
-- Existing rows are deliberately left untouched. Operators may have frozen or
-- customized a definition, and presence (not copy text/state) is the runtime
-- foreign-key prerequisite for minting. Replaying this migration therefore
-- inserts only missing badge IDs and verifies that the complete set exists.
--
-- Some legacy databases also carry NOT NULL *_v2 compatibility columns without
-- a synchronization trigger. The guarded insert below copies each canonical
-- value into every corresponding compatibility column that is actually present.

do $badge_seed$
declare
    compatibility_column text;
    canonical_column text;
    compatibility_insert_columns text := '';
    compatibility_select_columns text := '';
    missing_badge_ids text[];
begin
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
                || format(', %I', compatibility_column);
            compatibility_select_columns := compatibility_select_columns
                || format(', canonical.%I', canonical_column);
        end if;
    end loop;

    execute
        $badge_seed_insert$
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
        $badge_seed_insert$
        || compatibility_insert_columns
        || $badge_seed_insert$
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
        $badge_seed_insert$
        || compatibility_select_columns
        || $badge_seed_insert$
        from (
            values
                (
                    'top_performer_highest_rating',
                    'Top Performer: Highest Rating',
                    130,
                    'Top Performer Awards',
                    true,
                    true,
                    'legendary',
                    null::integer,
                    'trophy',
                    'The league closes with your rating on the peak.',
                    'Finish the season with the highest mark.',
                    'league',
                    'live'::public.badge_state
                ),
                (
                    'top_performer_most_improved',
                    'Top Performer: Most Improved',
                    125,
                    'Top Performer Awards',
                    true,
                    true,
                    'legendary',
                    null::integer,
                    'trophy',
                    'The biggest climb shows up in the final tape.',
                    'Make the largest rating leap in the league.',
                    'league',
                    'live'::public.badge_state
                ),
                (
                    'top_performer_best_win_pct',
                    'Top Performer: Best Win %',
                    120,
                    'Top Performer Awards',
                    true,
                    true,
                    'legendary',
                    null::integer,
                    'trophy',
                    'The league’s cleanest record shines at the top.',
                    'Finish with the best win percentage.',
                    'league',
                    'live'::public.badge_state
                ),
                (
                    'top_performer_most_wins',
                    'Top Performer: Most Wins',
                    115,
                    'Top Performer Awards',
                    true,
                    true,
                    'legendary',
                    null::integer,
                    'trophy',
                    'No one stacks wins faster when the season closes.',
                    'Lead the league in total wins.',
                    'league',
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
        $badge_seed_insert$;

    select coalesce(array_agg(expected.badge_id order by expected.badge_id), array[]::text[])
    into missing_badge_ids
    from (
        values
            ('top_performer_highest_rating'),
            ('top_performer_most_improved'),
            ('top_performer_best_win_pct'),
            ('top_performer_most_wins')
    ) as expected(badge_id)
    left join public.badges as seeded using (badge_id)
    where seeded.badge_id is null;

    if cardinality(missing_badge_ids) > 0 then
        raise exception
            'League Awards badge seed incomplete; missing badge IDs: %',
            array_to_string(missing_badge_ids, ', ');
    end if;
end
$badge_seed$;
