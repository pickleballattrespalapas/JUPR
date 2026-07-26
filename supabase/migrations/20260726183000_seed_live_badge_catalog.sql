-- Seed the live engine badge definitions required by guarded match exclusion.
--
-- The legacy migration archive contains older copies of some definitions.
-- These rows are sourced from the current Python BADGE_DEFINITIONS contract.
-- Existing presentation fields are deliberately preserved. The three
-- operational eligibility fields required by the engine are repaired on an
-- existing row: live state, active status, and the match_updated trigger.

alter table public.badges
    add column if not exists eval_triggers jsonb
        not null
        default '["match_recorded","match_updated"]'::jsonb;

update public.badges
set eval_triggers = '["match_recorded","match_updated"]'::jsonb
where eval_triggers is null;

alter table public.badges
    alter column eval_triggers
        set default '["match_recorded","match_updated"]'::jsonb,
    alter column eval_triggers set not null;

do $live_badge_seed$
declare
    compatibility_column text;
    canonical_column text;
    compatibility_insert_columns text := '';
    compatibility_select_columns text := '';
    missing_badge_ids text[];
    ineligible_badge_ids text[];
begin
    if to_regtype('public.badge_state') is null then
        raise exception
            'Live badge seed requires the public.badge_state type.';
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
                || format(', %I', compatibility_column);
            compatibility_select_columns := compatibility_select_columns
                || format(', canonical.%I', canonical_column);
        end if;
    end loop;

    execute
        $live_badge_seed_insert$
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
            state,
            eval_triggers
        $live_badge_seed_insert$
        || compatibility_insert_columns
        || $live_badge_seed_insert$
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
            canonical.state,
            canonical.eval_triggers
        $live_badge_seed_insert$
        || compatibility_select_columns
        || $live_badge_seed_insert$
        from (
            values
                ('blowout_artist', 'Blowout Artist', 45, 'Dominance & Quality', true, true, 'rare', null::integer, 'blowout_artist', 'The gap grew and never closed.', 'The margin kept widening.', 'match', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('bounce_back', 'Bounce Back', 25, 'Skill Growth & Momentum', true, true, 'common', null::integer, 'bounce_back', 'A stumble, then a reply. The echo lands clean.', 'The next frame told a different story.', 'match', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('clean_sweep_week', 'Clean Sweep Week', 55, 'Dominance & Quality', true, true, 'epic', null::integer, 'clean_sweep_week', 'Every clip in the week ended the same way.', 'A week with no counterpunches.', 'week', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('david_vs_goliath', 'David vs Goliath', 95, 'Prestige / Rarity', true, true, 'legendary', null::integer, 'david_vs_goliath', 'The mismatch didn’t stay a mismatch.', 'The odds were heavy on one side.', 'match', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('dedicated_participant_50', 'Dedicated Participant', 25, 'Participation & Habit Loop', false, true, 'rare', null::integer, 'dedicated_participant_50', 'Fifty matches is where routine becomes identity.', 'The schedule is starting to recognize you.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('draft_master', 'Draft Master', 55, 'Versatility & Social Graph', true, true, 'rare', null::integer, 'draft_master', 'Different pairings, same result. The tape shows the range.', 'This month keeps swapping jerseys.', 'week', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('first_win', 'First Win', 15, 'Participation & Habit Loop', false, true, 'common', null::integer, 'first_win', 'The first mark on the ledger. The tape starts rolling for real.', 'There is always a first frame on the reel.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('giant_slayer', 'Giant Slayer', 90, 'Prestige / Rarity', true, true, 'legendary', null::integer, 'giant_slayer', 'A giant hit the floor and the camera never blinked.', 'A higher shadow fell.', 'match', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('hall_of_fame_night', 'Hall of Fame Night', 100, 'Meta / Prestige', true, true, 'legendary', null::integer, 'hall_of_fame_night', 'A night that pulled the cameras in closer.', 'Some nights feel larger than the rest.', 'match', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('high_roller', 'High Roller', 40, 'Dominance & Quality', true, true, 'rare', null::integer, 'high_roller', 'The pace stayed high and the points kept pouring.', 'The scoreboard got a workout.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('hot_streak', 'Hot Streak', 50, 'Skill Growth & Momentum', true, true, 'epic', null::integer, 'hot_streak', 'Wins blur together on the tape. The run keeps rolling.', 'The film strip barely cools off.', 'league', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('ice_in_veins', 'Ice in Veins', 70, 'Clutch & Pressure', false, true, 'epic', null::integer, 'ice_in_veins', 'When the margin tightened, the answer didn’t.', 'Cold hands don’t shake the camera.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('iron_week', 'Iron Week', 30, 'Participation & Habit Loop', true, true, 'common', null::integer, 'iron_week', 'A week packed tight with tape. The grind leaves a signature.', 'Some weeks barely have room to breathe.', 'week', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('legendary_upset', 'Legendary Upset', 120, 'Prestige / Rarity', true, true, 'legendary', null::integer, 'legendary_upset', 'The tape caught a moment nobody predicted.', 'The odds looked impossible on this frame.', 'match', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('level_up', 'Level Up', 40, 'Skill Growth & Momentum', true, true, 'rare', null::integer, 'level_up', 'Another rung claimed. The league notices the climb.', 'A new number appears on the nameplate.', 'league', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('lifetime_participant_200', 'Lifetime Participant', 50, 'Participation & Habit Loop', false, true, 'epic', null::integer, 'lifetime_participant_200', 'Two hundred matches doesn’t happen by accident. It happens by refusing to disappear.', 'The archive is getting heavy.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('marathon_month', 'Marathon Month', 60, 'Participation & Habit Loop', true, true, 'rare', null::integer, 'marathon_month', 'A month that never slowed down. Every day kept its own clip.', 'The month is still writing.', 'month', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('most_improved_monthly', 'Most Improved', 55, 'Skill Growth & Momentum', true, true, 'epic', null::integer, 'most_improved', 'A month where the climb couldn’t be ignored.', 'One month tilted harder than the rest.', 'month', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('mountain_climber', 'Mountain Climber', 45, 'Skill Growth & Momentum', true, true, 'rare', null::integer, 'mountain_climber', 'Ranks flipped. The ascent left landmarks.', 'The ladder looks different now.', 'league', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('network_builder', 'Network Builder', 70, 'Versatility & Social Graph', false, true, 'epic', null::integer, 'network_builder', 'The web stretches across the club.', 'The partner map keeps expanding.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('participant', 'Participant', 10, 'Participation & Habit Loop', false, true, 'common', null::integer, 'participant', 'Every career starts the same way: a name on the sheet and a moment on tape.', 'The archive doesn’t recognize potential. Only appearances.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('pickle_perfection', 'Pickle Perfection', 75, 'Dominance & Quality', true, true, 'legendary', null::integer, 'pickle_perfection', 'A shutout with no extra narration needed.', 'Sometimes the other side never shows up on the scoreboard.', 'match', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('rocket_start', 'Rocket Start', 50, 'Skill Growth & Momentum', false, true, 'rare', null::integer, 'rocket_start', 'The opening run shook the scoreboard. The room leaned in early.', 'The first stretch left a streak on the floor.', 'league', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('social_butterfly', 'Social Butterfly', 45, 'Versatility & Social Graph', false, true, 'rare', null::integer, 'social_butterfly', 'The partner list turned into a montage.', 'So many different pairings on the same reel.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('steady_hand', 'Steady Hand', 55, 'Consistency & Reliability', false, true, 'rare', null::integer, 'steady_hand', 'The season stayed smooth, even when the lights changed.', 'The season never wandered far.', 'season', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('swiss_army_knife', 'Swiss Army Knife', 65, 'Versatility & Social Graph', false, true, 'epic', null::integer, 'swiss_army_knife', 'Versatility on the record. Different stages, same sharp edge.', 'The season shows more than one role.', 'season', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('untouchable', 'Untouchable', 85, 'Dominance & Quality', true, true, 'legendary', null::integer, 'untouchable', 'A run that didn’t flinch. The tape shows no breaks.', 'The run feels unbroken.', 'overall', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('upset_champion', 'Upset Champion', 100, 'Prestige / Rarity', true, true, 'legendary', null::integer, 'upset_champion', 'The month’s biggest swing stayed on the reel.', 'One month holds the loudest turn.', 'month', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb),
                ('weekly_regular', 'Weekly Regular', 35, 'Participation & Habit Loop', false, true, 'rare', null::integer, 'weekly_regular', 'The schedule starts to recognize the face. The weeks keep stacking.', 'The calendar keeps calling.', 'league', 'live'::public.badge_state, '["match_recorded","match_updated"]'::jsonb)
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
            state,
            eval_triggers
        )
        on conflict (badge_id) do update
        set state = excluded.state,
            is_active = excluded.is_active,
            eval_triggers = case
            when pg_catalog.jsonb_typeof(badges.eval_triggers) = 'array'
             and badges.eval_triggers @> '["match_updated"]'::jsonb
                then badges.eval_triggers
            when pg_catalog.jsonb_typeof(badges.eval_triggers) = 'array'
                then badges.eval_triggers || '["match_updated"]'::jsonb
            else excluded.eval_triggers
        end
        where badges.state is distinct from excluded.state
           or badges.is_active is distinct from excluded.is_active
           or pg_catalog.jsonb_typeof(badges.eval_triggers) is distinct from 'array'
           or not badges.eval_triggers @> '["match_updated"]'::jsonb;
        $live_badge_seed_insert$;

    select coalesce(
        pg_catalog.array_agg(expected.badge_id order by expected.badge_id),
        array[]::text[]
    )
    into missing_badge_ids
    from (
        values
            ('blowout_artist'),
            ('bounce_back'),
            ('clean_sweep_week'),
            ('david_vs_goliath'),
            ('dedicated_participant_50'),
            ('draft_master'),
            ('first_win'),
            ('giant_slayer'),
            ('hall_of_fame_night'),
            ('high_roller'),
            ('hot_streak'),
            ('ice_in_veins'),
            ('iron_week'),
            ('legendary_upset'),
            ('level_up'),
            ('lifetime_participant_200'),
            ('marathon_month'),
            ('most_improved_monthly'),
            ('mountain_climber'),
            ('network_builder'),
            ('participant'),
            ('pickle_perfection'),
            ('rocket_start'),
            ('social_butterfly'),
            ('steady_hand'),
            ('swiss_army_knife'),
            ('untouchable'),
            ('upset_champion'),
            ('weekly_regular')
    ) as expected(badge_id)
    left join public.badges as seeded using (badge_id)
    where seeded.badge_id is null;

    if pg_catalog.cardinality(missing_badge_ids) > 0 then
        raise exception
            'Live badge seed incomplete; missing badge IDs: %',
            pg_catalog.array_to_string(missing_badge_ids, ', ');
    end if;

    select coalesce(
        pg_catalog.array_agg(expected.badge_id order by expected.badge_id),
        array[]::text[]
    )
    into ineligible_badge_ids
    from (
        values
            ('blowout_artist'),
            ('bounce_back'),
            ('clean_sweep_week'),
            ('david_vs_goliath'),
            ('dedicated_participant_50'),
            ('draft_master'),
            ('first_win'),
            ('giant_slayer'),
            ('hall_of_fame_night'),
            ('high_roller'),
            ('hot_streak'),
            ('ice_in_veins'),
            ('iron_week'),
            ('legendary_upset'),
            ('level_up'),
            ('lifetime_participant_200'),
            ('marathon_month'),
            ('most_improved_monthly'),
            ('mountain_climber'),
            ('network_builder'),
            ('participant'),
            ('pickle_perfection'),
            ('rocket_start'),
            ('social_butterfly'),
            ('steady_hand'),
            ('swiss_army_knife'),
            ('untouchable'),
            ('upset_champion'),
            ('weekly_regular')
    ) as expected(badge_id)
    join public.badges as seeded using (badge_id)
    where pg_catalog.lower(pg_catalog.btrim(coalesce(seeded.state::text, ''))) <> 'live'
       or seeded.is_active is distinct from true
       or pg_catalog.jsonb_typeof(seeded.eval_triggers) is distinct from 'array'
       or not seeded.eval_triggers @> '["match_updated"]'::jsonb;

    if pg_catalog.cardinality(ineligible_badge_ids) > 0 then
        raise exception
            'Live badge seed left ineligible badge IDs: %',
            pg_catalog.array_to_string(ineligible_badge_ids, ', ');
    end if;
end
$live_badge_seed$;
