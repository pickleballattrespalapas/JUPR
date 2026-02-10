-- ============================================================================
-- STAGING ONLY — DO NOT RUN IN PRODUCTION
-- STAGING ONLY — DO NOT RUN IN PRODUCTION
-- STAGING ONLY — DO NOT RUN IN PRODUCTION
-- This script bootstraps all tracked migrations in lexicographic order.
-- It is idempotent and records applied files in public.schema_migrations.
-- ============================================================================

CREATE TABLE IF NOT EXISTS public.schema_migrations (
  filename text PRIMARY KEY,
  applied_at timestamptz NOT NULL DEFAULT now()
);

-- >>> migrations/20250220_badges_v1.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250220_badges_v1.sql') THEN
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
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250220_badges_v1.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20250301_player_inactivity.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250301_player_inactivity.sql') THEN
    ALTER TABLE public.players
      ADD COLUMN IF NOT EXISTS last_game_at timestamptz NULL,
      ADD COLUMN IF NOT EXISTS inactive_at timestamptz NULL;
    
    -- Backfill last_game_at from recorded matches (global across leagues).
    WITH player_matches AS (
      SELECT club_id, t1_p1 AS player_id, "date" AS game_at, score_t1, score_t2 FROM public.matches
      UNION ALL
      SELECT club_id, t1_p2 AS player_id, "date" AS game_at, score_t1, score_t2 FROM public.matches
      UNION ALL
      SELECT club_id, t2_p1 AS player_id, "date" AS game_at, score_t1, score_t2 FROM public.matches
      UNION ALL
      SELECT club_id, t2_p2 AS player_id, "date" AS game_at, score_t1, score_t2 FROM public.matches
    ),
    player_max AS (
      SELECT
        club_id,
        player_id,
        MAX(game_at) AS max_game_at
      FROM player_matches
      WHERE COALESCE(score_t1, 0) + COALESCE(score_t2, 0) > 0
      GROUP BY club_id, player_id
    )
    UPDATE public.players p
    SET last_game_at = pm.max_game_at
    FROM player_max pm
    WHERE p.club_id = pm.club_id
      AND p.id = pm.player_id;
    
    -- Never-played players use created_at for inactivity baseline.
    UPDATE public.players
    SET inactive_at = NOW()
    WHERE inactive_at IS NULL
      AND COALESCE(last_game_at, created_at) <= (NOW() - INTERVAL '14 days');
    
    CREATE INDEX IF NOT EXISTS idx_players_inactive_at ON public.players (inactive_at);
    CREATE INDEX IF NOT EXISTS idx_players_last_game_at ON public.players (last_game_at);
    
    CREATE OR REPLACE FUNCTION public.mark_inactive_players() RETURNS void
    LANGUAGE plpgsql
    AS $$
    BEGIN
      UPDATE public.players
      SET inactive_at = NOW()
      WHERE inactive_at IS NULL
        AND COALESCE(last_game_at, created_at) <= (NOW() - INTERVAL '14 days');
    END;
    $$;
    
    CREATE EXTENSION IF NOT EXISTS pg_cron WITH SCHEMA extensions;
    
    DO $$
    BEGIN
      IF NOT EXISTS (
        SELECT 1 FROM cron.job WHERE jobname = 'mark-inactive-players-daily'
      ) THEN
        PERFORM cron.schedule(
          'mark-inactive-players-daily',
          '0 3 * * *',
          $$SELECT public.mark_inactive_players();$$
        );
      END IF;
    END
    $$;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250301_player_inactivity.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20250320_gamification_v2.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250320_gamification_v2.sql') THEN
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
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250320_gamification_v2.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20250325_player_badges_dedupe.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250325_player_badges_dedupe.sql') THEN
    with ranked as (
        select
            id,
            row_number() over (
                partition by club_id, player_id, badge_id, context_id
                order by earned_at asc, id asc
            ) as rn
        from player_badges
    )
    delete from player_badges
    where id in (select id from ranked where rn > 1);
    
    alter table player_badges
        drop constraint if exists player_badges_unique_context,
        drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key,
        drop constraint if exists player_badges_unique_context_type,
        add constraint player_badges_unique_context unique (club_id, player_id, badge_id, context_id);
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250325_player_badges_dedupe.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20250410_badges_copy_pack.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250410_badges_copy_pack.sql') THEN
    alter table badges
        alter column rarity set default 'common',
        alter column lore set default '',
        alter column hint set default '',
        alter column scope set default 'overall';
    
    update badges
    set rarity = coalesce(rarity, 'common')
    where rarity is null;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250410_badges_copy_pack.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20250415_gamification_story_badges.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250415_gamification_story_badges.sql') THEN
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
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250415_gamification_story_badges.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20250420_badge_definitions_seed.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250420_badge_definitions_seed.sql') THEN
    insert into badges (badge_id, name, prestige, category, is_stackable, is_active, rarity, tier, icon_key, lore, hint, scope, state)
    values
        ('participant', 'Participant', 10, 'Participation & Habit Loop', false, true, 'common', null, 'participant', 'Every career starts the same way: a name on the sheet and a moment on tape.', 'The archive doesn’t recognize potential. Only appearances.', 'overall', 'live'),
        ('dedicated_participant_50', 'Dedicated Participant', 25, 'Participation & Habit Loop', false, true, 'rare', null, 'dedicated_participant_50', 'Fifty matches is where routine becomes identity.', 'The schedule is starting to recognize you.', 'overall', 'live'),
        ('lifetime_participant_200', 'Lifetime Participant', 50, 'Participation & Habit Loop', false, true, 'epic', null, 'lifetime_participant_200', 'Two hundred matches doesn’t happen by accident. It happens by refusing to disappear.', 'The archive is getting heavy.', 'overall', 'live'),
        ('first_win', 'First Win', 15, 'Participation & Habit Loop', false, true, 'common', null, 'first_win', 'The first mark on the ledger. The tape starts rolling for real.', 'There is always a first frame on the reel.', 'overall', 'live'),
        ('weekly_regular', 'Weekly Regular', 35, 'Participation & Habit Loop', false, true, 'rare', null, 'weekly_regular', 'The schedule starts to recognize the face. The weeks keep stacking.', 'The calendar keeps calling.', 'league', 'live'),
        ('iron_week', 'Iron Week', 30, 'Participation & Habit Loop', true, true, 'common', null, 'iron_week', 'A week packed tight with tape. The grind leaves a signature.', 'Some weeks barely have room to breathe.', 'week', 'live'),
        ('marathon_month', 'Marathon Month', 60, 'Participation & Habit Loop', true, true, 'rare', null, 'marathon_month', 'A month that never slowed down. Every day kept its own clip.', 'The month is still writing.', 'month', 'live'),
        ('level_up', 'Level Up', 40, 'Skill Growth & Momentum', true, true, 'rare', null, 'level_up', 'Another rung claimed. The league notices the climb.', 'A new number appears on the nameplate.', 'league', 'live'),
        ('rocket_start', 'Rocket Start', 50, 'Skill Growth & Momentum', false, true, 'rare', null, 'rocket_start', 'The opening run shook the scoreboard. The room leaned in early.', 'The first stretch left a streak on the floor.', 'league', 'live'),
        ('most_improved_monthly', 'Most Improved', 55, 'Skill Growth & Momentum', true, true, 'epic', null, 'most_improved', 'A month where the climb couldn’t be ignored.', 'One month tilted harder than the rest.', 'month', 'live'),
        ('mountain_climber', 'Mountain Climber', 45, 'Skill Growth & Momentum', true, true, 'rare', null, 'mountain_climber', 'Ranks flipped. The ascent left landmarks.', 'The ladder looks different now.', 'league', 'live'),
        ('hot_streak', 'Hot Streak', 50, 'Skill Growth & Momentum', true, true, 'epic', null, 'hot_streak', 'Wins blur together on the tape. The run keeps rolling.', 'The film strip barely cools off.', 'league', 'live'),
        ('bounce_back', 'Bounce Back', 25, 'Skill Growth & Momentum', true, true, 'common', null, 'bounce_back', 'A stumble, then a reply. The echo lands clean.', 'The next frame told a different story.', 'match', 'live'),
        ('breakthrough', 'Breakthrough', 55, 'Skill Growth & Momentum', false, true, 'epic', null, 'breakthrough', 'The tape catches the moment the ceiling moves.', 'There was a game when things started to look different.', 'overall', 'live'),
        ('above_expectations', 'Above Expectations', 50, 'Skill Growth & Momentum', true, true, 'rare', null, 'above_expectations', 'The room expected one thing. The tape shows another.', 'The projection didn’t match the result.', 'overall', 'live'),
        ('ice_in_veins', 'Ice in Veins', 70, 'Clutch & Pressure', false, true, 'epic', null, 'ice_in_veins', 'When the margin tightened, the answer didn’t.', 'Cold hands don’t shake the camera.', 'overall', 'live'),
        ('clutch_performer', 'Clutch Performer', 60, 'Clutch & Pressure', false, true, 'rare', null, 'clutch_performer', 'Close frames keep leaning the same way.', 'The last points keep finding the same jersey.', 'overall', 'live'),
        ('pickle_perfection', 'Pickle Perfection', 75, 'Dominance & Quality', true, true, 'legendary', null, 'pickle_perfection', 'A shutout with no extra narration needed.', 'Sometimes the other side never shows up on the scoreboard.', 'match', 'live'),
        ('blowout_artist', 'Blowout Artist', 45, 'Dominance & Quality', true, true, 'rare', null, 'blowout_artist', 'The gap grew and never closed.', 'The margin kept widening.', 'match', 'live'),
        ('untouchable', 'Untouchable', 85, 'Dominance & Quality', true, true, 'legendary', null, 'untouchable', 'A run that didn’t flinch. The tape shows no breaks.', 'The run feels unbroken.', 'overall', 'live'),
        ('clean_sweep_week', 'Clean Sweep Week', 55, 'Dominance & Quality', true, true, 'epic', null, 'clean_sweep_week', 'Every clip in the week ended the same way.', 'A week with no counterpunches.', 'week', 'live'),
        ('high_roller', 'High Roller', 40, 'Dominance & Quality', true, true, 'rare', null, 'high_roller', 'The pace stayed high and the points kept pouring.', 'The scoreboard got a workout.', 'match', 'live'),
        ('dominant_run', 'Dominant Run', 45, 'Dominance & Quality', true, true, 'rare', null, 'dominant_run', 'The tape shows the same ending over and over.', 'The run keeps bending the same way.', 'league', 'live'),
        ('high_output', 'High Output', 40, 'Dominance & Quality', true, true, 'common', null, 'high_output', 'The scoreboard fills faster than the room expects.', 'The total keeps climbing.', 'match', 'live'),
        ('social_butterfly', 'Social Butterfly', 45, 'Versatility & Social Graph', false, true, 'rare', null, 'social_butterfly', 'The partner list turned into a montage.', 'So many different pairings on the same reel.', 'overall', 'live'),
        ('network_builder', 'Network Builder', 70, 'Versatility & Social Graph', false, true, 'epic', null, 'network_builder', 'The web stretches across the club.', 'The partner map keeps expanding.', 'overall', 'live'),
        ('draft_master', 'Draft Master', 55, 'Versatility & Social Graph', true, true, 'rare', null, 'draft_master', 'Different pairings, same result. The tape shows the range.', 'This month keeps swapping jerseys.', 'month', 'live'),
        ('swiss_army_knife', 'Swiss Army Knife', 65, 'Versatility & Social Graph', false, true, 'epic', null, 'swiss_army_knife', 'Versatility on the record. Different stages, same sharp edge.', 'The season shows more than one role.', 'season', 'live'),
        ('giant_slayer', 'Giant Slayer', 90, 'Prestige / Rarity', true, true, 'legendary', null, 'giant_slayer', 'A giant hit the floor and the camera never blinked.', 'A higher shadow fell.', 'match', 'live'),
        ('david_vs_goliath', 'David vs Goliath', 95, 'Prestige / Rarity', true, true, 'legendary', null, 'david_vs_goliath', 'The mismatch didn’t stay a mismatch.', 'The odds were heavy on one side.', 'match', 'live'),
        ('upset_champion', 'Upset Champion', 100, 'Prestige / Rarity', true, true, 'legendary', null, 'upset_champion', 'The month’s biggest swing stayed on the reel.', 'One month holds the loudest turn.', 'month', 'live'),
        ('legendary_upset', 'Legendary Upset', 120, 'Prestige / Rarity', true, true, 'legendary', null, 'legendary_upset', 'The tape caught a moment nobody predicted.', 'The odds looked impossible on this frame.', 'match', 'live'),
        ('nemesis_found', 'Nemesis Found', 55, 'Rivalry & Nemesis', false, true, 'rare', null, 'nemesis_found', 'A name keeps showing up across the net.', 'The same opponent keeps reappearing.', 'opponent', 'live'),
        ('rivalry_win', 'Rivalry Win', 60, 'Rivalry & Nemesis', true, true, 'rare', null, 'rivalry_win', 'The rivalry leaned your way on this frame.', 'One chapter shifted the rivalry.', 'match', 'live'),
        ('rivalry_streak', 'Rivalry Streak', 70, 'Rivalry & Nemesis', true, true, 'epic', null, 'rivalry_streak', 'The rivalry kept tilting, frame after frame.', 'The rivalry reel runs long.', 'opponent', 'live'),
        ('settled_the_score', 'Settled the Score', 65, 'Rivalry & Nemesis', true, true, 'epic', null, 'settled_the_score', 'The ledger finally balanced, the tape agreed.', 'A long account just evened out.', 'opponent', 'live'),
        ('battle_tested', 'Battle Tested', 50, 'Consistency & Reliability', true, true, 'rare', null, 'battle_tested', 'The season ran long and the tape kept proving it.', 'There’s weight behind this reel.', 'season', 'live'),
        ('consistency', 'Consistency', 60, 'Consistency & Reliability', true, true, 'epic', null, 'consistency', 'The tape keeps landing in the same place.', 'There’s a steady rhythm on this reel.', 'season', 'live'),
        ('steady_hand', 'Steady Hand', 55, 'Consistency & Reliability', false, true, 'rare', null, 'steady_hand', 'The season stayed smooth, even when the lights changed.', 'The season never wandered far.', 'season', 'live'),
        ('mr_reliable', 'Mr. Reliable', 80, 'Consistency & Reliability', false, false, 'epic', null, 'mr_reliable', 'Week after week, the reel looks the same.', 'Reliable tape needs reliable history.', 'season', 'live'),
        ('league_champion', 'League Champion', 140, 'Meta / Prestige', false, false, 'legendary', null, 'league_champion', 'A season ends with your name at the top.', 'The final table has a single line above the rest.', 'season', 'live'),
        ('league_runner_up', 'League Runner-Up', 120, 'Meta / Prestige', false, false, 'legendary', null, 'podium', 'Second place is still a statement.', 'Finish close enough to feel the title.', 'season', 'live'),
        ('league_third_place', 'League Third Place', 110, 'Meta / Prestige', false, false, 'legendary', null, 'podium', 'Third place keeps your season on the podium.', 'Stay in the top three when the league closes.', 'season', 'live'),
        ('top_performer_highest_rating', 'Top Performer: Highest Rating', 130, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The league closes with your rating on the peak.', 'Finish the season with the highest mark.', 'league', 'live'),
        ('top_performer_most_improved', 'Top Performer: Most Improved', 125, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The biggest climb shows up in the final tape.', 'Make the largest rating leap in the league.', 'league', 'live'),
        ('top_performer_best_win_pct', 'Top Performer: Best Win %', 120, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The league’s cleanest record shines at the top.', 'Finish with the best win percentage.', 'league', 'live'),
        ('top_performer_most_wins', 'Top Performer: Most Wins', 115, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'No one stacks wins faster when the season closes.', 'Lead the league in total wins.', 'league', 'live'),
        ('podium', 'Podium', 110, 'Meta / Prestige', false, false, 'legendary', null, 'podium', 'A season ends with your name on the stage.', 'The podium holds only a few.', 'season', 'live'),
        ('hall_of_fame_night', 'Hall of Fame Night', 100, 'Meta / Prestige', true, true, 'legendary', null, 'hall_of_fame_night', 'A night that pulled the cameras in closer.', 'Some nights feel larger than the rest.', 'match', 'live'),
        ('good_sport', 'Good Sport', 35, 'Sportsmanship & Community', false, false, 'rare', null, 'good_sport', 'The tape shows respect running both ways.', 'Sportsmanship leaves quieter fingerprints.', 'overall', 'live'),
        ('community_builder', 'Community Builder', 90, 'Sportsmanship & Community', false, false, 'epic', null, 'community_builder', 'The club grew around the moments you hosted.', 'Some stories start before the match.', 'overall', 'live'),
        ('mentor', 'Mentor', 70, 'Sportsmanship & Community', false, false, 'epic', null, 'mentor', 'The tape shows the lesson without saying a word.', 'The gap was wide, the guidance wider.', 'match', 'live')
    
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
        scope = excluded.scope,
        state = excluded.state
    ;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250420_badge_definitions_seed.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20250425_tournaments.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250425_tournaments.sql') THEN
    -- Tournament tables and match context columns
    
    ALTER TABLE public.matches
      ADD COLUMN IF NOT EXISTS context_type text NULL,
      ADD COLUMN IF NOT EXISTS context_id uuid NULL,
      ADD COLUMN IF NOT EXISTS tournament_id uuid NULL,
      ADD COLUMN IF NOT EXISTS tournament_game_id uuid NULL;
    
    CREATE INDEX IF NOT EXISTS idx_matches_tournament_id ON public.matches (tournament_id);
    CREATE INDEX IF NOT EXISTS idx_matches_context_type ON public.matches (context_type);
    
    CREATE TABLE IF NOT EXISTS public.tournaments (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      club_id text NOT NULL,
      name text NOT NULL,
      status text NOT NULL DEFAULT 'DRAFT',
      team_count integer NOT NULL,
      playoff_advance_count integer NULL,
      created_by_admin_id text NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    
    CREATE INDEX IF NOT EXISTS idx_tournaments_club_id ON public.tournaments (club_id);
    
    CREATE TABLE IF NOT EXISTS public.tournament_teams (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      tournament_id uuid NOT NULL REFERENCES public.tournaments(id) ON DELETE CASCADE,
      team_number integer NOT NULL,
      player1_id integer NULL,
      player2_id integer NULL,
      seed integer NULL,
      CONSTRAINT uq_tournament_team_number UNIQUE (tournament_id, team_number)
    );
    
    CREATE INDEX IF NOT EXISTS idx_tournament_teams_tournament_id ON public.tournament_teams (tournament_id);
    
    CREATE TABLE IF NOT EXISTS public.tournament_games (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      tournament_id uuid NOT NULL REFERENCES public.tournaments(id) ON DELETE CASCADE,
      stage text NOT NULL,
      rr_round_number integer NULL,
      rr_slot_number integer NULL,
      playoff_game_code text NULL,
      playoff_round text NULL,
      team_a_id uuid NULL REFERENCES public.tournament_teams(id),
      team_b_id uuid NULL REFERENCES public.tournament_teams(id),
      team_a_source jsonb NULL,
      team_b_source jsonb NULL,
      score_a integer NULL,
      score_b integer NULL,
      winner_team_id uuid NULL REFERENCES public.tournament_teams(id),
      loser_team_id uuid NULL REFERENCES public.tournament_teams(id),
      finalized_at timestamptz NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      updated_at timestamptz NOT NULL DEFAULT now()
    );
    
    CREATE INDEX IF NOT EXISTS idx_tournament_games_tournament_id ON public.tournament_games (tournament_id);
    CREATE INDEX IF NOT EXISTS idx_tournament_games_stage ON public.tournament_games (stage);
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250425_tournaments.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20250505_tournament_podium.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20250505_tournament_podium.sql') THEN
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
    
    insert into public.badges (badge_id, name, prestige, category, is_stackable, is_active, rarity, tier, icon_key, lore, hint, scope, state)
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
        , 'live'),
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
        , 'live'),
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
        , 'live')
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
        scope = excluded.scope,
        state = excluded.state;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20250505_tournament_podium.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260130_unique_leagues_metadata.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260130_unique_leagues_metadata.sql') THEN
    create unique index if not exists leagues_metadata_club_id_league_name_unique
        on leagues_metadata (club_id, lower(league_name));
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260130_unique_leagues_metadata.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260207_weekly_recaps.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260207_weekly_recaps.sql') THEN
    create table if not exists public.weekly_recaps (
        id uuid primary key default gen_random_uuid(),
        club_id text not null,
        week_start date not null,
        week_end date not null,
        status text not null default 'draft',
        generated_json jsonb not null default '{}'::jsonb,
        edits_json jsonb not null default '{}'::jsonb,
        final_json jsonb not null default '{}'::jsonb,
        created_at timestamptz not null default now(),
        updated_at timestamptz not null default now(),
        published_at timestamptz null,
        published_by text null,
        unique (club_id, week_start)
    );
    
    create index if not exists weekly_recaps_club_week_idx
        on public.weekly_recaps (club_id, week_start desc);
    
    create index if not exists weekly_recaps_club_status_week_idx
        on public.weekly_recaps (club_id, status, week_start desc);
    
    create or replace function public.set_updated_at_timestamp()
    returns trigger as $$
    begin
        new.updated_at = now();
        return new;
    end;
    $$ language plpgsql;
    
    drop trigger if exists weekly_recaps_set_updated_at on public.weekly_recaps;
    create trigger weekly_recaps_set_updated_at
    before update on public.weekly_recaps
    for each row
    execute function public.set_updated_at_timestamp();
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260207_weekly_recaps.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260215_end_league_top_performers.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260215_end_league_top_performers.sql') THEN
    alter table if exists public.leagues_metadata
        add column if not exists ended_at timestamptz,
        add column if not exists ended_by text,
        add column if not exists status text,
        add column if not exists end_awards jsonb;
    
    create index if not exists leagues_metadata_ended_at_idx
        on public.leagues_metadata (ended_at);
    
    create index if not exists leagues_metadata_status_idx
        on public.leagues_metadata (status);
    
    insert into public.badges (badge_id, name, prestige, category, is_stackable, is_active, rarity, tier, icon_key, lore, hint, scope, state)
    values
        ('top_performer_highest_rating', 'Top Performer: Highest Rating', 130, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The league closes with your rating on the peak.', 'Finish the season with the highest mark.', 'league', 'live'),
        ('top_performer_most_improved', 'Top Performer: Most Improved', 125, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The biggest climb shows up in the final tape.', 'Make the largest rating leap in the league.', 'league', 'live'),
        ('top_performer_best_win_pct', 'Top Performer: Best Win %', 120, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The league’s cleanest record shines at the top.', 'Finish with the best win percentage.', 'league', 'live'),
        ('top_performer_most_wins', 'Top Performer: Most Wins', 115, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'No one stacks wins faster when the season closes.', 'Lead the league in total wins.', 'league', 'live')
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
        scope = excluded.scope,
        state = excluded.state;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260215_end_league_top_performers.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260301_premium_league_editor.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260301_premium_league_editor.sql') THEN
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
    
    insert into public.badges (badge_id, name, prestige, category, is_stackable, is_active, rarity, tier, icon_key, lore, hint, scope, state)
    values
        ('top_performer_highest_rating', 'Top Performer: Highest Rating', 130, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The league closes with your rating on the peak.', 'Finish the season with the highest mark.', 'league', 'live'),
        ('top_performer_most_improved', 'Top Performer: Most Improved', 125, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The biggest climb shows up in the final tape.', 'Make the largest rating leap in the league.', 'league', 'live'),
        ('top_performer_best_win_pct', 'Top Performer: Best Win %', 120, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'The league’s cleanest record shines at the top.', 'Finish with the best win percentage.', 'league', 'live'),
        ('top_performer_most_wins', 'Top Performer: Most Wins', 115, 'Top Performer Awards', true, true, 'legendary', null, 'trophy', 'No one stacks wins faster when the season closes.', 'Lead the league in total wins.', 'league', 'live')
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
        scope = excluded.scope,
        state = excluded.state;
    
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
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260301_premium_league_editor.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260615_player_badges_unique_contract.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260615_player_badges_unique_contract.sql') THEN
    -- Canonical player_badges uniqueness contract:
    -- (club_id, player_id, badge_id, context_id)
    -- context_id encodes scope; context_type is informational only.
    
    with ranked as (
        select
            id,
            row_number() over (
                partition by club_id, player_id, badge_id, context_id
                order by earned_at asc nulls last, id asc
            ) as rn
        from player_badges
    )
    delete from player_badges
    where id in (select id from ranked where rn > 1);
    
    do $$
    declare
        constraint_name text;
    begin
        for constraint_name in
            select con.conname
            from pg_constraint con
            join pg_class rel on rel.oid = con.conrelid
            join pg_namespace nsp on nsp.oid = rel.relnamespace
            join unnest(con.conkey) with ordinality as cols(attnum, ord) on true
            join pg_attribute att on att.attrelid = rel.oid and att.attnum = cols.attnum
            where con.contype = 'u'
              and nsp.nspname = 'public'
              and rel.relname = 'player_badges'
            group by con.conname
            having bool_or(att.attname = 'context_type')
        loop
            execute format('alter table public.player_badges drop constraint if exists %I', constraint_name);
        end loop;
    
        execute 'alter table public.player_badges drop constraint if exists player_badges_unique_context';
        execute 'alter table public.player_badges drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key';
        execute 'alter table public.player_badges drop constraint if exists player_badges_unique_context_type';
    
        if not exists (
            select 1
            from pg_constraint con
            join pg_class rel on rel.oid = con.conrelid
            join pg_namespace nsp on nsp.oid = rel.relnamespace
            where con.contype = 'u'
              and nsp.nspname = 'public'
              and rel.relname = 'player_badges'
              and con.conname = 'player_badges_unique_context'
        ) then
            alter table public.player_badges
                add constraint player_badges_unique_context
                unique (club_id, player_id, badge_id, context_id);
        end if;
    end $$;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260615_player_badges_unique_contract.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260620_badge_state.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260620_badge_state.sql') THEN
    do $$
    begin
        if not exists (select 1 from pg_type where typname = 'badge_state') then
            create type badge_state as enum ('live', 'frozen', 'deprecated');
        end if;
    end $$;
    
    alter table if exists badges
        add column if not exists state badge_state not null default 'live',
        add column if not exists state_changed_at timestamptz not null default now(),
        add column if not exists state_change_reason text;
    
    update badges
    set
        state = coalesce(state, 'live'),
        state_changed_at = coalesce(state_changed_at, now());
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260620_badge_state.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260625_badge_recompute_runs.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260625_badge_recompute_runs.sql') THEN
    create table if not exists public.badge_eval_runs (
        id uuid primary key default gen_random_uuid(),
        created_at timestamptz not null default now(),
        created_by text,
        mode text not null,
        scope_json jsonb not null default '{}'::jsonb,
        status text not null default 'queued',
        started_at timestamptz,
        finished_at timestamptz,
        summary_json jsonb not null default '{}'::jsonb,
        error text
    );
    
    alter table if exists public.player_badges
        add column if not exists awarded_by text not null default 'engine',
        add column if not exists rule_version text,
        add column if not exists eval_run_id uuid references public.badge_eval_runs(id);
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260625_badge_recompute_runs.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260630_player_badges_revocation.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260630_player_badges_revocation.sql') THEN
    alter table if exists public.player_badges
        add column if not exists revoked_at timestamptz,
        add column if not exists revoked_by text,
        add column if not exists revoke_reason text;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260630_player_badges_revocation.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260705_badge_eval_queue.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260705_badge_eval_queue.sql') THEN
    create table if not exists public.badge_eval_queue (
        id uuid primary key default gen_random_uuid(),
        created_at timestamptz not null default now(),
        club_id text not null,
        context_id text,
        event_type text not null,
        player_ids bigint[] not null default '{}',
        match_id text,
        payload_json jsonb not null default '{}'::jsonb,
        status text not null default 'pending',
        attempts integer not null default 0,
        last_error text,
        processed_at timestamptz
    );
    
    create index if not exists badge_eval_queue_status_created_idx
        on public.badge_eval_queue (status, created_at);
    
    create index if not exists badge_eval_queue_club_status_idx
        on public.badge_eval_queue (club_id, status);
    
    create unique index if not exists badge_eval_queue_event_match_uidx
        on public.badge_eval_queue (event_type, match_id)
        where match_id is not null;
    
    create table if not exists public.player_badge_facts (
        club_id text not null,
        player_id bigint not null,
        context_id text not null,
        fact_key text not null,
        fact_value_json jsonb,
        fact_value_num numeric,
        updated_at timestamptz not null default now(),
        primary key (club_id, player_id, context_id, fact_key)
    );
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260705_badge_eval_queue.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260712_badge_eval_triggers.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260712_badge_eval_triggers.sql') THEN
    alter table if exists public.badges
        add column if not exists eval_triggers jsonb not null default '["match_recorded","match_updated"]'::jsonb;
    
    update public.badges
    set eval_triggers = '["match_recorded","match_updated"]'::jsonb
    where eval_triggers is null;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260712_badge_eval_triggers.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260715_perf_indexes.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260715_perf_indexes.sql') THEN
    create index if not exists player_badges_club_player_idx
        on public.player_badges (club_id, player_id);
    
    create index if not exists player_badges_club_badge_idx
        on public.player_badges (club_id, badge_id);
    
    create index if not exists player_badges_club_player_badge_active_idx
        on public.player_badges (club_id, player_id, badge_id)
        where revoked_at is null;
    
    create index if not exists badge_eval_queue_club_status_created_idx
        on public.badge_eval_queue (club_id, status, created_at);
    
    create index if not exists player_badge_facts_lookup_idx
        on public.player_badge_facts (club_id, player_id, context_id);
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260715_perf_indexes.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260720_badge_eval_queue_backfill.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260720_badge_eval_queue_backfill.sql') THEN
    create table if not exists public.badge_eval_queue (
        id uuid primary key default gen_random_uuid(),
        created_at timestamptz not null default now(),
        club_id text not null,
        context_id text,
        event_type text not null,
        player_ids bigint[] not null default '{}',
        match_id text,
        payload_json jsonb not null default '{}'::jsonb,
        status text not null default 'pending',
        attempts integer not null default 0,
        last_error text,
        processed_at timestamptz
    );
    
    create index if not exists badge_eval_queue_status_created_idx
        on public.badge_eval_queue (status, created_at);
    
    create index if not exists badge_eval_queue_club_status_idx
        on public.badge_eval_queue (club_id, status);
    
    create unique index if not exists badge_eval_queue_event_match_uidx
        on public.badge_eval_queue (event_type, match_id)
        where match_id is not null;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260720_badge_eval_queue_backfill.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260720_player_badges_provenance_grants.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260720_player_badges_provenance_grants.sql') THEN
    grant select (
        id,
        club_id,
        player_id,
        badge_id,
        earned_at,
        context_type,
        context_id,
        match_id,
        value_num,
        value_json,
        awarded_by,
        rule_version,
        eval_run_id,
        revoked_at,
        revoked_by,
        revoke_reason
    ) on public.player_badges to anon, authenticated;
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260720_player_badges_provenance_grants.sql');
  END IF;
END
$bootstrap$;

-- >>> migrations/20260725_events.sql
DO $bootstrap$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM public.schema_migrations WHERE filename = 'migrations/20260725_events.sql') THEN
    -- Events table for popup round robins and other club events
    
    CREATE TABLE IF NOT EXISTS public.events (
      id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
      club_id text NOT NULL,
      name text NOT NULL,
      event_type text NOT NULL DEFAULT 'popup_rr',
      is_active boolean NOT NULL DEFAULT true,
      created_at timestamptz NOT NULL DEFAULT now(),
      starts_at timestamptz NULL,
      ends_at timestamptz NULL,
      notes text NULL
    );
    
    CREATE INDEX IF NOT EXISTS idx_events_club_active ON public.events (club_id, is_active);
    CREATE INDEX IF NOT EXISTS idx_events_club_name ON public.events (club_id, name);
    INSERT INTO public.schema_migrations (filename) VALUES ('migrations/20260725_events.sql');
  END IF;
END
$bootstrap$;

NOTIFY pgrst, 'reload schema';
