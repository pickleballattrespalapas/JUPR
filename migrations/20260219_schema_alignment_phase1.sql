-- SCHEMA ALIGNMENT PHASE 1
-- Deterministic alignment to rebuild branch expectations
-- Additive only

-- Type-aligned companion columns for badge_eval_queue.
-- TODO: Backfill from legacy columns and switch application reads/writes before deprecating legacy fields.
ALTER TABLE IF EXISTS public.badge_eval_queue
    ADD COLUMN IF NOT EXISTS attempts_v2 integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS club_id_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS created_at_v2 timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS event_type_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS payload_json_v2 jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS player_ids_v2 bigint[] NOT NULL DEFAULT '{}'::bigint[],
    ADD COLUMN IF NOT EXISTS processed_at_v2 timestamptz NULL,
    ADD COLUMN IF NOT EXISTS status_v2 text NOT NULL DEFAULT '';

CREATE INDEX IF NOT EXISTS badge_eval_queue_status_v2_created_v2_idx
    ON public.badge_eval_queue (status_v2, created_at_v2);

-- Type-aligned companion columns for badge_eval_runs.
-- TODO: Backfill from legacy columns and switch application reads/writes before deprecating legacy fields.
ALTER TABLE IF EXISTS public.badge_eval_runs
    ADD COLUMN IF NOT EXISTS created_at_v2 timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS finished_at_v2 timestamptz NULL,
    ADD COLUMN IF NOT EXISTS mode_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS scope_json_v2 jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS started_at_v2 timestamptz NULL,
    ADD COLUMN IF NOT EXISTS status_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS summary_json_v2 jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE INDEX IF NOT EXISTS badge_eval_runs_status_v2_created_v2_idx
    ON public.badge_eval_runs (status_v2, created_at_v2);

-- Missing columns and type-aligned companion columns for badges.
-- TODO: Backfill *_v2 columns from legacy columns and migrate reads/writes before cleanup.
ALTER TABLE IF EXISTS public.badges
    ADD COLUMN IF NOT EXISTS category_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS created_at_v2 timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS hint_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS is_active_v2 boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS is_stackable_v2 boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS lore_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS name_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS prestige_v2 integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS scope_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS state text,
    ADD COLUMN IF NOT EXISTS state_change_reason text,
    ADD COLUMN IF NOT EXISTS state_changed_at timestamptz;

CREATE INDEX IF NOT EXISTS badges_state_idx
    ON public.badges (state);

-- Type-aligned companion columns for player_badges.
-- TODO: Backfill *_v2 columns from legacy columns and migrate reads/writes before cleanup.
ALTER TABLE IF EXISTS public.player_badges
    ADD COLUMN IF NOT EXISTS badge_id_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS club_id_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS context_type_v2 text NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS earned_at_v2 timestamptz NOT NULL DEFAULT now(),
    ADD COLUMN IF NOT EXISTS player_id_v2 bigint NOT NULL DEFAULT 0;

CREATE INDEX IF NOT EXISTS player_badges_club_player_badge_v2_idx
    ON public.player_badges (club_id_v2, player_id_v2, badge_id_v2);

-- Type-aligned companion columns for players.
-- TODO: Backfill *_v2 columns from legacy columns and migrate reads/writes before cleanup.
ALTER TABLE IF EXISTS public.players
    ADD COLUMN IF NOT EXISTS last_game_at_v2 timestamptz,
    ADD COLUMN IF NOT EXISTS inactive_at_v2 timestamptz;
