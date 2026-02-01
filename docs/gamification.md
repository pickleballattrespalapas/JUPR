# Gamification System Guide

## Badge Catalog
Badges live in `jupr_app/domain/gamification/badge_catalog.py`. Each `BadgeDefinition` includes:
- Catalog metadata (name, category, rarity, prestige, tier, icon_key, scope)
- Narrative fields (`lore`, `hint`) written in documentary tone
- Flags for stackable/active behavior

When seeding, the catalog is upserted into the `badges` table so UI clients can build the Badge Codex.

## Badge State (Awardability)
Badges also have a `state` that controls award eligibility:
- **live**: eligible for new awards.
- **frozen**: no new awards, but previously earned badges remain visible.
- **deprecated**: no new awards and treated as legacy; previously earned badges remain visible.

State changes should be tracked with `state_changed_at` and optional `state_change_reason` in the `badges` table.

## Award Rules
Award logic lives in the badge engine evaluators (`jupr_app/domain/gamification/evaluators.py`)
and is registered in `jupr_app/domain/gamification/badge_registry.py`. The engine builds a
player-match fact table derived from match data. The rules:
- Always filter matches with `apply_match_filters` (no voided/invalid records)
- Produce stable context IDs for idempotency
- Attach a narrative `tape_excerpt` inside `value_json` for each award

Badges that need unavailable data should remain inactive in the catalog until the inputs exist.

## Award Auditing
Each award in `player_badges` can include provenance fields:
- `awarded_by`: `engine`, `recompute`, or `admin` (default is `engine`).
- `rule_version`: optional text to track evaluator revisions.
- `eval_run_id`: optional link to a `badge_eval_runs` entry used for recompute/backfill runs.

Recompute/backfill runs can be logged in `badge_eval_runs` with scope metadata and summaries.
Strict recompute runs may soft-revoke awards by setting `revoked_at`, `revoked_by`, and `revoke_reason`.

## Incremental evaluation queue
Match ingestion and edits enqueue work in `badge_eval_queue` so badge evaluation can run
incrementally instead of scanning all matches during page rendering.

## Story Cards
`jupr_app/domain/gamification/story_engine.py` builds story cards with:
- Highlights (new badges, signature wins, rare moments)
- Foreshadowing (nearly-there streaks or milestones)

Story rows are written to `player_stories` using a unique constraint to prevent duplicates.

## Profile Summary
`jupr_app/domain/gamification/profile.py` assembles the gamification payload:
- Prestige total (stackable badges count multiple times)
- Unlocked badges with latest tape excerpt
- Locked badges with lore + hint (no rules)
- Collection stats by category

## Running Locally
The gamification jobs run when `ensure_badges(ctx)` is executed (see
`jupr_app/domain/gamification/ensure_badges.py`). In Streamlit, this happens during
data refresh. You can also call the function manually in a notebook or shell by constructing a
context object with `supabase`, `club_id`, and match dataframes.

## Adding a New Badge
1. Add a `BadgeDefinition` entry in `badge_catalog.py` with lore + hint.
2. Implement award logic in `evaluators.py` and register it in `badge_registry.py`, returning a stable `context_id`.
3. Add any copy pack updates needed for tape excerpts/highlights.
4. Update tests to cover the new behavior and registry completeness.

## Verification Checklist (Production)
### SQL sanity checks
- Confirm the catalog seed count (should be 52 rows):
  ```sql
  select count(*) as badge_count from badges;
  ```
- Confirm active badge count (if using `is_active` gating in UI):
  ```sql
  select count(*) as active_badge_count from badges where is_active is true;
  ```
- Confirm player badge volume and recent awards:
  ```sql
  select count(*) as player_badge_count from player_badges;
  select pb.player_id, pb.badge_id, b.name, pb.earned_at
  from player_badges pb
  join badges b on b.badge_id = pb.badge_id
  order by pb.earned_at desc
  limit 10;
  ```
- Spot missing copy (should be empty if v1 copy complete):
  ```sql
  select badge_id, name
  from badges
  where coalesce(nullif(trim(hint), ''), '') = ''
     or coalesce(nullif(trim(lore), ''), '') = '';
  ```

### In-app checks
- Player Trophy Room:
  - Total badge count matches the catalog (52 definitions; active count will differ if `is_active` is used to hide badges).
  - Collection count increments when unlocking a new badge.
  - Top Prestige chips render with names and icons.
  - Locked badges show hint text; unlocked badges show lore.
