# Gamification System Guide

## Badge Catalog
Badges live in `jupr_app/domain/gamification/badge_catalog.py`. Each `BadgeDefinition` includes:
- Catalog metadata (name, category, rarity, prestige, tier, icon_key, scope)
- Narrative fields (`lore`, `hint`) written in documentary tone
- Flags for stackable/active behavior

When seeding, the catalog is upserted into the `badges` table so UI clients can build the Badge Codex.

## Award Rules
Award logic is isolated in `jupr_app/domain/gamification/badge_rules.py` and uses a reusable
player-match fact table derived from match data. The rules:
- Always filter matches with `apply_match_filters` (no voided/invalid records)
- Produce stable context IDs for idempotency
- Attach a narrative `tape_excerpt` inside `value_json` for each award

Badges that need unavailable data should remain inactive in the catalog until the inputs exist.

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
The gamification jobs run when `ensure_badges(ctx)` is executed. In Streamlit, this happens during
data refresh. You can also call the function manually in a notebook or shell by constructing a
context object with `supabase`, `club_id`, and match dataframes.

## Adding a New Badge
1. Add a `BadgeDefinition` entry in `badge_catalog.py` with lore + hint.
2. Implement award logic in `badge_rules.py`, returning a stable `context_id`.
3. Add a tape excerpt in the award rule and, if needed, a highlight in `story_engine.py`.
4. Update tests to cover the new behavior.
