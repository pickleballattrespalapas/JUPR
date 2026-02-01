# Badge Engine (Single Source of Truth)

## Where badge definitions live
- Catalog metadata lives in `jupr_app/domain/gamification/badge_catalog.py`.
- Narrative copy (tape excerpts, highlight titles) lives in the copy pack under
  `jupr_app/domain/gamification/copy_pack.py` and the JSON payload it loads.

## Where unlock logic lives
- Evaluators: `jupr_app/domain/gamification/evaluators.py`
- Registry wiring: `jupr_app/domain/gamification/badge_registry.py`
- Entry point (award + persist): `jupr_app/domain/gamification/ensure_badges.py`

The engine computes `BadgeCandidate` objects via the registry/evaluators and persists them through
`badges_repo.upsert_player_badges`, which enforces idempotency using the
`club_id,player_id,badge_id,context_id` unique key.

## Player badge uniqueness contract
- **Uniqueness key:** `(club_id, player_id, badge_id, context_id)`
- `context_id` encodes the award scope (overall / league / season / day) and **must always be set**.
- `context_type` is informational only and does **not** participate in uniqueness.
- A badge can be earned multiple times only if the `context_id` differs.

## How to run backfill
### Streamlit admin tool
Use **Admin Tools → Badge Backfill**. This calls `ensure_badges()` with optional league and as-of
filters.

### Programmatic (Python)
```python
from jupr_app.domain.gamification.ensure_badges import ensure_badges

# ctx should include supabase, club_id, df_matches, df_players_all, df_leagues, df_meta, etc.
ensure_badges(ctx)  # full club
ensure_badges(ctx, league_id="Spring 2024 Ladder")  # scoped league
```

## Adding a new badge end-to-end
1. Add the badge to the catalog (`badge_catalog.py`).
2. Add copy pack entries (tape excerpts/highlight titles).
3. Implement an evaluator in `evaluators.py`.
4. Register the evaluator in `badge_registry.py`.
5. Add/extend tests:
   - award parity/threshold coverage,
   - idempotency (ensure no duplicate awards),
   - registry completeness (active catalog badges have evaluators).

## Tournament podium badges
Tournament podium badges are computed by the engine when `ensure_badges()` is called with
`tournament_id` (the evaluator queries `tournament_podium` and `tournament_teams`).
