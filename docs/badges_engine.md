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

## Recompute CLI (audit + backfill)
Use the recompute CLI to run audited badge evaluations in dry-run, append-only, or strict modes.

```bash
python -m jupr_app.cli.badge_recompute \
  --club_id CLUB123 \
  --mode dry-run \
  --league_id "Spring 2024 Ladder"

python -m jupr_app.cli.badge_recompute \
  --club_id CLUB123 \
  --mode append-only \
  --badge_id participant \
  --created-by admin@example.com

python -m jupr_app.cli.badge_recompute \
  --club_id CLUB123 \
  --mode strict \
  --badge_id participant \
  --revoke-reason "strict recompute cleanup" \
  --allow-strict-global
```

Each run writes a row to `badge_eval_runs` and, when updating awards, tags `player_badges` with
`awarded_by='recompute'`, a `rule_version` hash, and the `eval_run_id`. Strict mode soft-revokes
awards by setting `revoked_at`, `revoked_by`, and `revoke_reason`.

## Schema notes (player_badges provenance + revocation)
The recompute engine expects the migrations in:
- `supabase/migrations/20260625_badge_recompute_runs.sql`
- `supabase/migrations/20260630_player_badges_revocation.sql`

If your Supabase environment isn't applying migrations automatically, run the SQL below in the
Supabase SQL editor to add the missing columns and grants:

```sql
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
  add column if not exists eval_run_id uuid references public.badge_eval_runs(id),
  add column if not exists revoked_at timestamptz,
  add column if not exists revoked_by text,
  add column if not exists revoke_reason text;

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
```

### Applying migrations from the repo
If you have direct Postgres access, apply migrations in order with:

```bash
DATABASE_URL="postgres://USER:PASSWORD@HOST:5432/dbname" make db-migrate
```

## Incremental badge evaluation
When matches are ingested or edited, the app enqueues badge evaluation work instead of evaluating
the full match history on every page view:
- `badge_eval_queue` stores pending evaluation events (match recorded/updated, player IDs, context).
- `player_badge_facts` stores incremental counters used by evaluators (future worker use).

Current implementation only enqueues events; a worker will dequeue and process them separately.

### Running the worker locally
```bash
python - <<'PY'
from jupr_app.data.client import make_supabase
from jupr_app.domain.gamification.badge_worker import process_badge_eval_queue

supabase = make_supabase("https://YOUR_PROJECT.supabase.co", "SERVICE_ROLE_KEY")
process_badge_eval_queue(supabase, max_jobs=10, time_budget_seconds=5)
PY
```

### Scheduler suggestion
Run the worker on a cron (e.g., Supabase Scheduled Function or GitHub Actions) every few minutes to
drain the `badge_eval_queue` table without blocking Streamlit requests.

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
