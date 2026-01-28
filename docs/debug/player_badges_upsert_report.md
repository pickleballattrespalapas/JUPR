# player_badges UPSERT / ON CONFLICT Diagnostic Report

## 1) Symptoms / Errors
- PostgREST APIError 42P10: “there is no unique or exclusion constraint matching the ON CONFLICT specification”.
- SQL errors when attempting to add constraints on columns that don’t exist (e.g., `context_key`, `scope`).
- Unique index creation failing due to duplicates in would-be key.
- Dedupe query failing because `created_at` column does not exist.

## 2) Code findings

### 2.1 `upsert_player_badges()` payload + on_conflict
- **Target table:** `player_badges`.
- **on_conflict string:** `club_id,player_id,badge_id,context_id`.
- **Payload columns (per row):** `id`, `club_id`, `player_id`, `badge_id`, `earned_at`, `context_type`, `context_id`, `match_id`, `value_num`, `value_json`.

Evidence (excerpt):
```
supabase.table("player_badges").upsert(
    rows[i : i + chunk],
    on_conflict=PLAYER_BADGES_ON_CONFLICT,
).execute()
```
```
rows.append(
    {
        "id": str(uuid4()),
        "club_id": club_id,
        "player_id": int(candidate.player_id),
        "badge_id": candidate.badge_id,
        "earned_at": now,
        "context_type": candidate.context_type,
        "context_id": context_id,
        "match_id": candidate.match_id,
        "value_num": candidate.value_num,
        "value_json": value_json,
    }
)
```
Source: `jupr_app/domain/gamification/badges_repo.py`.【F:jupr_app/domain/gamification/badges_repo.py†L13-L104】

### 2.2 Call sites and context fields
- `award_tournament_trophies_from_podium()` builds `BadgeCandidate` with:
  - `context_type="tournament"`
  - `context_id=f"{tournament_id}:podium:{placement}"`
  - `match_id=None`
  - `value_json` includes `tournament_id`, `tournament_name`, `placement`, `team_id`, `team_number`
  - `value_num=float(placement)`
  - then calls `upsert_player_badges()`.

Evidence: `jupr_app/domain/tournament_podium.py`.【F:jupr_app/domain/tournament_podium.py†L56-L118】

- `ensure_league_top_performer_awards()` builds `BadgeCandidate` with:
  - `context_type="league"`
  - `context_id=f"{league_id}:top_performer:{category_key}:{rank}"`
  - `match_id=None`
  - `value_json` includes `league_id`, `category_key`, `category_label`, `metric_value`, `metric_display`, `min_games`, `rank`
  - `value_num=award.get("metric_value")`
  - then calls `upsert_player_badges()`.

Evidence: `jupr_app/domain/gamification/top_performer_awards.py`.【F:jupr_app/domain/gamification/top_performer_awards.py†L109-L151】

- Admin backfill uses `compute_candidates_for_club(...)` and calls `upsert_player_badges()` with whatever `BadgeCandidate`s are produced (context fields depend on badge rules).

Evidence: `jupr_app/ui/pages/admin_tools.py`.【F:jupr_app/ui/pages/admin_tools.py†L292-L333】

### 2.3 Debug logging around upsert (added)
A temporary debug log now records:
- the `on_conflict` string
- a redacted sample of the first 3 rows
- batch size

Evidence: `jupr_app/domain/gamification/badges_repo.py`.【F:jupr_app/domain/gamification/badges_repo.py†L58-L103】

### 2.4 Runtime trace attempt
The requested runtime trace for `award_tournament_trophies_from_podium()` requires a Supabase-connected environment. It has **not** been executed here due to missing Supabase credentials. Once credentials are available, re-run the tournament trophy flow and collect the log output from the debug lines above.

## 3) DB schema findings

### 3.1 Repo schema (migrations)
The repo migrations define `public.player_badges` with the following columns:
- `id uuid`, `club_id text`, `player_id bigint`, `badge_id text`, `earned_at timestamptz`, `context_type text`, `context_id text`, `match_id text`, `value_num numeric`, `value_json jsonb`.

Evidence: `migrations/20250220_badges_v1.sql`.【F:migrations/20250220_badges_v1.sql†L17-L33】

The most recent migration updates the unique constraint to:
- `unique (club_id, player_id, badge_id, context_id)`

Evidence: `migrations/20250320_gamification_v2.sql`.【F:migrations/20250320_gamification_v2.sql†L17-L21】

A later dedupe migration also assumes uniqueness on `(club_id, player_id, badge_id, context_id)`.

Evidence: `migrations/20250325_player_badges_dedupe.sql`.【F:migrations/20250325_player_badges_dedupe.sql†L1-L17】

### 3.2 Supabase (actual) schema
A CLI script was added to query Supabase directly (information_schema + constraints + indexes + duplicates). The script expects `DATABASE_URL` or `SUPABASE_DB_URL` and uses `psycopg2`.

Script: `scripts/player_badges_db_debug.py`.【F:scripts/player_badges_db_debug.py†L1-L95】

**Attempted execution (in this environment):**
```
$ python scripts/player_badges_db_debug.py
Missing DATABASE_URL/SUPABASE_DB_URL env var.
```
Because DB credentials are not available in this environment, **no live SQL output is captured yet**. Run the script in your Supabase-connected environment to populate the report sections below.

## 4) Data hygiene findings (duplicates)

### 4.1 Repo dedupe logic
The repo includes a dedupe migration that removes duplicate rows by `(club_id, player_id, badge_id, context_id)` without relying on `created_at` (it uses `earned_at` and `id`).

Evidence: `migrations/20250325_player_badges_dedupe.sql`.【F:migrations/20250325_player_badges_dedupe.sql†L1-L12】

### 4.2 Required live checks (not yet executed)
Run the CLI script to obtain:
- `player_badges` columns list
- existing primary/unique constraints
- indexes
- duplicates by `(club_id, player_id, badge_id)` + total extra rows

The script will print the exact results for these required queries.

## 5) Root cause
Based on repo evidence and the error `42P10` (“no unique or exclusion constraint matching the ON CONFLICT specification”), the **most likely** root causes are:

1. **Supabase schema is out of sync with repo migrations**, leaving a unique constraint that does *not* match `on_conflict=club_id,player_id,badge_id,context_id`.
   - Example: if the DB still uses the earlier constraint with `context_type` (from `20250220_badges_v1.sql`), the current `on_conflict` will not match, triggering 42P10.
2. **Constraints/indexes were attempted on non-existent columns** (`context_key`, `scope`), which do not exist on `player_badges` in repo migrations.
3. **Duplicates already exist** in the target key, preventing unique index creation, and any dedupe query using `created_at` will fail because `created_at` is not in `player_badges` (repo uses `earned_at`).

**DB confirmation required:** Run `scripts/player_badges_db_debug.py` to see the actual constraints and duplicates in your Supabase database and confirm which of these applies.

## 6) Fix options

### Option A — Align DB constraint to existing code (minimal change)
**Goal:** Ensure `player_badges` has a unique constraint that matches `on_conflict=club_id,player_id,badge_id,context_id`.

**Pros:**
- No code change required (except optional logging).
- Matches repo expectations and existing tests.

**Cons:**
- If you need to award the *same badge* multiple times with different semantics, you must ensure `context_id` is distinct per award.

**SQL migration (exact):**
```sql
-- 1) Optional: dedupe first (if duplicates exist)
with ranked as (
    select
        id,
        row_number() over (
            partition by club_id, player_id, badge_id, context_id
            order by earned_at asc, id asc
        ) as rn
    from public.player_badges
)
delete from public.player_badges
where id in (select id from ranked where rn > 1);

-- 2) Ensure correct unique constraint
alter table public.player_badges
    drop constraint if exists player_badges_unique_context,
    drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key,
    drop constraint if exists player_badges_unique_context_type,
    add constraint player_badges_unique_context unique (club_id, player_id, badge_id, context_id);
```

**Code change (if any):** none; `upsert_player_badges()` already uses the correct `on_conflict` string for this constraint.【F:jupr_app/domain/gamification/badges_repo.py†L13-L69】

### Option B — Add deterministic `award_key` to support repeatable awards
**Goal:** If some awards should be repeatable even with same `badge_id` and `context_id`, store a deterministic `award_key` and enforce uniqueness on that key.

**Pros:**
- Allows repeatable awards without overloading `context_id`.
- Makes idempotency explicit in the DB.

**Cons:**
- Requires schema change + code changes to compute/store `award_key` for every insert/upsert.

**SQL migration (exact):**
```sql
alter table public.player_badges
    add column if not exists award_key text;

create unique index if not exists player_badges_award_key_uidx
    on public.player_badges (award_key);
```

**Code change (exact):**
- Update `upsert_player_badges()` to compute `award_key` and upsert on it (or insert a unique constraint on `(club_id, player_id, badge_id, context_id, award_key)` if keeping other fields).
- Example deterministic key: `f"{club_id}:{player_id}:{badge_id}:{context_type}:{context_id}"` (or include extra award-specific discriminators).

## 7) Recommended plan (single best)
**Recommend Option A** unless live DB inspection shows that you *require* repeated awards with identical `(club_id, player_id, badge_id, context_id)`.

Steps:
1. Run `scripts/player_badges_db_debug.py` in your Supabase-connected environment to confirm actual constraints and duplicates.
2. If duplicates exist, run the dedupe SQL above (safe: uses `earned_at` and `id`, not `created_at`).
3. Apply the unique constraint on `(club_id, player_id, badge_id, context_id)`.
4. Re-run the failing path to confirm `ON CONFLICT` is accepted.

## 8) Exact SQL migrations + exact code changes

### SQL (Option A)
```sql
-- Dedupe (only if duplicates exist)
with ranked as (
    select
        id,
        row_number() over (
            partition by club_id, player_id, badge_id, context_id
            order by earned_at asc, id asc
        ) as rn
    from public.player_badges
)
delete from public.player_badges
where id in (select id from ranked where rn > 1);

-- Unique constraint fix
alter table public.player_badges
    drop constraint if exists player_badges_unique_context,
    drop constraint if exists player_badges_club_id_player_id_badge_id_context_id_key,
    drop constraint if exists player_badges_unique_context_type,
    add constraint player_badges_unique_context unique (club_id, player_id, badge_id, context_id);
```

### Code edits (debug + optional change)
- **Debug logging already added** around `upsert_player_badges()` to emit `on_conflict` and sample rows. This is temporary and should be removed after verification.

Evidence: `jupr_app/domain/gamification/badges_repo.py`.【F:jupr_app/domain/gamification/badges_repo.py†L58-L123】

If Option B is chosen, update `upsert_player_badges()` to compute `award_key` and switch the `on_conflict` to `award_key`.

## Risk notes
- **Data deletion risk (dedupe):** The dedupe SQL deletes duplicate rows for the same `(club_id, player_id, badge_id, context_id)`; export duplicates first if you need a record of historical awards.
- **Idempotency:** Without a matching unique constraint, upserts will fail or insert duplicates, breaking idempotency. Align `on_conflict` with the DB constraint to restore safe replays.
- **Tournament trophies:** The tournament trophy award path sets `context_id` to `"{tournament_id}:podium:{placement}"` and should be repeatable per tournament + placement. Ensure any uniqueness key includes this context to avoid duplicate awards across replays.
