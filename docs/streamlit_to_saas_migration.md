# JUPR Streamlit-to-SaaS Migration Roadmap

Branch reviewed: `Test`

This document is the durable reference for moving JUPR from a Streamlit-centered product into a long-term SaaS architecture. It is intentionally conservative: do not start by rewriting the UI in Next.js. The correct path is to turn the current Python codebase into a reusable product engine first, then add API, worker, and web frontends around it.

## Strategic Direction

JUPR should become:

```text
Python product engine + Supabase database/auth/storage
  ├── Streamlit admin client, temporarily
  ├── FastAPI/service API, later
  ├── worker processes for jobs
  └── Next.js public web client, later
```

The goal is not to replace Streamlit immediately. The goal is to stop making Streamlit the only place where routing, auth, domain operations, side effects, and jobs happen.

## Current Branch Reality

As of branch `Test`:

- `streamlit_app.py` is still the main application shell: config, Supabase setup, auth/session restoration, public/admin routing, role resolution, data loading, page imports, rendering, footer/header, and opportunistic badge queue processing.
- `jupr_app/ui/page_registry.py` already separates public and admin-only pages.
- `jupr_app/domain/admin/roles.py` already has role definitions for `super_admin`, `club_owner`, `organizer`, `scorekeeper`, and `read_only`.
- `jupr_app/ui/admin_page_permissions.py` already maps admin pages to permissions.
- `jupr_app/domain/match_processing.py` is the current canonical match/rating write engine, but it does too much in one function.
- Score-entry pages call domain functions directly, especially `process_matches()`.
- Badge queue and worker functions already exist, but are still triggered from Streamlit/admin flows.
- Player update subscriptions, digests, and outbox logic already exist, but email sending still depends on Streamlit-aware config/context.
- `jupr_app/ui/branding.py` still hardcodes Tres Palapas identity with `CLUB_ID = "tres_palapas"`.
- There is no Next.js app, FastAPI service, worker service, root `package.json`, or root `pyproject.toml` yet.

## Migration Sequence

1. Add a service layer inside the current Python app.
2. Route all match writes through the service layer.
3. Remove Streamlit imports from domain/service/worker-safe code.
4. Add worker entrypoints for badge queue and player-update emails.
5. Convert replay into job-tracked infrastructure.
6. Add database-backed club configuration.
7. Add public read models/views.
8. Add a small FastAPI shell.
9. Add the first Next.js public pages.
10. Gradually replace day-to-day admin workflows in Next.js.

## Non-Negotiable Rules

- Do not move rating math into JavaScript.
- Do not allow Next.js pages to write directly to rating tables.
- Do not duplicate match-processing logic in UI pages.
- Do not introduce a second match write path.
- Do not let background jobs require a live Streamlit session.
- Do not remove Streamlit admin until replacement workflows exist.
- All new write operations must be club-scoped.
- Preserve existing behavior unless a PR explicitly says otherwise.

---

# Codex PR Prompts

Use these prompts as one PR each. Base all branches from `Test` unless explicitly instructed otherwise.

---

## PR 01 — Add Service Layer Shell

**Suggested branch:** `codex/saas-01-service-layer-shell`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test

Create the first service-layer shell for the Streamlit-to-SaaS migration. This PR should add structure only and avoid behavioral changes.

Goal:
Introduce a reusable `jupr_app/services/` layer that can later be called by Streamlit pages, worker commands, and a future FastAPI service.

Create these files:
- jupr_app/services/__init__.py
- jupr_app/services/context.py
- jupr_app/services/result_types.py
- jupr_app/services/match_service.py
- tests/test_services_match_service_shell.py

Implementation details:
1. `jupr_app/services/context.py`
   - Define a lightweight dataclass `ServiceContext` with:
     - supabase: object
     - club_id: str
     - actor_email: str | None = None
     - actor_role: str | None = None
     - source: str | None = None
     - public_base_url: str | None = None
   - This must not import Streamlit.

2. `jupr_app/services/result_types.py`
   - Define a dataclass `ServiceResult` with:
     - ok: bool
     - data: dict[str, Any]
     - warnings: list[str]
     - errors: list[str]
   - Add classmethods:
     - `success(data: dict | None = None, warnings: list[str] | None = None)`
     - `failure(errors: list[str] | str, data: dict | None = None, warnings: list[str] | None = None)`
   - This must not import Streamlit.

3. `jupr_app/services/match_service.py`
   - Add a placeholder public function:
     - `submit_match_batch(ctx: ServiceContext, matches: list[dict[str, Any]], *, name_to_id: dict[str, int], df_players_all, df_leagues, df_meta) -> ServiceResult`
   - For this PR, it may call existing `jupr_app.domain.match_processing.process_matches()` directly and wrap the return in `ServiceResult.success(...)`.
   - Catch exceptions and return `ServiceResult.failure(...)`.
   - Do not change `process_matches()` internals in this PR.

4. Tests:
   - Add tests that verify `ServiceContext` can be created without Streamlit.
   - Add tests that verify `ServiceResult.success()` and `ServiceResult.failure()` behave predictably.
   - Add a monkeypatched test for `submit_match_batch()` proving it calls `process_matches()` with the expected club_id and wraps result data.

Constraints:
- Do not update Streamlit pages yet.
- Do not change rating behavior.
- Do not add FastAPI or Next.js.
- Do not add new dependencies.
- Keep imports worker-safe: no `streamlit` import in `jupr_app/services/*`.

Acceptance criteria:
- Tests pass.
- Existing app behavior is unchanged.
- New service layer exists and is ready for Streamlit pages to call in later PRs.
```

---

## PR 02 — Route Match Uploader Through Match Service

**Suggested branch:** `codex/saas-02-match-uploader-service`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test
Depends on: PR 01

Route the Match Uploader page through the new match service instead of calling `process_matches()` directly.

Goal:
Make `jupr_app/ui/pages/match_uploader.py` a client of `jupr_app.services.match_service.submit_match_batch()`.

Files likely changed:
- jupr_app/ui/pages/match_uploader.py
- jupr_app/services/match_service.py, only if minor service result ergonomics are needed
- tests/test_match_uploader_service_integration.py or similar, if practical

Implementation details:
1. Replace direct imports of `process_matches` in `match_uploader.py` with:
   - `ServiceContext`
   - `submit_match_batch`

2. Where the page currently calls `process_matches(...)`, build:
   - ServiceContext(
       supabase=supabase,
       club_id=str(club_id),
       actor_email=best available admin email from session_state if present,
       actor_role=st.session_state.get("admin_role"),
       source="match_uploader",
       public_base_url=st.session_state.get("base_url"),
     )

3. Call:
   - submit_match_batch(ctx_service, valid_batch_or_payload, name_to_id=..., df_players_all=..., df_leagues=..., df_meta=...)

4. Preserve current success/error user experience:
   - If result.ok is true, show inserted/skipped counts as before.
   - If result.ok is false, show a Streamlit error with the result errors.

5. Preserve current session behavior:
   - `force_data_refresh` must still be set after successful submission.
   - Existing `st.rerun()` flows should remain.

Constraints:
- Do not change match payload shape.
- Do not change rating calculations.
- Do not change player creation flow.
- Do not change UI layout except where needed for service result display.
- Do not edit unrelated pages.

Acceptance criteria:
- Match Uploader can still submit manual/batch matches.
- Match Uploader can still submit Single Round Robin matches.
- Existing match insert/update/rating behavior is preserved because the service still calls the existing engine.
- No direct `process_matches` import remains in `match_uploader.py`.
```

---

## PR 03 — Route All Score-Entry Pages Through Match Service

**Suggested branch:** `codex/saas-03-score-entry-service`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test
Depends on: PR 01 and PR 02

Route every match-submission UI page through `jupr_app.services.match_service.submit_match_batch()` so JUPR has one UI-to-service match write boundary.

Goal:
No Streamlit page should call `jupr_app.domain.match_processing.process_matches()` directly. Pages should call the match service.

Search for all direct imports/usages of `process_matches`. Likely files include:
- jupr_app/ui/pages/league_manager.py
- jupr_app/ui/pages/jupr_live_admin.py
- jupr_app/ui/pages/moneyball.py
- jupr_app/ui/pages/tournament_ops.py
- jupr_app/ui/pages/tournament_live.py
- jupr_app/ui/pages/challenge_ladder_admin.py
- jupr_app/ui/pages/admin_tools.py, if it is using process_matches for diagnostics/test tools
- Any other page found by search

Implementation details:
1. For each UI page that submits matches:
   - Replace direct `process_matches(...)` calls with `submit_match_batch(...)`.
   - Build a `ServiceContext` with:
     - supabase
     - club_id
     - actor_email where available
     - actor_role from `st.session_state.get("admin_role")`
     - source set to the page key, e.g. `league_manager`, `moneyball`, `tournament_ops`
     - public_base_url from session_state if available

2. Preserve page-specific payload construction.
   - Do not redesign forms.
   - Do not change match payload keys unless required by the service.

3. Preserve current success/error messages where possible.
   - If result.ok is false, show errors without crashing.
   - If result.ok is true, show inserted/skipped counts using result.data.

4. Update or add tests where feasible:
   - At minimum, add a static/import test that affected pages no longer import `process_matches` directly.
   - Add a service monkeypatch test for one representative page if practical.

Constraints:
- Do not change rating math.
- Do not change replay behavior.
- Do not change tournament result logic except the final submission boundary.
- Do not add API/Next.js/worker infrastructure in this PR.

Acceptance criteria:
- No Streamlit UI page directly imports `process_matches` except possibly approved diagnostics with an explanatory comment.
- All score-entry flows still call the same underlying match engine through the service.
- Tests pass.
```

---

## PR 04 — Split Match Processing Into Orchestrated Internals

**Suggested branch:** `codex/saas-04-match-processing-internals`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test
Depends on: PR 03

Refactor `jupr_app/domain/match_processing.py` so it is easier to test and eventually expose through an API, without changing behavior.

Goal:
Keep `process_matches()` as the public function for compatibility, but split its internals into smaller helper functions and types.

Files likely changed:
- jupr_app/domain/match_processing.py
- Optionally new files under jupr_app/domain/matches/
- tests around match processing, if existing; add new regression tests if practical

Preferred structure:
- jupr_app/domain/matches/__init__.py
- jupr_app/domain/matches/types.py
- jupr_app/domain/matches/normalization.py
- jupr_app/domain/matches/rating_engine.py
- jupr_app/domain/matches/persistence.py
- jupr_app/domain/matches/side_effects.py

Implementation details:
1. Keep the existing public API:
   - `process_matches(match_list, *, supabase, club_id, name_to_id, df_players_all, df_leagues, df_meta, sb_retry=None, default_k_factor=32, min_win_delta_elo=1.0, cap_loser_gain_elo=16.0)`

2. Extract pure/helper pieces where safe:
   - player id resolution
   - score extraction
   - rating_scope normalization
   - candidate player/league collection
   - match row construction
   - badge/player-update side effects

3. Preserve current return shape:
   - inserted
   - skipped_incomplete
   - skipped_empty
   - skipped_unrated
   - badge_summary
   - player_update_queue

4. Add regression tests for edge cases if feasible:
   - unrated matches are skipped from DB insert and counted as skipped_unrated
   - PopUp matches update overall but not island/league ratings
   - missing/incomplete players increment skipped_incomplete
   - zero-score rows increment skipped_empty

Constraints:
- This PR must be behavior-preserving.
- Do not change table names or schema.
- Do not add new dependencies.
- Do not modify Streamlit pages unless required by import moves.
- Do not introduce API endpoints.

Acceptance criteria:
- `process_matches()` remains import-compatible.
- Existing tests pass.
- New structure makes it easier to separately test normalization/rating/persistence/side effects.
```

---

## PR 05 — Remove Streamlit Dependency From Notification Sending

**Suggested branch:** `codex/saas-05-notification-config-cleanup`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test

Make notification sending worker-safe by removing Streamlit dependencies from domain notification modules.

Goal:
`jupr_app/domain/notifications/player_update_sender.py` and `jupr_app/domain/notifications/smtp_mailer.py` should not import Streamlit directly.

Files likely changed:
- jupr_app/config.py, new
- jupr_app/domain/notifications/player_update_sender.py
- jupr_app/domain/notifications/smtp_mailer.py
- jupr_app/ui/pages/player_updates_admin.py
- tests/test_notification_config.py

Implementation details:
1. Create `jupr_app/config.py`.
   - Define `get_env_or_default(name: str, default: str = "") -> str`.
   - Define `get_public_base_url(default: str = "http://localhost:8501") -> str` that reads env vars such as:
     - JUPR_PUBLIC_BASE_URL
     - PUBLIC_BASE_URL
   - Define `get_smtp_config()` returning a dict or dataclass with:
     - host
     - port
     - username
     - password
     - from_email
     - from_name
     - reply_to
     - use_tls
   - Do not import Streamlit in `jupr_app/config.py`.

2. Update `smtp_mailer.py`:
   - Remove `import streamlit as st`.
   - Read SMTP config through `jupr_app.config.get_smtp_config()`.
   - Preserve existing env var names:
     - SMTP_HOST
     - SMTP_PORT
     - SMTP_USERNAME
     - SMTP_PASSWORD
     - SMTP_FROM_EMAIL
     - SMTP_FROM_NAME
     - SMTP_REPLY_TO
     - SMTP_USE_TLS

3. Update `player_update_sender.py`:
   - Remove `import streamlit as st`.
   - Replace `_public_base_url()` with config/context-based resolution.
   - Functions that need public_base_url should accept it explicitly or read from a lightweight service context.
   - Preserve existing email content and links.

4. Update Streamlit page `player_updates_admin.py`:
   - Where it calls sender functions, pass base URL/config if needed.
   - It is okay for UI code to read Streamlit state/secrets, but domain notification code must not.

5. Tests:
   - Assert `jupr_app.domain.notifications.smtp_mailer` can import without Streamlit.
   - Assert missing SMTP config produces a clear error.
   - Assert public base URL can be resolved from environment.

Constraints:
- Do not change email templates.
- Do not change outbox table behavior.
- Do not change subscription approval/rejection flows.
- Do not add MailerSend API integration in this PR; preserve SMTP path.

Acceptance criteria:
- No `streamlit` import remains in notification domain sender/mailer modules.
- Player Updates Admin still works.
- Worker-safe email sending is now possible.
```

---

## PR 06 — Add Badge Queue Worker CLI

**Suggested branch:** `codex/saas-06-badge-worker-cli`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test

Add a CLI entrypoint for processing badge queue jobs outside Streamlit.

Goal:
Make badge queue processing runnable by a scheduled job or worker process without opening Admin Tools.

Files likely changed:
- jupr_app/workers/__init__.py, new
- jupr_app/workers/badge_queue_worker.py, new
- jupr_app/config.py, if needed
- tests/test_badge_worker_cli.py
- README or docs update, optional

Implementation details:
1. Create `jupr_app/workers/badge_queue_worker.py`.

2. The module should support:
   - `python -m jupr_app.workers.badge_queue_worker --club-id tres_palapas --max-total-jobs 500 --max-wall-clock-seconds 90`

3. It should:
   - Read Supabase URL/key from environment:
     - SUPABASE_URL
     - SUPABASE_SERVICE_ROLE_KEY preferred
     - SUPABASE_ANON_KEY fallback only if service role is not present
   - Build Supabase client using existing `jupr_app.data.client.make_supabase()`.
   - Call `process_badge_eval_queue_until_empty(...)` from existing badge worker code.
   - Print a JSON summary to stdout.
   - Exit nonzero if required config is missing or if the worker reports errors above a safe threshold.

4. Add a small callable function:
   - `run_badge_queue_worker(club_id: str, ...) -> dict`
   so tests can call it without spawning a subprocess.

5. Tests:
   - Monkeypatch Supabase/client and badge worker function.
   - Assert arguments are passed correctly.
   - Assert missing config produces a clean failure.

Constraints:
- Do not remove the Streamlit Admin Tools badge queue controls.
- Do not change badge awarding logic.
- Do not add external worker dependencies.

Acceptance criteria:
- Badge jobs can be processed outside Streamlit.
- Existing Admin Tools queue buttons still work.
- CLI is documented enough to run from deployment scheduler later.
```

---

## PR 07 — Add Player Update Email Worker CLI

**Suggested branch:** `codex/saas-07-player-update-worker-cli`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test
Depends on: PR 05

Add a CLI worker for sending pending player update emails from the outbox.

Goal:
Player update emails should be sendable by a scheduled process, not only by clicking "Send Pending" inside Streamlit Admin.

Files likely changed:
- jupr_app/workers/player_update_email_worker.py, new
- jupr_app/services/notification_service.py, optional new wrapper
- jupr_app/domain/notifications/player_update_sender.py, small signature cleanup if needed
- tests/test_player_update_email_worker.py

Implementation details:
1. Create `jupr_app/workers/player_update_email_worker.py`.

2. The module should support:
   - `python -m jupr_app.workers.player_update_email_worker --club-id tres_palapas --limit 250`

3. It should:
   - Read Supabase URL/key from env.
   - Read public base URL from env/config.
   - Use existing Supabase client factory.
   - Send pending player update emails using existing sender logic.
   - Print a JSON summary:
     - attempted
     - sent
     - skipped
     - errors

4. If current `send_pending_player_update_emails(ctx, ...)` requires the full Streamlit `AppContext`, create a lightweight service context or adapter that provides only what the sender needs:
   - supabase
   - club_id
   - public_base_url
   - data needed for digest generation

5. Do not duplicate digest generation logic.

6. Tests:
   - Monkeypatch sender function and Supabase config.
   - Verify CLI passes club_id and limit.
   - Verify JSON-like result shape.

Constraints:
- Do not remove the Player Updates Admin "Send Pending" button.
- Do not change email templates.
- Do not change subscription statuses.
- Do not add new external dependencies.

Acceptance criteria:
- Pending player update emails can be sent outside Streamlit.
- Existing Streamlit workflow still works.
- Worker-safe notification path exists.
```

---

## PR 08 — Add Replay Job Tracking

**Suggested branch:** `codex/saas-08-replay-jobs`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test

Add replay job tracking so destructive/rebuild operations have durable audit history and can later move to a worker.

Goal:
Keep current replay behavior available, but wrap it in a job record with status/result/error metadata.

Files likely changed:
- supabase/migrations/YYYYMMDDHHMMSS_replay_jobs.sql, new
- jupr_app/services/replay_service.py, new
- jupr_app/domain/replay_history.py, only if necessary
- jupr_app/ui/pages/admin_tools.py
- tests/test_replay_service.py

Migration should create table `replay_jobs`:
- id uuid primary key default gen_random_uuid()
- club_id text not null
- target_reset text not null
- status text not null default 'pending'
- actor_email text
- actor_role text
- started_at timestamptz
- finished_at timestamptz
- result_json jsonb not null default '{}'::jsonb
- error_text text
- created_at timestamptz not null default now()
- updated_at timestamptz not null default now()

Suggested status values:
- pending
- running
- succeeded
- failed

Implementation details:
1. Add `jupr_app/services/replay_service.py`.
   - `create_replay_job(...)`
   - `mark_replay_job_running(...)`
   - `mark_replay_job_succeeded(...)`
   - `mark_replay_job_failed(...)`
   - `run_replay_with_job_tracking(...)`

2. In `admin_tools.py`, where replay currently calls `replay_history(...)`, wrap with `run_replay_with_job_tracking(...)`.

3. Preserve current replay button behavior:
   - User still clicks replay.
   - Replay can still run synchronously in this PR.
   - UI shows same replay summary plus job id/status.

4. Tests:
   - Mock Supabase table calls.
   - Verify success path marks running then succeeded.
   - Verify exception path marks failed and preserves error text.

Constraints:
- Do not convert replay to async worker yet.
- Do not change replay math.
- Do not change bulk RPC names.
- Do not weaken permission checks.

Acceptance criteria:
- Every replay run creates a replay_jobs row.
- Success/failure is recorded.
- Existing replay UI remains usable.
```

---

## PR 09 — Add Database-Backed Club Configuration

**Suggested branch:** `codex/saas-09-club-config`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test

Add database-backed club configuration while preserving Tres Palapas as the default club.

Goal:
Prepare JUPR for multi-club SaaS without breaking the current single-club app.

Files likely changed:
- supabase/migrations/YYYYMMDDHHMMSS_clubs_config.sql, new
- jupr_app/domain/clubs.py or jupr_app/services/club_service.py, new
- jupr_app/ui/branding.py
- streamlit_app.py, minimal changes
- tests/test_club_config.py

Migration should create or update table `clubs`:
- id text primary key
- slug text unique not null
- name text not null
- tagline text
- support_email text
- public_base_url text
- logo_url text
- primary_color text
- is_active boolean not null default true
- created_at timestamptz not null default now()
- updated_at timestamptz not null default now()

Seed/upsert Tres Palapas:
- id: tres_palapas
- slug: tres-palapas
- name: Tres Palapas
- tagline: Official player ratings and events for Tres Palapas
- support_email: joe@juprleagues.com
- public_base_url: https://juprtrespalapas.streamlit.app
- is_active: true

Implementation details:
1. Add club config loader:
   - `get_default_club_id()` returns env `JUPR_DEFAULT_CLUB_ID` or `tres_palapas`.
   - `get_club_config(supabase, club_id: str) -> dict`
   - Must gracefully fallback to existing constants if table is missing.

2. Update `branding.py` carefully:
   - Keep constants for backward compatibility.
   - Add comments that constants are fallback defaults until DB club config is fully adopted.

3. Update `streamlit_app.py` minimally:
   - Continue using Tres Palapas default unless query/env override is explicitly available.
   - Do not break existing public links.

4. Tests:
   - Missing table fallback returns Tres Palapas defaults.
   - Existing row returns DB values.
   - Default club id can come from env.

Constraints:
- Do not attempt full multi-club routing in this PR.
- Do not change all queries.
- Do not remove `CLUB_ID` yet.
- Do not break current public URLs.

Acceptance criteria:
- Club config table/migration exists.
- Tres Palapas remains default.
- Future `/clubs/{slug}` routing is now possible.
```

---

## PR 10 — Add Public Leaderboard Read Model

**Suggested branch:** `codex/saas-10-public-leaderboard-read-model`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test
Depends on: PR 09 preferred, but can be done before if necessary

Add a clean public leaderboard read model for future API/Next.js consumption.

Goal:
Create a stable database view or service output so public web pages do not need to reconstruct leaderboards from raw app context DataFrames.

Files likely changed:
- supabase/migrations/YYYYMMDDHHMMSS_public_leaderboards_view.sql, new
- jupr_app/services/leaderboard_service.py, new
- tests/test_leaderboard_service.py
- Optional: update existing leaderboards page to use service in a low-risk way

Migration/view concept:
Create view `public_leaderboards` with safe public fields only, likely sourced from `players`, `league_ratings`, and `leagues_metadata`.

Suggested columns:
- club_id
- league_name
- player_id
- player_name
- rating
- rating_jupr
- wins
- losses
- matches_played
- is_active
- rank_position, if feasible
- updated_at, if available

Implementation details:
1. Add SQL view or table-backed read model.
   - Prefer a view for first iteration.
   - Keep it read-only.
   - Do not expose emails or admin/private fields.

2. Add `jupr_app/services/leaderboard_service.py`:
   - `get_public_leaderboard(supabase, club_id: str, league_name: str | None = None) -> list[dict]`
   - It should read from `public_leaderboards` if available.
   - It should fallback to existing tables if the view is missing, so deploys do not crash before migration is applied.

3. Tests:
   - View-backed path returns normalized list.
   - Missing-view fallback returns a safe result or clear warning.
   - No private fields are returned.

Constraints:
- Do not redesign leaderboard UI in this PR.
- Do not change rating calculations.
- Do not expose private subscription/admin data.

Acceptance criteria:
- Future API/Next.js can read public leaderboard data from one stable service/view.
- Existing Streamlit leaderboard remains functional.
```

---

## PR 11 — Add FastAPI Skeleton

**Suggested branch:** `codex/saas-11-fastapi-skeleton`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test
Depends on: PR 01 and preferably PR 10

Add a minimal FastAPI service skeleton without changing Streamlit deployment.

Goal:
Introduce a future API layer that can call existing JUPR services, starting with read-only health/club/leaderboard endpoints.

Files likely changed:
- services/api/main.py, new
- services/api/requirements.txt, new
- services/api/README.md, new
- tests/test_api_import.py or tests/test_api_health.py
- Optional root docs update

Implementation details:
1. Add `services/api/main.py`.
   - Create FastAPI app.
   - Endpoints:
     - GET /health returns `{ "ok": true, "service": "jupr-api" }`
     - GET /clubs/{club_slug} returns basic club info using club service/fallback
     - GET /clubs/{club_slug}/leaderboards returns public leaderboard using leaderboard service if available

2. API should read Supabase config from env:
   - SUPABASE_URL
   - SUPABASE_SERVICE_ROLE_KEY preferred for server-side API
   - SUPABASE_ANON_KEY fallback only for read-only local dev

3. Add `services/api/requirements.txt`:
   - fastapi
   - uvicorn
   - supabase
   - pandas only if unavoidable through current service imports

4. Keep Streamlit untouched.

5. Tests:
   - App imports successfully.
   - /health test passes using FastAPI TestClient if available.
   - If adding TestClient creates dependency issues, at least test app creation/import.

Constraints:
- Do not add admin write endpoints yet.
- Do not move Streamlit.
- Do not add Next.js.
- Do not duplicate rating logic.
- Do not expose private/admin tables.

Acceptance criteria:
- `services/api` exists and can run locally with uvicorn.
- Health endpoint works.
- API reads through service layer, not direct duplicated logic where avoidable.
```

---

## PR 12 — Add Next.js Public Shell

**Suggested branch:** `codex/saas-12-nextjs-public-shell`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test
Depends on: PR 10 and PR 11 preferred

Add the first Next.js public web shell for JUPR without replacing Streamlit.

Goal:
Create a separate public frontend that can eventually serve juprleagues.com and club pages, starting with read-only pages.

Files likely changed:
- apps/web/package.json, new
- apps/web/next.config.js or next.config.ts, new
- apps/web/app/layout.tsx, new
- apps/web/app/page.tsx, new
- apps/web/app/clubs/[clubSlug]/page.tsx, new
- apps/web/app/clubs/[clubSlug]/leaderboards/page.tsx, new
- apps/web/lib/api.ts, new
- apps/web/README.md, new

Implementation details:
1. Create a minimal Next.js app under `apps/web`.

2. Pages:
   - `/` simple JUPR landing page with link to `/clubs/tres-palapas`
   - `/clubs/[clubSlug]` club landing page
   - `/clubs/[clubSlug]/leaderboards` public leaderboard page

3. Data access:
   - Prefer calling the FastAPI service from PR 11 via env `NEXT_PUBLIC_JUPR_API_BASE_URL` or server-side env `JUPR_API_BASE_URL`.
   - If API is unavailable, show a graceful empty/error state.
   - Do not connect directly to Supabase from client-side browser code in this PR.

4. UI:
   - Simple, clean, mobile-friendly.
   - No admin login yet.
   - No score entry.
   - No rating writes.

5. Documentation:
   - Add `apps/web/README.md` with local dev commands and required env vars.

Constraints:
- Do not remove Streamlit public pages.
- Do not add admin functionality.
- Do not write to Supabase.
- Do not move rating logic to JavaScript.

Acceptance criteria:
- Next.js app can run locally.
- Public club and leaderboard pages render from API data or graceful fallback.
- Streamlit app remains untouched and deployable.
```

---

## PR 13 — First Next.js Admin Workflow: Scorekeeper Score Entry

> **Implemented contract:** The canonical score-entry route is now
> `/clubs/[clubSlug]/admin/score-entry`. It restores the operator's Supabase
> session and sends `Authorization: Bearer <access token>` directly to the
> protected FastAPI endpoint. The historical shared-token Next proxy described
> below has been retired; `/admin/clubs/[clubId]/score-entry` is compatibility
> routing only and resolves to the auth-aware club route; FastAPI remains the
> Supabase JWT authorization boundary.

**Suggested branch:** `codex/saas-13-nextjs-scorekeeper-entry`

```text
Repo: pickleballattrespalapas/JUPR
Base branch: Test
Depends on: PR 03, PR 11, PR 12

Add the first small Next.js admin workflow: scorekeeper-friendly score entry for an existing event/session.

Goal:
Prove that non-super-admin users can enter scores through the future web admin without replacing the full Streamlit admin console.

Scope:
- This should be a thin workflow.
- It should call a protected API endpoint.
- The API endpoint should call `jupr_app.services.match_service.submit_match_batch()`.

Backend changes:
1. Add protected endpoint:
   - POST /admin/clubs/{club_id}/matches/batch

2. Request body should include:
   - matches: list[dict]
   - source: string

3. Auth/permissions:
   - For first iteration, use a server-side admin token or placeholder guard documented clearly.
   - Do not pretend this is production-grade auth if it is not.
   - The endpoint must be structured so real Supabase JWT/role checks can be added later.
   - Require intended permission: `enter_scores`.

4. Endpoint must call match service, not `process_matches()` directly.

Frontend changes:
1. Add route:
   - `/admin/clubs/[clubId]/score-entry`

2. Minimal UI:
   - Enter/select league
   - Enter four player IDs or names
   - Enter scores
   - Submit
   - Show inserted/skipped result

3. No tournament manager, no replay, no player editor.

Constraints:
- Do not remove Streamlit admin.
- Do not implement full auth UI unless already available.
- Do not duplicate match/rating logic.
- Do not expose destructive admin tools.

Acceptance criteria:
- A future scorekeeper workflow exists.
- It uses the API/service path.
- It does not bypass existing match engine.
```

---

# Working Notes for Future Codex Runs

When running these prompts, keep PRs small. If Codex tries to implement multiple phases in one PR, stop and split the work. The safest path is boring, incremental architecture.

The most important checkpoint before adding serious Next.js work:

```text
All score-entry paths call the same Python service boundary.
Notification and badge jobs can run without Streamlit.
Public data has stable read models.
```

Only then should the Next.js app become more than a public shell.
