# Session Ladder Engine — Repo Discovery (JUPR)

## Scope of this discovery
This document maps the current codebase for implementing a new **Session Ladder Engine** without changing production code in this step.

---

## 1) Stack identification

### Frontend framework
- Primary app UI is **Streamlit** (Python), with route/page composition in `streamlit_app.py` and page modules under `jupr_app/ui/pages/`.
- There is also a **React + TypeScript + Vite** custom component in `jupr_court_board/frontend` used by Streamlit (`streamlit-component-lib`).

### Backend/API style
- No standalone REST/GraphQL server (no FastAPI/Flask router discovered).
- Backend is **server-side Streamlit + domain modules**, with data IO through the Supabase Python client.
- Data operations use:
  - PostgREST-style table CRUD (`supabase.table(...).select/insert/upsert/update/delete`), and
  - Supabase RPC calls (`supabase.rpc(...)`) via wrappers in `jupr_app/data/sb_write.py`.

### Database + migrations tooling
- Primary migration history is in **`supabase/migrations/`**.
- A legacy root `migrations/` folder exists; CI enforces that root SQL files do not carry operational DDL (`scripts/check_root_migrations_no_ddl.py`).
- Local migration runner script exists (`scripts/db_migrate.sh`) and applies `migrations/*.sql` against `DATABASE_URL` with a `schema_migrations` table.
- CI includes schema drift checks via Supabase CLI (`.github/workflows/schema-drift.yml`).

### Auth and roles
- Auth is handled via **Supabase Auth** in `streamlit_app.py` (`get_supabase_auth`, session resolution, login, magic link).
- Tenant scoping is derived from JWT `user_metadata.club_id`.
- Admin behavior is currently a two-step model:
  - authenticated Supabase user session, plus
  - local admin elevation gate via `ADMIN_PASSWORD` (session `admin_override`).
- Role-style UI permissions are mostly represented by `admin_logged_in` and public/admin page visibility lists.

### Existing league/tournament entities
Observed in loaders/migrations/domain modules:
- League-related:
  - `league_ratings`, `leagues_metadata`, and league-context `matches` usage in `jupr_app/data/load.py` and domain modules.
- Tournament-related:
  - canonical tournament tables in `supabase/migrations/20260221_create_tournaments_baseline.sql`:
    - `tournaments`
    - `tournament_teams`
    - `tournament_games`
    - `tournament_podium`
  - plus tournament sync logic in `jupr_app/domain/tournaments/`.
- Related event/recap entities already present:
  - `events`, `weekly_recaps`.

### Rating system
- Rating logic lives in `jupr_app/domain/ratings.py` and `jupr_app/domain/match_processing.py`.
- It uses a **hybrid Elo-style delta** (`calculate_hybrid_elo`) and applies updates for:
  - overall player ratings (`players`), and
  - league-island ratings (`league_ratings`) when context is league/non-popup.
- Match ingestion pipeline entrypoint is `jupr_app.domain.match_pipeline` (legacy wrapper at `services/match_pipeline.py`).

---

## 2) Locations requested

### Where DB migrations live
- Canonical: `supabase/migrations/`
- Legacy/non-canonical bucket: `migrations/`
- Guardrail script: `scripts/check_root_migrations_no_ddl.py`
- Migration runner script: `scripts/db_migrate.sh`

### Where API endpoints/services live
- There is no separate HTTP API layer currently.
- Service/domain entrypoints are in:
  - `services/` (e.g., `services/match_pipeline.py`)
  - `jupr_app/domain/` (business logic)
  - `jupr_app/data/` (Supabase IO wrappers and data loading)
- DB-RPC endpoint definitions are represented via SQL functions in `supabase/migrations/*.sql` and called from Python via `sb_rpc`/`supabase.rpc`.

### Where UI routes/pages live
- Navigation and route mapping: `streamlit_app.py` (`PAGES`, `PAGE_KEY_TO_LABEL`, `LABEL_TO_PAGE_KEY`, query-param handling).
- Page modules: `jupr_app/ui/pages/*.py`
- Shared UI components: `jupr_app/ui/components/*.py`
- Custom JS/TS court board component: `jupr_court_board/frontend/src/`

### Where tests live and how to run them
- Tests are under `tests/`.
- Primary runner: `pytest`.
- CI smoke gates are defined in `.github/workflows/ci.yml` (compile/import checks).

---

## 3) Proposed exact file paths/modules to create

### A) Session ladder domain logic
- `jupr_app/domain/session_ladder/`
  - `__init__.py`
  - `models.py` (session/round/court/game dataclasses/typing)
  - `rotation.py` (4p/5p templates + validation)
  - `stats.py` (W/L, PF, PA, PD recomputation)
  - `tiebreaks.py` (Wins → PD → PF → H2H → playoff resolver)
  - `movement.py` (adjacent-court movement engine, movers=1/2)
  - `state_machine.py` (session lifecycle transitions)
  - `resume.py` (resume pointer + deep-link helpers)
  - `feature_gate.py` (checks for `FEATURE_SESSION_LADDER` / `FEATURE_LIVE_SCORING`)

### B) Session console UI
- `jupr_app/ui/pages/session_console.py` (main `/sessions/:id` style page equivalent in Streamlit page map)
- `jupr_app/ui/components/session_console/`
  - `__init__.py`
  - `round_nav.py`
  - `court_board.py`
  - `score_entry.py`
  - `movement_preview.py`
  - `resume_banner.py`
- Optional custom front-end extension (if drag/drop court UX needed):
  - `jupr_court_board/frontend/src/sessionConsole.tsx`

### C) DB schema/migrations
- Add new canonical migrations in `supabase/migrations/` only, e.g.:
  - `supabase/migrations/2026xxxxxx_session_ladder_core.sql`
  - `supabase/migrations/2026xxxxxx_session_ladder_rounds_games.sql`
  - `supabase/migrations/2026xxxxxx_session_ladder_indexes_rls.sql`
- Suggested tables:
  - `session_ladders` (session header + config snapshot)
  - `session_ladder_entries` (roster/check-in + seed order)
  - `session_ladder_rounds` (round index + timing/status)
  - `session_ladder_courts` (court pods per round)
  - `session_ladder_games` (first-class game records + scores)
  - `session_ladder_resume` (per-user last pointer server fallback)

### D) API endpoints/services
Given current architecture, implement as Python service functions + optional SQL RPC (not REST):
- `jupr_app/services/session_ladder_service.py`
  - create/start session
  - check-in/import roster
  - generate round pods/templates
  - submit game scores
  - close round + apply movement
  - finalize session snapshot
  - get/set resume pointer
- `jupr_app/data/session_ladder_repo.py`
  - table CRUD + transactional helper calls
- Optional RPC migrations (if needed for atomic transitions):
  - `supabase/migrations/2026xxxxxx_session_ladder_rpc.sql`

---

## 4) Exact commands to run tests in this repo

### Unit tests
- Run all unit tests:
  - `pytest -q`
- Run a targeted suite (useful while building Session Ladder):
  - `pytest -q tests/test_session_ladder_domain.py`
- Import/smoke unit checks used by current CI posture:
  - `pytest -q tests/test_imports.py tests/test_navigation_determinism.py`

### E2E tests
- **Current state:** no dedicated automated E2E framework (no Playwright/Cypress test suite checked in).
- Practical E2E-like manual smoke commands:
  1. Start app:
     - `streamlit run streamlit_app.py`
  2. (If working on custom board component) build component:
     - `npm --prefix jupr_court_board/frontend run build`
  3. Optional local preview of component bundle:
     - `npm --prefix jupr_court_board/frontend run preview`

If automated E2E is introduced later, recommended location is `tests/e2e/` (Playwright) with a dedicated CI job.

---

## 5) Notes for Session Ladder implementation fit
- The repository already has ladder/league concepts in `league_manager` and challenge ladder pages; Session Ladder should be introduced as a separate feature-gated surface to avoid regressions.
- Feature flags currently available in config include:
  - `FEATURE_SESSION_LADDER`
  - `FEATURE_LIVE_SCORING`
- For this project’s architecture, “API endpoints” are best represented as service-layer operations and (optionally) Supabase RPC functions instead of a new web server.
