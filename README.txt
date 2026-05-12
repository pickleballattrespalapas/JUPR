# JUPR Operator README (Joe)

This is the operator README for JUPR at Tres Palapas.
It is **not** a public contributor guide.

> **Security rule (non-negotiable):** never paste secret values into the repo, commit history, issues, PRs, ChatGPT, Streamlit logs, GitHub Actions logs, or issue comments.

## 1) What JUPR is

- JUPR at Tres Palapas is the official player ratings and events system.
- It is used for official player ratings and official Tres Palapas events.
- Production data and production workflows are live operations.
- The SaaS path is being built around the working product, not by replacing it all at once.

## 2) SaaS direction

Target architecture:

```text
Python product engine + Supabase database/auth/storage
  ├── Streamlit admin client, active for now
  ├── FastAPI service API, staging first
  ├── worker processes for jobs
  └── Next.js public web client, staging/read-only first
```

Direction rules:

- Keep Streamlit as the active admin console for rated events until replacement workflows are proven.
- Launch FastAPI + Next.js as a read-only public SaaS surface first.
- Move background jobs out of Streamlit through worker CLIs.
- Do not enable Next.js admin score entry for rated matches until real auth, club-scoped authorization, and audit attribution are implemented.
- Do not duplicate rating logic in JavaScript.
- Do not introduce a second match-processing write path.
- Use `docs/saas_status.md` as the durable SaaS implementation status tracker (done/in progress/blocked/not started).

## 3) Branch and environment model

Branch model:

- `rollback-feb8` = **production**
- `Test` = **staging**
- `professional/*` = **feature/PR branches** that merge into `Test`

Workflow:

1. Work on `professional/*`.
2. Open PR into `Test`.
3. Validate in staging against staging Supabase.
4. Promote to `rollback-feb8` only after staging is stable and production impact is understood.

Environment model:

- Production Streamlit deploys from `rollback-feb8` and uses production Supabase.
- Test Streamlit deploys from `Test` and should use staging Supabase when validating staging/SaaS changes.
- FastAPI staging API uses staging Supabase.
- Next.js staging web app talks to FastAPI staging API.
- Production traffic remains on Streamlit until the SaaS path is explicitly approved for production.

## 4) Supabase data environments

- **Production Supabase project** = live JUPR production data.
- **Staging Supabase project** = non-production database for `Test` branch validation, SaaS API/web testing, migrations, workers, and auth experiments.

Rules:

- Never point staging API, staging web app, or Test Streamlit deployment at production Supabase by accident.
- Never point `SUPABASE_TEST_DATABASE_URL` at production.
- Never use production service-role keys in staging.
- Do not copy production table data into staging unless intentionally approved and sanitized.
- Schema-only copy is allowed for staging validation when run through the guarded GitHub Actions workflow.
- `supabase/migrations/` is the canonical migration folder.
- Apply migrations to staging before production.
- Production SQL changes should be manually reviewed before running in Supabase SQL editor.

Required staging variables:

```bash
JUPR_ENV=staging
SUPABASE_URL=<staging Supabase project URL>
SUPABASE_SERVICE_ROLE_KEY=<staging service role key>
SUPABASE_ANON_KEY=<staging anon key if needed>
JUPR_API_BASE_URL=<staging FastAPI URL>
NEXT_PUBLIC_JUPR_API_BASE_URL=<staging FastAPI URL if needed>
JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0
NEXT_PUBLIC_JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0
```

## 5) SaaS guardrails

Next.js admin score entry for rated matches must remain disabled until all of the following exist:

- Supabase Auth JWT validation in FastAPI.
- Admin role lookup from `admin_role_assignments`.
- Club-scoped authorization for every write.
- Audit identity attribution in `admin_activity_log`.
- Session/CSRF strategy defined for the chosen auth model.
- Staging E2E tests proving scorekeeper writes affect only the intended club.

Non-negotiable rules:

- No static admin token for production user-initiated actions.
- No service-role key in browser/client code.
- No direct browser writes to rating tables.
- No JavaScript rating math.
- No duplicate match-write engine.
- All new write operations must be club-scoped.
- Preserve current production behavior unless a PR explicitly says otherwise.

## 6) Streamlit deployment

- Production Streamlit deploys from `rollback-feb8`.
- Test Streamlit deploys from `Test`.
- Secrets live in Streamlit Community Cloud.
- Production Streamlit should use production Supabase.
- Test Streamlit should use staging Supabase when validating staging/SaaS changes.

## 7) FastAPI and Next.js staging path

- `services/api` = FastAPI
- `apps/web` = Next.js

Local API commands:

```bash
pip install -r requirements.txt
pip install -r services/api/requirements.txt
uvicorn services.api.main:app --reload
```

Local Next.js commands:

```bash
cd apps/web
npm install
npm run dev
```

First production-worthy SaaS milestone:

- Read-only public club pages and leaderboards served by Next.js + FastAPI,
- backed by staging-tested Supabase read models,
- while Streamlit remains the trusted admin console.

## 8) Staging validation checklist

Data/environment checks:

- `JUPR_ENV=staging`.
- `SUPABASE_URL` points to staging Supabase.
- `SUPABASE_SERVICE_ROLE_KEY` belongs to staging.
- `SUPABASE_TEST_DATABASE_URL` is not production.
- `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` unless intentionally testing the disabled/experimental admin path.

Schema checks:

- Apply pending SQL from `supabase/migrations/` to staging.
- Verify expected tables/views exist, including:
  - `clubs`
  - `public_leaderboards`
  - `admin_role_assignments`
  - `admin_activity_log`
  - `replay_jobs`
  - player update subscription/outbox tables
  - badge queue/state tables, if testing workers

Guardrail commands:

```bash
make check-migration-sources
python scripts/check_supabase_migration_versions.py
python scripts/check_staging_environment.py --require-supabase --require-api --club-slug tres-palapas --club-id tres_palapas
```

API/web checks:

- `GET /health`
- `GET /clubs/tres-palapas`
- `GET /clubs/tres-palapas/leaderboards`
- Next.js `/`
- Next.js `/clubs/tres-palapas`
- Next.js `/clubs/tres-palapas/leaderboards`
- Confirm graceful empty/error states.

Streamlit smoke checks:

- Public home
- Public navigation/header/footer
- Leaderboards
- Player profile/search
- Tournament registration page
- Admin login
- Match uploader
- League manager
- Match log/admin page
- Player updates admin
- Replay/admin tools (only if staging DB is ready)
- Logout/session restore

Worker checks:

```bash
python -m jupr_app.workers.badge_queue_worker --club-id tres_palapas --max-total-jobs 25 --max-wall-clock-seconds 20
python -m jupr_app.workers.player_update_email_worker --club-id tres_palapas --limit 25
```

Warning:

- Only run email worker checks when staging email config is safe and cannot send unintended production emails.

## 9) Migration checklist

Staging-first process:

1. Review SQL.
2. Apply SQL to staging Supabase.
3. Verify expected columns/tables/views.
4. Deploy app branch to staging.
5. Run staging smoke tests.
6. Decide whether migration is safe for production.
7. Apply production SQL manually only after review and only when production deploy is planned.

## 10) Production promotion checklist

- Confirm staging uses staging Supabase, not production.
- Review diff from `rollback-feb8`.
- Identify schema changes and whether production SQL is required.
- Confirm Streamlit production-critical paths still work in staging.
- Keep FastAPI + Next.js production traffic disabled unless approved.
- Keep Next.js admin score entry disabled.
- Apply production SQL only when required and reviewed.
- Deploy production Streamlit.
- Smoke test production home, leaderboards, player profile/search, admin login, match uploader/match log, tournament registration.
- Record follow-up fixes.

## 11) Rollback checklist

If production has issues:

1. Pause new changes and log the issue.
2. Re-deploy `rollback-feb8` to production.
3. Verify critical paths (home, leaderboards, player search/profile, admin login, registration).
4. Confirm data integrity after rollback.
5. Record follow-up fixes before next deploy.

## 12) Local setup (practical/minimal)

If you do not run locally, skip this.

Simple Streamlit setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

API contract test command:

```bash
make api-test
```

If tests are skipped because dependencies are missing, install API requirements and rerun:

```bash
pip install -r services/api/requirements.txt
python -m pytest tests/test_api_health.py tests/test_api_contract_clubs.py tests/test_api_contract_leaderboards.py
```

## 13) MailerSend / email

- Sender name: **JUPR Notifications**
- Reply-to: **joe@juprleagues.com**
- Unsubscribe must be enforced by JUPR DB state, not only provider state.
- Staging-safe warning: do not run staging email flows unless recipients/configuration are verified safe and non-production.

## 14) Badge queue worker CLI

You can process badge evaluation queue jobs outside Streamlit Admin Tools with a worker/scheduler process.

Required environment variables:

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (preferred)
- `SUPABASE_ANON_KEY` (fallback only when service-role key is unavailable)

Safety:

- Ensure staging workers use staging Supabase keys.
- Ensure production workers use production Supabase keys.
- Never cross-wire staging worker config to production data.

Example:

```bash
python -m jupr_app.workers.badge_queue_worker \
  --club-id tres_palapas \
  --max-total-jobs 500 \
  --max-wall-clock-seconds 90
```

The command prints a JSON summary and exits nonzero when configuration is missing or queue errors hit the configured safety threshold.

## 15) Player update email worker CLI

Run player update email worker:

```bash
python -m jupr_app.workers.player_update_email_worker --club-id tres_palapas --limit 250
```

Safety:

- Only run when environment variables and email configuration are confirmed staging-safe or intentionally production-safe.

## 16) Making repo private

Before making the repo private:

1. Stabilize `Test` first.
2. Confirm Streamlit Community Cloud still has access after privacy change.
3. Confirm GitHub Actions permissions/secrets are configured for a private repo.
4. Confirm any deployment services (API/web/worker hosts) retain repository access.

## 17) What not to commit

Never commit:

- Supabase keys
- Streamlit secrets
- MailerSend API keys
- SMTP passwords
- Private player exports
- Production DB dumps
- Production schema artifacts unless explicitly intended and scrubbed
- GitHub Actions logs containing secrets

If any secret is exposed, rotate it immediately and remove it from history where possible.

## 13) Docs consistency check

- `README.txt` = operator source of truth.
- `docs/saas_staging_deploy.md` = staging deployment guardrails for FastAPI + Next.js.
- `docs/next_admin_auth_design.md` = future admin auth contract before Next admin score entry can be enabled for rated workflows.

## Manual staging smoke workflow

Run GitHub Actions workflow `Staging Smoke` via `workflow_dispatch` to perform read-only API/web checks against staging. Configure `STAGING_JUPR_API_BASE_URL` secret (required) and `STAGING_WEB_BASE_URL` (optional).

## 10) Production promotion readiness report

Before promoting `Test` to `rollback-feb8`, generate a readiness summary:

```bash
python scripts/production_readiness_report.py --base rollback-feb8 --head Test
```

Optional JSON output:

```bash
python scripts/production_readiness_report.py --base rollback-feb8 --head Test --json
```
