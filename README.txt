# JUPR Operator README (Joe)

This README is for **Joe as operator** of JUPR at Tres Palapas.
It is not a public contributor guide.

> Security rule: never paste secret values into this repo, commit history, issues, PRs, or ChatGPT.

## 1) What JUPR is

- JUPR at Tres Palapas is the official player ratings and events system.
- It is used for official player ratings and official Tres Palapas events.
- Treat JUPR data and workflows as production operations.

## 2) Branch model

Use this branch model:

- `rollback-feb8` = **production**
- `Test` = **staging**
- `professional/*` = **feature/PR branches** that merge into `Test`

Simple workflow:

1. Do work on `professional/*`.
2. Open PR into `Test`.
3. Validate in staging.
4. Promote to `rollback-feb8` only after staging is stable.

## 3) Streamlit deployment

- Production Streamlit app deploys from `rollback-feb8`.
- Test Streamlit app deploys from `Test`.
- Secrets live in **Streamlit Community Cloud**.
- Never paste secrets into the repo or into ChatGPT.

## 4) Supabase

- There is currently one **production Supabase project**.
- There is **no staging Supabase project yet**.
- SQL migrations are often manually pasted into Supabase SQL editor.
- Prefer `supabase/migrations/` as the canonical migration folder.

## 5) MailerSend / email

- Sender name: **JUPR Notifications**
- Reply-to: **joe@juprleagues.com**
- Unsubscribe must be enforced by JUPR DB state, not only provider state.

## 6) Local setup (practical/minimal)

If you do not run locally, skip this.

1. Check Streamlit app settings and record the Python version used there.
2. Install that same Python version locally.
3. Create and activate a virtual environment.
4. Install dependencies from `requirements.txt`.
5. Run the app locally for quick smoke checks.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 7) Testing checklist before deploy

Before any deploy, smoke test:

- Public home
- Leaderboards
- Player profile/search
- Tournament registration page
- Admin login
- Match log/admin page smoke test

## 8) Migration checklist

Detailed policy: see `docs/migrations.md`.

1. Review SQL.
2. Apply SQL in Supabase SQL editor.
3. Verify expected columns/tables.
4. Deploy app branch.
5. Smoke test.

Guardrail command:

```bash
make check-migration-sources
```

## 9) Rollback checklist

If production has issues:

1. Pause new changes and log the issue.
2. Re-deploy `rollback-feb8` to production.
3. Verify critical paths (home, leaderboards, player search/profile, admin login, registration).
4. Confirm data integrity after rollback.
5. Record follow-up fixes before next deploy.

## 10) Making repo private

1. Stabilize `Test` first.
2. Confirm Streamlit GitHub access to the private repo.
3. Then make repo private.

## 11) What not to commit

Never commit:

- Supabase keys
- Streamlit secrets
- MailerSend API keys
- Private player exports
- Production DB dumps

If any secret is exposed, rotate it immediately and remove it from history where possible.

## 12) Badge queue worker CLI

You can process badge evaluation queue jobs outside Streamlit Admin Tools with a worker/scheduler process.

Required environment variables:
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (preferred)
- `SUPABASE_ANON_KEY` (fallback only when service role key is unavailable)

Example:

```bash
python -m jupr_app.workers.badge_queue_worker \
  --club-id tres_palapas \
  --max-total-jobs 500 \
  --max-wall-clock-seconds 90
```

The command prints a JSON summary and exits nonzero when configuration is missing or queue errors hit the configured safety threshold.
