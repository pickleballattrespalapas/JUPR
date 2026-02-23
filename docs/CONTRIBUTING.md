# Contributing to JUPR

## Branching
main = production
rebuild = experimental

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## Migrations
Located in /migrations
Run in Supabase SQL editor.

## Schema Drift CI
On every pull request, `.github/workflows/schema-drift.yml` runs `python3 scripts/schema_migration_audit.py`.
Audit exit code `2` is treated as a warning in CI (findings are logged but do not hard-fail the job).
If `supabase/config.toml` exists, CI also starts a local Supabase stack and runs `supabase db diff` in check mode.
If drift is detected, run `supabase db diff -f <name>` locally and commit the generated migration.

## Stability Rules
- Guard against None
- Never block UI with heavy compute
- Use caching carefully

## Staging Deploy Secrets (GitHub Actions)
The `.github/workflows/deploy_fly_staging.yml` workflow expects these GitHub repository secrets:
- `PUBLIC_BASE_URL`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_ADMIN_PASSWORD`
- `SUPABASE_ADMIN_SESSION_SECRET`
- `FLY_API_TOKEN`
