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
