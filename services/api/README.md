# JUPR API (FastAPI skeleton)

Minimal read-only API scaffold for future service integration.

## Run locally

```bash
pip install -r services/api/requirements.txt
uvicorn services.api.main:app --reload
```

## Environment variables

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (preferred for server-side API)
- `SUPABASE_ANON_KEY` (read-only local fallback)
- `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY` (`1` / `true` / `yes` to enable, disabled by default)

## Endpoints

- `GET /health`
- `GET /clubs/{club_slug}`
- `GET /clubs/{club_slug}/leaderboards?league_name=...`

`GET /clubs/{club_slug}` is backed by `public.clubs` (slug-first lookup with id fallback) and returns a normalized public club contract (`id`, `slug`, `name`, `tagline`, `support_email`, `public_base_url`, `logo_url`, `primary_color`, `is_active`). Tres Palapas default slug is `tres-palapas`.

The leaderboard endpoint delegates to `jupr_app.services.leaderboard_service.get_public_leaderboard` and only returns public-safe fields.
- `POST /admin/clubs/{club_id}/matches/batch`

## Admin score-entry guard (experimental + temporary)

`POST /admin/clubs/{club_id}/matches/batch` is **disabled by default**.

When disabled, the endpoint returns:

`Next admin score entry is disabled. Use Streamlit admin until Supabase JWT role auth is implemented.`

To enable in staging experiments only, set `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1` (or `true` / `yes`).

When enabled, this route still uses a **server token placeholder guard** and is not production-grade auth.

Required headers:
- `x-admin-token: <JUPR_ADMIN_API_TOKEN>`
- `x-admin-permission: enter_scores`

This endpoint is intentionally structured so Supabase JWT and role-based authorization can replace the placeholder guard later without changing the route contract.

Auth design reference for future replacement of the temporary guard:
- `docs/next_admin_auth_design.md`


## Staging deployment guardrails

This API path is currently **staging-only** for the new SaaS rollout.

- Set `JUPR_ENV=staging` in staging runtime.
- Use a **staging Supabase project** via `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.
- Keep `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` until explicitly approved.
- Do **not** point this staging API at production Supabase.

See `docs/saas_staging_deploy.md` for the full checklist and rollout constraints.
