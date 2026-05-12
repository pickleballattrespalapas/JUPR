# JUPR API (FastAPI skeleton)

Minimal read-only API scaffold for future service integration.

## Run locally

```bash
pip install -r requirements.txt
pip install -r services/api/requirements.txt
uvicorn services.api.main:app --reload
```

## Environment variables

- `JUPR_ENV=staging`
- `SUPABASE_URL` (must be the staging Supabase project URL for staging runtime)
- `SUPABASE_SERVICE_ROLE_KEY` (preferred for server-side API; use staging credentials in staging)
- `SUPABASE_ANON_KEY` (read-only local fallback)
- `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` (leave disabled by default)

## Endpoints

- `GET /health`
- `GET /clubs/{club_slug}`
- `GET /clubs/{club_slug}/leaderboards?league_name=...`

`GET /clubs/{club_slug}` is backed by `public.clubs` (slug-first lookup with id fallback) and returns a normalized public club contract (`id`, `slug`, `name`, `tagline`, `support_email`, `public_base_url`, `logo_url`, `primary_color`, `is_active`). Tres Palapas default slug is `tres-palapas`.

The leaderboard endpoint delegates to `jupr_app.services.leaderboard_service.get_public_leaderboard` and only returns public-safe fields.
- `POST /admin/clubs/{club_id}/matches/batch`

## Admin score-entry guard (experimental + temporary)

Admin score-entry is **experimental and disabled by default**. Keep it disabled unless running explicit staging-only experiments.

`POST /admin/clubs/{club_id}/matches/batch` is **disabled by default**.

When disabled, the endpoint returns:

`Next admin score entry is disabled. Use Streamlit admin until Supabase JWT role auth is implemented.`

To enable in staging experiments only, set `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=1` (or `true` / `yes`).

When enabled, this route requires Supabase JWT Bearer auth and role-based authorization for `enter_scores`.

Auth design reference for future replacement of the temporary guard:
- `docs/next_admin_auth_design.md`


## Staging deployment guardrails

This API path is currently **staging-only** for the new SaaS rollout.

- Set `JUPR_ENV=staging` in staging runtime.
- Use a **staging Supabase project** via `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.
- Keep `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY=0` until explicitly approved.
- Do **not** point this staging API at production Supabase.

See `docs/saas_staging_deploy.md` for the full checklist and rollout constraints.
Also see `README.txt` (operator branch/promotion model) and `docs/saas_status.md` (current SaaS phase/status).

When running staging API deployments, credentials must come from the staging Supabase project only.


## Admin JWT auth (Supabase)

- `POST /admin/clubs/{club_id}/matches/batch` requires `Authorization: Bearer <access_token>`.
- Feature flag `JUPR_ENABLE_NEXT_ADMIN_SCORE_ENTRY` remains disabled by default.
- When disabled, endpoint returns `403` before auth and writes.
- JWT verification config (server-side only):
  - Secret mode (default): `SUPABASE_JWT_SECRET` (+ optional `SUPABASE_JWT_AUDIENCE`, default `authenticated`).
  - JWKS mode: `JUPR_SUPABASE_JWT_MODE=jwks` + `SUPABASE_JWKS_URL` (+ optional audience).
- Do not use service-role or JWT secrets in browser/client code.

## Admin audit logging

- Successful admin match-batch writes attempt to write `admin_activity_log` records with actor + club attribution.
- Denied authenticated writes are flagged for review in audit logs when safe to do so.
- `JUPR_REQUIRE_API_AUDIT_LOG=1` enables strict mode: write requests fail if audit logging cannot be recorded.
- Default behavior degrades gracefully when audit table migration is missing (staging-safe).
- Audit payloads include summary metadata (`source_page`, `source_client`) and do not include raw bearer tokens or secrets.
