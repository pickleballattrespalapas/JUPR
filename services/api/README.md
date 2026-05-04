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

## Endpoints

- `GET /health`
- `GET /clubs/{club_slug}`
- `GET /clubs/{club_slug}/leaderboards?league_name=...`

`GET /clubs/{club_slug}` is backed by `public.clubs` (slug-first lookup with id fallback) and returns a normalized public club contract (`id`, `slug`, `name`, `tagline`, `support_email`, `public_base_url`, `logo_url`, `primary_color`, `is_active`). Tres Palapas default slug is `tres-palapas`.

The leaderboard endpoint delegates to `jupr_app.services.leaderboard_service.get_public_leaderboard` and only returns public-safe fields.
- `POST /admin/clubs/{club_id}/matches/batch`

## Admin score-entry guard (v1 placeholder)

`POST /admin/clubs/{club_id}/matches/batch` currently uses a **server token placeholder guard** and is not production-grade auth.

Required headers:
- `x-admin-token: <JUPR_ADMIN_API_TOKEN>`
- `x-admin-permission: enter_scores`

This endpoint is intentionally structured so Supabase JWT and role-based authorization can replace the placeholder guard later without changing the route contract.
