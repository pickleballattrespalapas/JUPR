# Fly API deployment checklist

This checklist deploys the FastAPI backend as an always-on Fly app at `api.juprleagues.com`.

The frontend remains on Vercel. Do not put server-only Supabase secrets in Vercel.

## Repo files

The deployment is defined by these root-level files:

- `Dockerfile.api`
- `fly.toml`
- `.dockerignore`

The Docker build context is the repository root, and the container starts:

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

## 1. Install and log in to Fly

```bash
brew install flyctl
fly auth login
```

If you are not on macOS, use Fly's installer instead of Homebrew.

## 2. Check the app name and region

`fly.toml` defaults to:

```toml
app = "juprleagues-api"
primary_region = "dfw"
```

Before the first deploy, change `app` if that Fly app name is unavailable. If production Supabase is in another region, set `primary_region` to the closest Fly region to Supabase.

## 3. Create the Fly app

From the repository root:

```bash
fly apps create juprleagues-api
```

If you changed the `app` value in `fly.toml`, use that same name in this command.

## 4. Add production backend secrets

Use production Supabase values here. Do not commit these values and do not paste them into GitHub issues, PR comments, or frontend env vars.

```bash
cat <<'EOF' | fly secrets import -a juprleagues-api
SUPABASE_URL=https://YOUR_PROJECT_REF.supabase.co
SUPABASE_SERVICE_ROLE_KEY=YOUR_PRODUCTION_SERVICE_ROLE_KEY
SUPABASE_ANON_KEY=YOUR_PRODUCTION_ANON_KEY
EOF
```

Confirm only the secret names:

```bash
fly secrets list -a juprleagues-api
```

## 5. Deploy

```bash
fly config validate -a juprleagues-api
fly deploy -a juprleagues-api
fly status -a juprleagues-api
fly logs -a juprleagues-api
```

## 6. Smoke test the Fly hostname

```bash
curl -i https://juprleagues-api.fly.dev/health
curl -i https://juprleagues-api.fly.dev/clubs/tres-palapas
curl -i https://juprleagues-api.fly.dev/clubs/tres-palapas/leaderboards
```

## 7. Attach `api.juprleagues.com`

```bash
fly certs add api.juprleagues.com -a juprleagues-api
fly certs setup api.juprleagues.com -a juprleagues-api
```

Add the DNS record Fly shows you in GoDaddy. For a CNAME record, the GoDaddy host/name is usually `api`, not the full domain.

Then check certificate status:

```bash
fly certs check api.juprleagues.com -a juprleagues-api
```

## 8. Smoke test the custom API domain

```bash
curl -i https://api.juprleagues.com/health
curl -i https://api.juprleagues.com/clubs/tres-palapas
curl -i https://api.juprleagues.com/clubs/tres-palapas/leaderboards
```

## 9. Point Vercel to the API and redeploy

In Vercel, set this production environment variable for the web app:

```bash
JUPR_API_BASE_URL=https://api.juprleagues.com
```

`NEXT_PUBLIC_JUPR_API_BASE_URL` also works because the web app checks both names, but prefer `JUPR_API_BASE_URL` for the production server-rendered app unless browser-side code explicitly needs the URL.

Then redeploy Vercel production.

## 10. Full live-stack smoke test

```bash
curl -i https://juprleagues.com/
curl -i https://juprleagues.com/clubs/tres-palapas
curl -i https://juprleagues.com/clubs/tres-palapas/leaderboards

curl -i https://api.juprleagues.com/health
curl -i https://api.juprleagues.com/clubs/tres-palapas
curl -i https://api.juprleagues.com/clubs/tres-palapas/leaderboards
```

## Troubleshooting

If `fly deploy` succeeds but requests fail, check that the app is listening on `0.0.0.0:$PORT` and that `fly.toml` uses the same `internal_port` as the container.

If browser requests fail but `curl` works, check CORS. The backend env var is:

```bash
JUPR_ALLOWED_ORIGINS=https://juprleagues.com,https://www.juprleagues.com
```

If health checks fail, make sure `/health` returns 200 without auth or redirects.
