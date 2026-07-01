# Fly API deployment checklist

This checklist deploys the FastAPI backend as an always-on Fly app at `api.juprleagues.com`.

This project supports an online-only deploy path through GitHub Actions. You do not need to run commands locally.

The frontend remains on Vercel. Do not put server-only Supabase secrets in Vercel.

## Repo files

The deployment is defined by these root-level files:

- `Dockerfile.api`
- `fly.toml`
- `.dockerignore`
- `.github/workflows/fly_api_deploy.yml`

The Docker build context is the repository root, and the container starts:

```bash
uvicorn services.api.main:app --host 0.0.0.0 --port $PORT --proxy-headers
```

`fly.toml` defaults to:

```toml
app = "juprleagues-api"
primary_region = "dfw"
```

The GitHub Actions workflow lets you override the app name and primary region at run time.

## 1. Merge the Fly deploy PR

Merge the PR that adds the Fly deployment files into `rollback-feb8`.

## 2. Create a Fly API token in the Fly dashboard

In the Fly dashboard, create an org-scoped deploy token for the organization that will own the API app.

Use the narrowest token that can create/manage the app and deploy it. If the app already exists, an app-scoped deploy token is enough. If the GitHub workflow needs to create the Fly app for you, use an org-scoped deploy token.

Copy the token value immediately; you will paste it into GitHub as a repository secret.

## 3. Add GitHub Actions secrets

In GitHub, open the repo and go to:

`Settings` -> `Secrets and variables` -> `Actions` -> `New repository secret`

Add these repository secrets:

```text
FLY_API_TOKEN=<Fly org-scoped deploy token>
SUPABASE_URL=<production Supabase project URL>
SUPABASE_SERVICE_ROLE_KEY=<production Supabase service role key>
SUPABASE_ANON_KEY=<production Supabase anon key>
```

Do not add `SUPABASE_SERVICE_ROLE_KEY` to Vercel.

## 4. Run the online Fly deploy workflow

In GitHub, open:

`Actions` -> `Deploy FastAPI backend to Fly` -> `Run workflow`

Use these inputs:

```text
Branch: rollback-feb8
app_name: juprleagues-api
primary_region: dfw
fly_org: <blank unless Fly requires your org slug>
custom_domain: api.juprleagues.com
```

The workflow will:

1. Validate required GitHub secrets.
2. Apply the selected app name and region to `fly.toml` inside the workflow run.
3. Create the Fly app if it does not already exist.
4. Stage production Supabase runtime secrets in Fly.
5. Validate the Fly config.
6. Deploy the API with Fly remote build.
7. Attach `api.juprleagues.com` and print DNS setup instructions.
8. Smoke test `https://<app_name>.fly.dev/health`.

## 5. Smoke test the Fly hostname

After the workflow completes, open these in a browser:

```text
https://juprleagues-api.fly.dev/health
https://juprleagues-api.fly.dev/clubs/tres-palapas
https://juprleagues-api.fly.dev/clubs/tres-palapas/leaderboards
```

If you changed `app_name`, replace `juprleagues-api` in the `.fly.dev` hostname.

## 6. Connect `api.juprleagues.com` in GoDaddy

Open the completed GitHub Actions run logs and expand the step:

`Attach custom API domain and show DNS setup`

Use the DNS record Fly prints there, or open the Fly dashboard:

`Apps` -> `juprleagues-api` -> `Certificates` -> `api.juprleagues.com`

For a CNAME record in GoDaddy, the host/name is usually:

```text
api
```

not the full `api.juprleagues.com`.

The value/target should be the Fly CNAME target shown in the GitHub Actions logs or Fly dashboard.

## 7. Check the custom API domain

Once DNS propagates and the Fly certificate is issued, open these in a browser:

```text
https://api.juprleagues.com/health
https://api.juprleagues.com/clubs/tres-palapas
https://api.juprleagues.com/clubs/tres-palapas/leaderboards
```

## 8. Point Vercel to the API and redeploy

In Vercel, set this production environment variable for the web app:

```text
JUPR_API_BASE_URL=https://api.juprleagues.com
```

`NEXT_PUBLIC_JUPR_API_BASE_URL` also works because the web app checks both names, but prefer `JUPR_API_BASE_URL` for the production server-rendered app unless browser-side code explicitly needs the URL.

Do not put `SUPABASE_SERVICE_ROLE_KEY` in Vercel.

Then redeploy Vercel production from the Vercel dashboard.

## 9. Full live-stack smoke test

Open these in a browser:

```text
https://juprleagues.com/
https://juprleagues.com/clubs/tres-palapas
https://juprleagues.com/clubs/tres-palapas/leaderboards

https://api.juprleagues.com/health
https://api.juprleagues.com/clubs/tres-palapas
https://api.juprleagues.com/clubs/tres-palapas/leaderboards
```

## Troubleshooting

If the GitHub Actions deploy fails before creating the app, the Fly token probably does not have enough scope. Use an org-scoped deploy token or create the app manually in the Fly dashboard and retry with an app-scoped deploy token.

If the deploy succeeds but requests fail, check that the app is listening on `0.0.0.0:$PORT` and that `fly.toml` uses the same `internal_port` as the container.

If browser requests fail but direct URL tests work, check CORS. The backend env var is already set in `fly.toml`:

```text
JUPR_ALLOWED_ORIGINS=https://juprleagues.com,https://www.juprleagues.com
```

If health checks fail, make sure `/health` returns 200 without auth or redirects.
