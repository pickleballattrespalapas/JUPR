# Production FastAPI deployment policy

The production FastAPI backend is the existing Fly app
`juprleagues-api`. Production deployment remains a separate finish line from
staging acceptance. Do not run this workflow until one exact staging candidate
has been formally accepted and merged to the canonical production branch
(`rollback-feb8` for this reviewed workflow version).

The protected workflow is
`.github/workflows/fly_api_deploy.yml` (`Deploy FastAPI production to Fly`).
It is manual-only and does not create Fly apps, change DNS, add certificates,
apply migrations, enable business-data writes, or send customer email.

## Protected GitHub configuration

Configure a GitHub environment named `production` with required reviewers.
Store the exact production Supabase ref as an environment variable:

```text
PRODUCTION_SUPABASE_PROJECT_REF=<20-character production project ref>
PRODUCTION_MIGRATION_LEDGER_HEAD=<reviewed connector ledger head>
```

Configure these production secrets:

```text
FLY_API_TOKEN=<app-scoped deploy token for juprleagues-api>
FLY_SSH_TOKEN=<short-lived app-scoped SSH token for juprleagues-api>
SUPABASE_URL=https://<production-ref>.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<production service-role key>
SUPABASE_ANON_KEY=<production anon key>
SUPABASE_PROD_DATABASE_URL=<production direct or transaction-pooler URL>
```

Use an app-scoped Fly deploy token for app changes and a separately scoped,
short-lived Fly SSH token for the no-write runtime attestation. The workflow
has no reason to create or manage any other app. Rotate both tokens on the
release schedule.

`SUPABASE_PROD_DATABASE_URL` is used only for read-only migration-ledger
verification in GitHub Actions. It is not staged as a Fly runtime secret.
`SUPABASE_SERVICE_ROLE_KEY` must never be added to Vercel.

## Dispatch contract

Run the workflow from the protected `rollback-feb8` branch with:

```text
candidate_sha: <exact 40-character lowercase SHA at the protected rollback-feb8 HEAD>
confirmation: DEPLOY PRODUCTION API
```

The workflow rejects the run unless all four identities are identical:

1. checked-out `HEAD`;
2. GitHub's workflow `GITHUB_SHA`;
3. the current `origin/rollback-feb8` tip;
4. the supplied `candidate_sha`.

The app, region, API origin, web origin, and production branch are constants,
not operator-selectable inputs. A production-branch rename requires a separate
reviewed workflow change; a mutable repository default cannot silently retarget
a production deployment.

## Fail-closed verification

Before deployment, the workflow:

1. validates every required protected setting without printing secret values;
2. proves both Supabase URLs target the protected production project and
   explicitly rejects the staging project;
3. validates `fly.toml` against the exact app, region, CORS allowlist, and
   read-only feature projection;
4. requires the existing `juprleagues-api` Fly app instead of creating one;
5. captures the current runtime SHA, image-build SHA, configured Fly image,
   image digest, and immutable digest reference as a 90-day
   `production-api-rollback-<run-id>` artifact before any mutation;
6. validates `supabase_migrations.schema_migrations` by the reviewed logical
   name profile in `config/production_migration_contract.json`, probes the
   schema-only registration hotfix contract, and requires the remote head to
   equal the protected reviewed `PRODUCTION_MIGRATION_LEDGER_HEAD`;
7. rejects any existing Fly secret with a `Staged`, `Partial`, or unknown
   deployment status;
8. activates `write_wave=none`, the read-only production policy, and the
   disabled feature projection on the current image, then verifies those
   runtime values over an app-scoped SSH session before changing the image;
9. deploys the candidate without staging secrets. Code identity comes from the
   Git SHA baked into the image, not from a mutable runtime secret.

Supabase connector-assigned ledger versions are not repository filename
timestamps. For example, the reconciled staging ledger records
`server_only_data_api_lockdown` at `20260720033650`, while the canonical
repository filename begins `20260719155515`. Treating those numbers as equal
would incorrectly request duplicate migration application.

The reviewed profile therefore requires the exact 38 logical ledger names
known from the reconciled environment and rejects missing, duplicate, malformed,
or additional names. Its repository-content fingerprint hashes every canonical
migration filename and its SQL bytes, forcing an explicit profile review when
either the inventory or reviewed SQL changes. The
idempotent
`tournament_registrations_player_id_postgrest_reload` migration is deliberately
schema-contract-only because connector application did not create a same-name
ledger row. The workflow directly requires:

- `tournament_registrations.player_id`;
- `idx_tournament_registrations_player_id`;
- `uq_tournament_registrations_tournament_player`;
- zero duplicate non-null `(tournament_id, player_id)` groups.

The logical-name profile, schema probes, and protected reviewed head must all
pass before and after deployment. The workflow never applies or repairs schema.

The initial production promotion policy is deliberately read-only:

```text
JUPR_ENV=production
JUPR_PRODUCTION_WRITE_POLICY=read_only
JUPR_STAGING_WRITE_WAVE=none
JUPR_EMAIL_MODE=dry_run
```

Every known Next administrative and business-write feature flag is explicitly
`0`. FastAPI middleware denies unsafe production requests unless a later,
separately reviewed release changes the production write policy to `enabled`.
Do not make that change as part of the initial public deployment.

After deployment, the workflow fails unless:

- Fly `/health` and `api.juprleagues.com/health` return the same identity;
- runtime SHA and image-build SHA equal the candidate;
- every nonterminal Fly machine resolves to one exact production image and
  immutable digest, and the health image ref matches that identity;
- environment, Fly app, Supabase project, JWT project, web origin, migration
  profile/contract/head, audit prerequisites, email mode, write policy, and
  every feature flag match the exact expected projection;
- required Fly secret names are present (values remain unavailable);
- `/openapi.json` is valid and includes `/health`;
- a public leaderboard request completes the required Supabase table reads;
- the migration ledger still has the verified logical-name profile and the
  protected reviewed connector head;
- CORS preflight succeeds for exactly the four approved public origins on both
  the Fly hostname and custom API hostname, while an unapproved origin receives
  no allow-origin header;
- the final health, nonterminal-machine, and secret readbacks still match the
  complete project/JWT/migration/CORS/flag/audit/email/no-write projection.

After the pre-deploy no-write projection has run, an `always()` finalizer
remotely verifies the complete read-only projection. The workflow never stages
secrets. If the initial active safe-bundle rollout fails, the finalizer permits
only pending names inside that exact bundle, overwrites them with the exact
safe values, and retries convergence; an external pending name remains a hard
stop. If image deployment was attempted and deployment or any acceptance check
fails, the finalizer redeploys the snapshot's immutable digest with the
captured pre-deploy Fly config and requires the final image and baked SHA to
match that snapshot exactly. If image deployment was skipped, it does not
create an unnecessary release. If every acceptance check succeeds, final
identity must remain bound to the candidate SHA.

Any mismatch returns a non-zero workflow result. There is no `|| true`
production bypass.

The snapshot step deliberately rejects a legacy production image that cannot
report both its runtime Git SHA and baked image Git SHA. Do not guess that
identity. The first transition from such a legacy image requires a separately
reviewed production-owner bootstrap decision before this workflow can be used.

## Current production readiness blockers (2026-07-24)

Read-only reconciliation of production Supabase project
`dnoockbwfenunhcibwfn` found exactly one migration-ledger row:
`20250220 badges_v1`. The reviewed deployment contract requires the exact
38-name Next/FastAPI ledger profile and its reviewed head. Although the
registration `player_id` column/index/unique-index probes currently pass and
no duplicate tournament/player groups were found, that schema shape is not a
substitute for ledger provenance. Do not set
`PRODUCTION_MIGRATION_LEDGER_HEAD` to the target head or run this workflow until
an owner-approved schema/ledger promotion has reconciled production.

The production Security Advisor also reports release-blocking findings:
public tables without RLS, a security-definer `league_settings` view,
sensitive columns exposed through the API, executable `SECURITY DEFINER`
functions available to public/signed-in roles, and leaked-password protection
disabled. Resolve and rerun the advisor under an owner-reviewed plan; do not
dismiss or bypass these checks. See Supabase's
[Security Advisors](https://supabase.com/docs/guides/database/database-advisors),
[API security guidance](https://supabase.com/docs/guides/api/securing-your-api),
and [password security guidance](https://supabase.com/docs/guides/auth/password-security).

Performance Advisor findings remain a reviewed, no-DDL backlog: eight
duplicate-index groups (`badge_eval_queue`, `ladder_player_flags`,
`ladder_roster`, `league_ratings`, `leagues_metadata`, `matches`,
`player_badges`, and `players`), plus unindexed foreign keys, multiple
permissive policies, and absolute Auth connection allocation. Do not
automatically drop indexes; constraint ownership and operational compatibility
need an owner-reviewed plan.

## Current public origins

The canonical web origin is:

```text
https://pickleballclubsandwich.com
```

The exact CORS allowlist is:

```text
https://juprleagues.com
https://www.juprleagues.com
https://pickleballclubsandwich.com
https://www.pickleballclubsandwich.com
```

Production does not allow a CORS regex, Vercel preview origins, localhost, or
operator-supplied domains.

## Rollback preparation

Download the `production-api-rollback-<run-id>` artifact and attach it to the
release ticket. It contains the exact previous app, runtime/image SHA,
configured image ref, digest, immutable image ref, captured pre-deploy Fly
config, and its SHA-256 fingerprint; it contains no secret values. Preserve
both files as the rollback identity.

If rollback is required:

1. stop new production actions and keep
   `JUPR_PRODUCTION_WRITE_POLICY=read_only` and
   `JUPR_STAGING_WRITE_WAVE=none`;
2. select the previous accepted immutable image and its baked commit SHA as one
   unit;
3. reject any staged, partial, or unknown Fly secret deployment before
   restoring the previous image;
4. do not roll the database backward automatically;
5. re-run health, OpenAPI, Fly image, Supabase project, migration-head, CORS,
   and final read-only identity checks;
6. keep Streamlit available as the administrative fallback until the incident
   is closed.

If a schema change is not backward compatible with the previous image, stop
and use a reviewed forward-fix plan. Do not improvise a production schema
rollback.

From an approved production-owner shell with `flyctl`, `jq`, the repository,
and an app-scoped deploy token configured, the image rollback itself is:

```bash
JUPR_ROLLBACK_FILE=/path/to/production-rollback-snapshot.json
JUPR_ROLLBACK_CONFIG=/path/to/captured-predeploy-fly.toml
JUPR_ROLLBACK_SHA="$(jq -er '.image_build_git_sha' "$JUPR_ROLLBACK_FILE")"
JUPR_ROLLBACK_IMAGE="$(jq -er '.fly_immutable_image_ref' "$JUPR_ROLLBACK_FILE")"
JUPR_ROLLBACK_CONFIG_SHA="$(jq -er '.fly_config_sha256' "$JUPR_ROLLBACK_FILE")"

[[ "$JUPR_ROLLBACK_SHA" =~ ^[0-9a-f]{40}$ ]]
[[ "$JUPR_ROLLBACK_IMAGE" =~ ^registry\.fly\.io/juprleagues-api@sha256:[0-9a-f]{64}$ ]]
[[ "$(sha256sum "$JUPR_ROLLBACK_CONFIG" | awk '{print $1}')" == "$JUPR_ROLLBACK_CONFIG_SHA" ]]

flyctl secrets list --app juprleagues-api --json > /tmp/jupr-prod-secrets.json
python scripts/deployment_verifier.py secrets \
  --no-pending-only \
  --fly-secrets-json /tmp/jupr-prod-secrets.json
flyctl deploy \
  --app juprleagues-api \
  --config "$JUPR_ROLLBACK_CONFIG" \
  --image "$JUPR_ROLLBACK_IMAGE"

curl --fail --silent --show-error \
  https://juprleagues-api.fly.dev/health |
  jq -e \
    --arg sha "$JUPR_ROLLBACK_SHA" \
    --arg image "$JUPR_ROLLBACK_IMAGE" \
    '.git_commit_sha == $sha
     and .image_build_git_sha == $sha
     and .fly_image_ref == $image
     and .write_wave == "none"
     and .business_data_write_wave_active == false
     and .production_business_write_policy == "read_only"'
```

Then run the same OpenAPI, custom-domain identity, database-backed read,
migration-profile, active-machine image, CORS, and final no-write checks as the
deployment workflow before closing rollback.
