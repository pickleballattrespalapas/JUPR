# JUPR staging agent guide

This file is the repository source of truth for coding agents continuing work in a new chat.

## Environment boundaries

- `staging` is the canonical integration and staging-deployment branch. Open feature pull requests explicitly against `staging`.
- `rollback-feb8` is the production branch. Do not merge, push, dispatch, deploy, or apply migrations to production unless Joe separately and explicitly authorizes that exact production action.
- The only permitted Supabase project for staging work is `sijpxjxvdtrehmqvirfi`.
- The production Supabase project `dnoockbwfenunhcibwfn` is forbidden during staging work.
- The only permitted staging API app and origin are `juprleagues-api-staging` and `https://juprleagues-api-staging.fly.dev`.
- The canonical staging web alias is `https://jupr-git-staging-pickleballattrespalapas1.vercel.app`.
- Do not target the production Fly app `juprleagues-api`, `https://api.juprleagues.com`, `https://pickleballclubsandwich.com`, or another configurable origin from staging automation.
- Never print, commit, upload, or place credentials in a handoff, issue, pull request, log, or chat.

## Current staging posture

- Canonical staging is intentionally persistent-open for acceptance testing: `JUPR_STAGING_WRITE_WAVE=open`.
- The checked-in `fly.staging.toml` remains `none` as a fail-safe source configuration. The staging deploy workflow projects the reviewed persistent-open flags at deploy time.
- Email must remain `JUPR_EMAIL_MODE=dry_run`; live player-update email and every production write override remain disabled.
- `.github/workflows/staging-write-recovery.yml` is the explicit emergency stop. Do not dispatch it unless Joe asks to close staging writes.
- Staging authorization does not authorize production access or mutation.

## Standard continuation workflow

1. Inspect `git status`, the current branch, `origin/staging`, and existing user changes before editing.
2. Work on a short-lived feature branch or worktree and preserve unrelated changes.
3. Run the focused Python tests, Next component/type/build checks, migration guards, and `git diff --check` appropriate to the change.
4. Put schema changes only in `supabase/migrations/`. Do not use `scripts/db_migrate.sh`; it targets the legacy `migrations/` directory.
5. Apply reviewed migrations only to the exact staging Supabase project, then open and merge the reviewed pull request into `staging`.
6. A merge to `staging` triggers the isolated Fly deploy and the Git-connected Vercel preview. The Fly workflow waits for both surfaces to attest the same SHA and then creates the handoff artifact.
7. Treat staging as ready for manual testing only when the exact staging SHA has a successful `Deploy FastAPI staging to Fly` run and its `staging-handoff-<sha>` artifact reports `ready_for_manual_testing`.

## Cross-chat handoff

The latest successful staging deployment contains:

- `staging-handoff.json` for machine-readable continuation;
- `staging-handoff.md` for a short human summary.

Both files bind the candidate SHA to the Fly image, Vercel deployment, staging Supabase project, persistent-open posture, and read-only smoke results. A later chat must compare the artifact SHA with current `origin/staging`; never combine evidence from different candidates.
