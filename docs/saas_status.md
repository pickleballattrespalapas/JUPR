# SaaS Implementation Status

This document is the durable source of truth for the Pickleball Club Sandwich
(JUPR) migration so Joe and Codex stay aligned without relying on memory.

## Current status summary

- Streamlit production is active and remains the fallback admin runtime.
- `staging` is the canonical staging integration/deployment branch. `Test` is legacy/deprecated and must not receive new staging PRs or evidence.
- Staging Supabase exists and is the non-production data environment for staging validation.
- FastAPI + Next.js are staging-first for public traffic, with closed-club production-write pilot mode now defined for staff workflows.
- Next.js public routes now cover the read-only product spine: club home, leaderboards, league results, Badge Codex, Challenge Ladder, Weekly Recap, Match Explorer, players, player profiles, matches, match detail, JUPR Live, ratings explainer, FAQ, privacy, terms, support/contact, data-correction instructions, and public tournament registration intake.
- Next `/admin` is now the operations migration cockpit backed by FastAPI `GET /admin/operations/status`.
- `docs/next_streamlit_parity_matrix.md` is the control board for reaching 100% Streamlit workflow parity on Next/Vercel/FastAPI.
- Public/staff smoke tooling exists for staging FastAPI + Vercel validation.
- The last fully reconciled pre-hardening checkpoint is Git SHA
  `eab384545c493f145af383c8e26d8bf97686ab21`: Vercel deployment
  `dpl_FTmDVFSdftMmAj4sMwAHrvQpVYLE`, Fly image
  `registry.fly.io/juprleagues-api-staging:deployment-01KY5SVX80SHEZMN7GX37NASRP`,
  and staging Supabase project `sijpxjxvdtrehmqvirfi`. Vercel and Fly served that
  exact SHA, Fly was healthy at `write_wave=none`, and canonical Staging Smoke run
  `29957623653` passed all 56 strict browser checks. PR `#1022` is the
  candidate-evidence PR; its identity must be replaced as one unit after the next
  exact candidate is deployed.
- The later live pre-evidence candidate
  `1b1c66b9ff1b1ea2f90ad0491be87921ec1524d8` was identity-aligned across
  Vercel and Fly, but Parity Final Evidence runs `30126824035` and
  `30126850714` failed environment preflight before identity/browser execution.
  Their artifacts contain no browser invocations or writes and are not acceptance
  evidence. The replacement candidate keeps communications reads available at
  `write_wave=none`, independently guards every mutation, mints a fresh masked
  staging-admin JWT per authenticated workflow run, and validates exact disposable
  fixtures instead of depending on stale stored tokens or IDs.
- All 37 staging migration-ledger entries are applied through
  `20260720123402_baseline_worker_run_log`. All 29 implementation orders have
  landed, with roughly 70 Next page components covering the 47 tracked Streamlit
  surfaces. The 45 matrix rows still marked `Partial` remain so until their
  candidate-bound manual evidence is complete.
- Baseline support intake/deduplication and admin dismissal were proven in
  staging, then Fly was restored to `write_wave=none` in successful GitHub run
  `29795882496`; the all-false controlled-write projection is the steady state.
- Next admin score entry remains disabled by default.
- No formal Next/FastAPI production candidate has been accepted or cut over.
  The apex domains currently resolve to a Vercel site, but that fact is not
  acceptance evidence; Streamlit remains the production/admin fallback until the
  cutover gates are met.

## Status table

| Area | Status | Current implementation | Next action | Risk level |
|---|---|---|---|---|
| Staging Supabase | Done | Staging Supabase project exists and is used for SaaS validation. | Keep environment guards in place and apply schema changes staging-first. | Medium |
| Streamlit production | Done | Streamlit remains the active fallback admin/runtime surface. | Keep available while Next admin workflows are piloted one at a time. | Medium |
| Streamlit-to-Next parity control | In progress | `docs/next_streamlit_parity_matrix.md` inventories every `page_registry.py` page and tracks public cutover plus closed-club admin pilot gates. | Keep the matrix current with `make check-next-parity-matrix`; use `/admin` as the operations cockpit. | Medium |
| FastAPI public/API surface | In progress | FastAPI exists under `services/api` with public club, leaderboard, league-results, badge-codex, challenge-ladder, weekly-recap, tournament-registration, player, match, match-explorer, live-session, and admin-operations status endpoints. | Deploy/verify staging API against staging Supabase and keep public/admin-status contracts sanitized. | Medium |
| Next.js public web | In progress | Next.js app exists under `apps/web` with the public product spine plus FAQ/legal/support/static routes and tournament registration intake/confirmation routes; a branch Preview is deployed. | Bind the next Vercel candidate to the same SHA as Fly, run canonical `Staging Smoke` after `write_wave=none`, and obtain later legal-copy approval before any public/custom-domain move. | Medium |
| Next.js admin operations cockpit | Automated-ready; manual acceptance pending | `/admin` renders migration mode, workflow flags, pilot gates, and Streamlit fallback from FastAPI status. The tracked operational surfaces are implemented behind scoped gates. | Complete the consolidated exact-candidate read and bounded-write acceptance book; keep Streamlit available. | Medium |
| Vercel staging | In progress | The `staging` branch Preview alias and immutable deployment identity are available; `/api/environment` participates in candidate attestation. | After the hardening merge, capture the replacement deployment ID and immutable origin, then run canonical `Staging Smoke` against that exact candidate. | Medium |
| Next.js admin score entry | Blocked by default | Admin rated score entry is disabled by feature flag and guarded by Match Log/Replay recovery contracts. | Exercise it only in a bounded staging write wave with disposable rows, Match Log and Replay History open, and exact restoration. | High |
| Admin auth/JWT | Implemented; acceptance pending | Next stores a Supabase admin session and FastAPI validates Supabase JWTs for guarded routes. | Complete exact-candidate authenticated browser acceptance, including expired/invalid-session behavior. | High |
| Club-scoped roles | Implemented; acceptance pending | Guarded FastAPI admin routes enforce club-scoped permissions and audit attribution. | Prove representative denial and allowed-action paths in the consolidated staging session before broad staff use. | High |
| Public leaderboard/read models | Automated-ready; manual acceptance pending | Public leaderboard, League Results, Badge Codex, Challenge Ladder, Weekly Recap, Match Explorer, player, match, live, tournament-registration, roster, partner-board, and static/support surfaces are exposed as sanitized public routes with route-specific checks. | Complete the exact-candidate non-mutating browser session and the separately bounded intake/pairing rows. | Medium |
| Workers | In progress | Worker CLIs exist and are part of moving jobs out of Streamlit. | Validate worker reliability and club-scoped safety in staging. | Medium |
| Email safety | At-rest safe; inbox acceptance blocked | Staging is reconciled at `JUPR_EMAIL_MODE=dry_run`; live customer delivery remains disabled. | With Joe's approved staging inbox, prove redirected delivery, recovery, unsubscribe/preferences, provider/audit evidence, and return to `dry_run`. | High |
| CI/API contract tests | In progress | API contracts now include production-deployment and authenticated admin UX safety suites; Next Web Build runs the guided Tournament Setup payload contract on pinned Node 20; canonical `Staging Smoke` requires 56 strict public-read checks plus five no-skip guided Tournament Setup browser checks. Parity Final Evidence is registered on the default branch but always checks out and binds executable evidence to canonical `staging`; authenticated modes mint and validate a fresh masked staging token. Deployment identity checks remain fail closed, and `make public-web-smoke` remains a noncanonical diagnostic. | Require the parity/manual/manifest guards on staging PRs and manually dispatch canonical `Staging Smoke` only after the final same-SHA Fly `none` release. | Medium |
| Migrations | In progress | `supabase/migrations/` is canonical and staging-first migration flow is defined. | Continue staging-first apply/verify discipline before production SQL changes. | Medium |
| Observability | In progress | Vercel runtime/build logs, Fly health/logs, FastAPI health/identity, GitHub workflow evidence, and Supabase logs/advisors are inspectable. | Define owner-visible minimum alerts and escalation/rollback instructions before public cutover. | Medium |
| Tenant onboarding | Not started | Multi-club onboarding process is not yet standardized for SaaS rollout. | Define repeatable club onboarding checklist and config flow in staging. | Medium |
| Billing/self-serve | Not started (deferred) | Billing and self-serve are intentionally deferred at this phase. | Revisit only after production admin migration is proven. | Low |

## Release finish lines

1. **Tres Palapas public launch:** freeze and accept one staging candidate,
   complete legal/domain/minimum-monitoring/email prerequisites, and launch the
   public Next/FastAPI product while retaining Streamlit as the admin fallback.
2. **Administrative replacement:** finish the remaining guarded write/recovery
   acceptance batches and retire Streamlit only after every migrated workflow is
   formally proven.
3. **Broader multi-club SaaS:** standardize onboarding, validate a second club,
   and add billing/self-serve capabilities as later work.

Billing, self-serve onboarding, and a second-club pilot do not block the initial
Tres Palapas public launch.

## Explicit non-goals for now

- No self-serve billing.
- No broad, ungated production Next.js admin rated writes.
- No replacement of Streamlit admin before workflow-specific parity gates are satisfied.
- No JavaScript rating logic.
- No direct browser writes to rating tables.
- No production custom-domain cutover until the staging public smoke checklist passes.
- No browser-side data-correction writes; correction requests remain staff-reviewed.
- No public tournament registration draw seeding, score entry, bracket mutation, or rating writes.

## Related docs

- `README.txt` (operator source of truth, branch model, promotion checklist)
- `docs/next_streamlit_parity_matrix.md` (Streamlit-to-Next parity control board)
- `docs/next_admin_operations_migration.md` (closed-club admin pilot control plan)
- `docs/saas_staging_deploy.md` (staging deployment guardrails)
- `docs/next_admin_auth_design.md` (required auth/authorization contract before admin enablement)
- `docs/streamlit_to_saas_migration.md` (broader migration context and PR sequencing)
- `docs/migrations.md` (staging-first migration source-of-truth and apply discipline)
- `docs/supabase_staging_advisor_review_2026-07-24.md` (read-only staging advisor and duplicate-index review)
- `docs/second_club_staging_pilot.md` (phase-4 pilot plan)
- `apps/web/README.md` (Next/Vercel web runtime)
- `services/api/README.md` (FastAPI runtime)
