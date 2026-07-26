# SaaS Implementation Status

This document is the durable source of truth for the Pickleball Club Sandwich
(JUPR) migration so Joe and Codex stay aligned without relying on memory.

## Current status summary

- Streamlit production is active and remains the fallback admin runtime.
- `staging` is the canonical staging integration/deployment branch. `Test` is legacy/deprecated and must not receive new staging PRs or evidence.
- Staging Supabase exists and is the non-production data environment for staging validation.
- FastAPI + Next.js are staging-first for public traffic, with closed-club production-write pilot mode now defined for staff workflows.
- Next.js public routes now cover the read-only product spine: club home, leaderboards, league results, Badge Codex, Challenge Ladder, Weekly Recap, Match Explorer, players, player profiles, matches, match detail, JUPR Live, ratings explainer, FAQ, privacy, terms, support/contact, data-correction instructions, and public tournament registration intake.
- Next `/admin` is the protected operations migration cockpit. The server shell
  contains no posture data; a capability-checked browser session must call the
  bearer-protected, club-scoped FastAPI `GET /admin/operations/status` endpoint
  before any workflow flags or infrastructure readiness are rendered.
- PR `#1037` landed the protected cockpit, shared responsive Tournament Admin
  navigation, richer verified Challenge Ladder results, and atomic/idempotent
  Challenge publication and recovery. PR `#1038` then changed Moneyball status
  reads to query canonical `players.active` first, eliminating the expected
  legacy-column 400 noise while retaining a controlled compatibility fallback.
- `docs/next_streamlit_parity_matrix.md` is the control board for reaching 100% Streamlit workflow parity on Next/Vercel/FastAPI.
- Public/staff smoke tooling exists for staging FastAPI + Vercel validation.
- The 2026-07-24 launch baseline remains Git SHA
  `eab384545c493f145af383c8e26d8bf97686ab21`: Vercel deployment
  `dpl_FTmDVFSdftMmAj4sMwAHrvQpVYLE`, Fly image
  `registry.fly.io/juprleagues-api-staging:deployment-01KY5SVX80SHEZMN7GX37NASRP`,
  and staging Supabase project `sijpxjxvdtrehmqvirfi`. Vercel and Fly served that
  exact SHA, Fly was healthy at `write_wave=none`, and canonical Staging Smoke run
  `29957623653` passed all 56 strict browser checks. PR `#1022` remains the
  historical candidate-evidence PR and must not be mistaken for the final
  repair/freeze PR.
- Candidate `ccf2e469b6ef76cffbbd5525c5b1ff1f5ff503bc` is preserved as
  pre-fix diagnostic evidence, not as the final accepted candidate. Parity Final
  Evidence public-read run `30160267159` passed all `56/56` expected browser
  tests with zero skips, flakes, or unexpected results. Artifact `8620058712`
  has digest
  `sha256:6391d4e163e186b04e5069fb33ea7f099dcd79bbc1af4b3d259537163385140a`.
  Admin-read-export run `30161862524` then failed safely during session
  preparation because the former stored staging-admin email/password inputs were
  absent. It ran zero browser tests, performed zero writes, uploaded no artifact,
  and left Fly at `write_wave=none`. The session-bootstrap hardening changes the
  application SHA, so the successful public-read run cannot be carried forward
  as final-candidate acceptance.
- Candidate `7cedb81ca251023806b0953db996a6e7b80c381a` then completed an
  exact-SHA Fly `none` deployment (run `30166837522`), `public-read` `56/56`
  (run `30167694296`, artifact `8622060773`), and
  `admin-read-export` `16/16` (run `30167701340`, artifact `8622073617`).
  A fixture/recovery audit subsequently found three defects: managed singles
  replay/restoration, League Live round-robin preview gating, and ignored
  Match Log recovery context. That identity and its otherwise-green read
  artifacts are explicitly superseded; every later candidate must repeat each
  candidate-bound run.
- Authenticated evidence no longer depends on a stored admin email, password,
  bearer token, user ID, or fixture ID. The server-side preparation step uses the
  staging service role to require exactly one eligible, existing, user-bound
  staging admin assignment, verifies that Auth identity, requests a Supabase
  Admin `generate_link`, exchanges its token hash directly without sending an
  email, validates club-scoped FastAPI capabilities and allowlisted fixtures, and
  exposes only short-lived session material to the browser step. Cleanup is
  mandatory and fails the job unless the refreshable session is ended or already
  inactive. The access JWT is exact-identity bound, limited to one hour, cleared
  from subsequent workflow steps, and explicitly treated as potentially valid
  until `exp`. Retained browser artifacts redact the JWT, operator email, and
  Vercel bypass value. The process creates or deletes no Auth user and changes no
  role assignment or business row. It intentionally creates and consumes bounded
  staging Auth link-token, session, and audit metadata.
- The current live staging readback has 39 migration-ledger entries through the
  connector-assigned `20260726011915_challenge_ladder_public_results` head.
  Post-apply probes verified the nullable Challenge public-result relation and
  validated constraint, valid active-claim index, five mutation triggers, six
  invoker functions with fixed `pg_catalog` search paths and execution limited
  to `service_role`, and zero active Challenge operations. All 29 implementation
  orders have landed,
  with roughly 70 Next page components covering the 47 tracked Streamlit
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
| FastAPI public/API surface | In progress | FastAPI exists under `services/api` with public club, leaderboard, league-results, badge-codex, challenge-ladder, weekly-recap, tournament-registration, player, match, match-explorer, and live-session endpoints; admin-operations status is bearer- and assignment-protected. | Deploy/verify staging API against staging Supabase and keep public contracts sanitized and admin posture fail-closed. | Medium |
| Next.js public web | In progress | Next.js app exists under `apps/web` with the public product spine plus FAQ/legal/support/static routes and tournament registration intake/confirmation routes; a branch Preview is deployed. | Bind the next Vercel candidate to the same SHA as Fly, run canonical `Staging Smoke` after `write_wave=none`, and obtain later legal-copy approval before any public/custom-domain move. | Medium |
| Next.js admin operations cockpit | Automated-ready; manual acceptance pending | `/admin` renders only a data-free access-check shell anonymously. After a capability-checked session it fetches the protected posture, revalidates every minute and on focus/visibility return, and immediately clears stale posture on logout, scope change, or denial. The API independently requires a verified JWT and bound admin assignment. | Complete authenticated exact-candidate acceptance plus representative anonymous, wrong-club, revoked-assignment, and expired-session denial checks; keep Streamlit available. | Medium |
| Vercel staging | In progress | The `staging` branch Preview alias and immutable deployment identity are available; `/api/environment` participates in candidate attestation. The last successful read checkpoint is superseded by later repairs. | For each freeze candidate, require the same exact `staging` SHA on Vercel and Fly, capture the deployment ID and immutable origin, then rerun exact-candidate read evidence and canonical `Staging Smoke` against the restored none release. | Medium |
| Next.js admin score entry | Blocked by default | Admin rated score entry is disabled by feature flag and guarded by Match Log/Replay recovery contracts. | Exercise it only in a bounded staging write wave with disposable rows, Match Log and Replay History open, and exact restoration. | High |
| Direct Match Uploader singles | Implemented; blocked | UI/API and replay-managed contracts exist, and the singles replay schema is applied in staging, but `JUPR_ENABLE_NEXT_ADMIN_MATCH_UPLOADER_SINGLES=0` is forced in every staging wave and production because the direct writer is not yet atomic. | Make the writer atomic, then run the preserved rated/unrated exact-readback and full-replay protocol in a later isolated wave. | High |
| Destructive Match Log actions | Implemented; blocked | Duplicate cleanup and bulk exclusion are guarded by dormant `JUPR_ENABLE_NEXT_ADMIN_MATCH_LOG_DESTRUCTIVE=0` pending atomic idempotent recovery. Ordinary atomic edits and duplicate no-issue remain available. | Keep dependent match-producing manual acceptance deferred; prove retries, unknown outcomes, exact recovery, and baseline restoration before a later enablement. | High |
| Tournament official singles publish | Automated-ready; manual recovery blocked | The separate Tournament Operations official-singles publisher preserves the replay-managed marker through an atomic CAS RPC and does not use the direct-uploader singles gate; its schema is applied in staging. | Retain automated coverage and defer manual publish→exclude→full-Replay evidence until destructive exclusion is safely enabled. | High |
| Admin auth/JWT | Implemented; acceptance pending | Next stores a Supabase admin session and FastAPI validates Supabase JWTs for guarded routes. Candidate evidence bootstraps one existing bound staging admin through a server-only no-email magic-link exchange, ends its refreshable session after use, clears exported credentials, and records that the access JWT may remain valid until its bounded `exp`; no user/role/business row is created, deleted, or changed, while bounded Auth token/session/audit metadata is created and consumed. | Complete exact-candidate live-session/sign-out evidence plus manual password-entry, recovery/inbox, expired/invalid-session, wrong-club, and permission-denial acceptance. | High |
| Club-scoped roles | Implemented; acceptance pending | Guarded FastAPI admin routes enforce club-scoped permissions and audit attribution. | Prove representative denial and allowed-action paths in the consolidated staging session before broad staff use. | High |
| Public leaderboard/read models | Automated-ready; manual acceptance pending | Public leaderboard, League Results, Badge Codex, Challenge Ladder, Weekly Recap, Match Explorer, player, match, live, tournament-registration, roster, partner-board, and static/support surfaces are exposed as sanitized public routes with route-specific checks. | Complete the exact-candidate non-mutating browser session and the separately bounded intake/pairing rows. | Medium |
| Workers | In progress | Worker CLIs exist and are part of moving jobs out of Streamlit. | Validate worker reliability and club-scoped safety in staging. | Medium |
| Email safety | At-rest safe; inbox acceptance blocked | Staging is reconciled at `JUPR_EMAIL_MODE=dry_run`; live customer delivery remains disabled. | With Joe's approved staging inbox, prove redirected delivery, recovery, unsubscribe/preferences, provider/audit evidence, and return to `dry_run`. | High |
| CI/API contract tests | In progress | API contracts now include production-deployment and authenticated admin UX safety suites; Next Web Build runs the Challenge Ladder full-result rendering contract and guided Tournament Setup payload contract on pinned Node 20; canonical `Staging Smoke` requires 69 strict public-read checks plus five no-skip guided Tournament Setup browser checks. Parity Final Evidence is registered on the default branch but always checks out and binds executable evidence to canonical `staging`; authenticated modes use a server-only service-role lookup plus no-email token-hash exchange for exactly one existing bound staging admin, validate an exact-identity access JWT with a maximum one-hour lifetime, and require confirmed refresh-session termination in an always-running cleanup step. Cleanup clears exported credentials and reports that the access JWT may remain valid until `exp`; retained browser output is redacted. The exact-candidate browser test proves a live capability-checked session and sign-out, not manual password entry or recovery-email delivery. Deployment identity checks remain fail closed, and `make public-web-smoke` remains a noncanonical diagnostic. | For each freeze candidate, run the parity/manual/manifest guards, repeat every evidence mode after its exact Vercel/Fly deployment, and manually dispatch canonical `Staging Smoke` only after the final same-SHA Fly `none` release. | Medium |
| Migrations | In progress | `supabase/migrations/` is canonical; staging has all 39 reviewed logical names through connector head `20260726011915_challenge_ladder_public_results`, with its schema, mutation guards, RPC security, and zero-active-operation state verified. | Preserve the 39-name staging readback in the candidate evidence and keep staging-first discipline before any separately approved production SQL. | Medium |
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
