# SaaS Implementation Status

This document is the durable source of truth for JUPR SaaS migration status so Joe and Codex stay aligned without relying on memory.

## Current status summary

- Streamlit production is active.
- `Test` branch is staging by policy, while recent SaaS PRs have been landing on `rollback-feb8`; reconcile branch promotion before any public cutover.
- Staging Supabase exists and is the non-production data environment for staging validation.
- FastAPI + Next.js are staging-first.
- Next.js public routes now cover the read-only product spine: club home, leaderboards, league results, Match Explorer, players, player profiles, matches, match detail, JUPR Live, and ratings explainer.
- `docs/next_streamlit_parity_matrix.md` is the control board for reaching 100% Streamlit workflow parity on Next/Vercel/FastAPI.
- Public smoke tooling exists for staging FastAPI + Vercel validation.
- Next admin score entry remains disabled.
- Production traffic has not moved to Next.js; Streamlit remains the production runtime.

## Status table

| Area | Status | Current implementation | Next action | Risk level |
|---|---|---|---|---|
| Staging Supabase | Done | Staging Supabase project exists and is used for SaaS validation. | Keep environment guards in place and apply schema changes staging-first. | Medium |
| Streamlit production | Done | Streamlit remains the active production admin/runtime surface. | Keep production stable while SaaS surfaces are proven in staging. | Medium |
| Streamlit-to-Next parity control | In progress | `docs/next_streamlit_parity_matrix.md` inventories every `page_registry.py` page and tracks Next/FastAPI parity gates. | Keep the matrix current with `make check-next-parity-matrix`; validate public Match Explorer and League Results in staging, then start Badge Codex. | Medium |
| FastAPI public API | In progress | FastAPI exists under `services/api` with public club, leaderboard, league-results, player, match, match-explorer, and live-session endpoints. | Deploy/verify staging API against staging Supabase and keep public contracts sanitized. | Medium |
| Next.js public web | In progress | Next.js app exists under `apps/web` with the read-only public product spine. | Deploy Vercel staging and run `make public-web-smoke` before any public/custom-domain move. | Medium |
| Vercel staging | In progress | `apps/web/.env.example`, deployment docs, and public smoke harness define the staging path. | Configure Vercel staging env vars and wire the deployed URL into the manual smoke workflow. | Medium |
| Next.js admin score entry | Blocked (intended) | Admin rated score entry path is disabled by feature flag and guardrails. | Keep disabled until auth, authorization, audit, and E2E safety criteria are complete. | High |
| Admin auth/JWT | In progress | JWT scaffolding exists for staged admin routes, but end-to-end production-grade auth/session hardening is not complete. | Complete validation hardening, session strategy, and staging E2E auth failure coverage. | High |
| Club-scoped roles | In progress | Role assignment checks exist for the FastAPI score-entry endpoint, but broader club-scoped authorization still needs end-to-end coverage. | Enforce and test club-scoped role checks on every write path before enabling admin writes. | High |
| Public leaderboard/read models | In progress | Public leaderboard, League Results, Match Explorer, player, match, and live read surfaces are being exposed as sanitized FastAPI contracts. | Continue public parity with Badge Codex, Challenge Ladder, Weekly Recap, and static/legal/support pages. | Medium |
| Workers | In progress | Worker CLIs exist and are part of moving jobs out of Streamlit. | Validate worker reliability and club-scoped safety in staging. | Medium |
| Email safety | Blocked (safety gate) | Email worker usage is intentionally constrained pending safe staging email configuration. | Confirm safe non-production email routing before broader worker runs. | High |
| CI/API contract tests | In progress | `make api-test` covers API contracts; `scripts/smoke_public_web.py` covers staged public API/web smoke checks; `make check-next-parity-matrix` guards parity drift. | Wire public smoke into GitHub Actions secrets/inputs and require parity matrix coverage before public/admin migration PRs. | Medium |
| Migrations | In progress | `supabase/migrations/` is canonical and staging-first migration flow is defined. | Continue staging-first apply/verify discipline before production SQL changes. | Medium |
| Observability | Not started | No fully defined cross-surface observability baseline is documented as a release gate. | Establish logs, metrics, and alerting minimums for API/web/workers. | Medium |
| Tenant onboarding | Not started | Multi-club onboarding process is not yet standardized for SaaS rollout. | Define repeatable club onboarding checklist and config flow in staging. | Medium |
| Billing/self-serve | Not started (deferred) | Billing and self-serve are intentionally deferred at this phase. | Revisit only after production admin migration is proven. | Low |

## Roadmap phases

1. **Phase 1 — staging environment verification:** verify branch/runtime wiring, staging Supabase isolation, and smoke checks.
2. **Phase 2 — public read-only SaaS demo:** ship/validate read-only FastAPI + Next.js public club and leaderboard/player/match/live flows in staging.
3. **Phase 2.5 — public Streamlit parity:** port public Match Explorer, League Results, Badge Codex, Challenge Ladder, Weekly Recap, static/legal/support pages, and public tournament registration surfaces.
4. **Phase 3 — club-scoped auth/admin writes in staging:** enforce JWT auth, role checks, club scoping, and audit attribution for admin writes.
5. **Phase 4 — second-club staging pilot:** validate multi-club onboarding and isolation in staging (see `docs/second_club_staging_pilot.md`).
6. **Phase 5 — production read-only public cutover:** move public read-only traffic to Next.js/FastAPI after staged proof.
7. **Phase 6 — admin migration after proven:** migrate admin workflows from Streamlit only after production read-only stability and rollback confidence.

## Explicit non-goals for now

- No self-serve billing.
- No production Next.js admin rated writes.
- No replacement of Streamlit admin before parity gates are satisfied.
- No JavaScript rating logic.
- No direct browser writes to rating tables.
- No production custom-domain cutover until the staging public smoke checklist passes.

## Related docs

- `README.txt` (operator source of truth, branch model, promotion checklist)
- `docs/next_streamlit_parity_matrix.md` (Streamlit-to-Next parity control board)
- `docs/saas_staging_deploy.md` (staging deployment guardrails)
- `docs/next_admin_auth_design.md` (required auth/authorization contract before admin enablement)
- `docs/streamlit_to_saas_migration.md` (broader migration context and PR sequencing)
- `docs/migrations.md` (staging-first migration source-of-truth and apply discipline)
- `docs/second_club_staging_pilot.md` (phase-4 pilot plan)
- `apps/web/README.md` (Next/Vercel web runtime)
- `services/api/README.md` (FastAPI runtime)
