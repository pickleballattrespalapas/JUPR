# SaaS Implementation Status

This document is the durable source of truth for JUPR SaaS migration status so Joe and Codex stay aligned without relying on memory.

## Current status summary

- Streamlit production is active.
- `Test` branch is staging.
- Staging Supabase exists and is the non-production data environment for staging validation.
- FastAPI + Next.js are staging-first.
- Next admin score entry remains disabled.
- Production traffic has not moved to Next.js; Streamlit remains the production runtime.

## Status table

| Area | Status | Current implementation | Next action | Risk level |
|---|---|---|---|---|
| Staging Supabase | Done | Staging Supabase project exists and is used for `Test` branch SaaS validation. | Keep environment guards in place and apply schema changes staging-first. | Medium |
| Streamlit production | Done | Streamlit remains the active production admin/runtime surface. | Keep production stable while SaaS surfaces are proven in staging. | Medium |
| FastAPI public API | In progress | FastAPI exists under `services/api` and is being staged as the public API layer. | Complete and validate read-only public endpoints against staging Supabase. | Medium |
| Next.js public web | In progress | Next.js app exists under `apps/web` and is used staging-first for public pages. | Finish read-only public flows and staging smoke coverage. | Medium |
| Next.js admin score entry | Blocked (intended) | Admin rated score entry path is disabled by feature flag and guardrails. | Keep disabled until auth, authorization, audit, and E2E safety criteria are complete. | High |
| Admin auth/JWT | In progress | JWT scaffolding exists for staged admin routes, but end-to-end production-grade auth/session hardening is not complete. | Complete validation hardening, session strategy, and staging E2E auth failure coverage. | High |
| Club-scoped roles | Not started | Club-scoped authorization is a required guardrail and not yet fully enforced for new admin writes. | Enforce club-scoped role checks on every write path. | High |
| Public leaderboard read model | In progress | Public read model/tables are defined directionally for SaaS read-only consumption. | Finalize read-model contracts and verify endpoint behavior on staging data. | Medium |
| Workers | In progress | Worker CLIs exist and are part of moving jobs out of Streamlit. | Validate worker reliability and club-scoped safety in staging. | Medium |
| Email safety | Blocked (safety gate) | Email worker usage is intentionally constrained pending safe staging email configuration. | Confirm safe non-production email routing before broader worker runs. | High |
| CI/API contract tests | Not started | No complete staging contract suite is yet the enforced gate for SaaS cutover. | Add API contract tests + staging CI gates for public endpoints and auth failures. | Medium |
| Migrations | In progress | `supabase/migrations/` is canonical and staging-first migration flow is defined. | Continue staging-first apply/verify discipline before production SQL changes. | Medium |
| Observability | Not started | No fully defined cross-surface observability baseline is documented as a release gate. | Establish logs, metrics, and alerting minimums for API/web/workers. | Medium |
| Tenant onboarding | Not started | Multi-club onboarding process is not yet standardized for SaaS rollout. | Define repeatable club onboarding checklist and config flow in staging. | Medium |
| Billing/self-serve | Not started (deferred) | Billing and self-serve are intentionally deferred at this phase. | Revisit only after production admin migration is proven. | Low |

## Roadmap phases

1. **Phase 1 — staging environment verification:** verify branch/runtime wiring, staging Supabase isolation, and smoke checks.
2. **Phase 2 — public read-only SaaS demo:** ship/validate read-only FastAPI + Next.js public club and leaderboard flows in staging.
3. **Phase 3 — club-scoped auth/admin writes in staging:** enforce JWT auth, role checks, club scoping, and audit attribution for admin writes.
4. **Phase 4 — second-club staging pilot:** validate multi-club onboarding and isolation in staging (see `docs/second_club_staging_pilot.md`).
5. **Phase 5 — production read-only public cutover:** move public read-only traffic to Next.js/FastAPI after staged proof.
6. **Phase 6 — admin migration after proven:** migrate admin workflows from Streamlit only after production read-only stability and rollback confidence.

## Explicit non-goals for now

- No self-serve billing.
- No production Next.js admin rated writes.
- No replacement of Streamlit admin.
- No JavaScript rating logic.

## Related docs

- `README.txt` (operator source of truth, branch model, promotion checklist)
- `docs/saas_staging_deploy.md` (staging-only deployment guardrails)
- `docs/next_admin_auth_design.md` (required auth/authorization contract before admin enablement)
- `docs/streamlit_to_saas_migration.md` (broader migration context and PR sequencing)
- `docs/migrations.md` (staging-first migration source-of-truth and apply discipline)
- `docs/second_club_staging_pilot.md` (phase-4 pilot plan)
