# SaaS Implementation Status

This document is the durable source of truth for JUPR SaaS migration status so Joe and Codex stay aligned without relying on memory.

## Current status summary

- Streamlit production is active.
- `Test` branch is staging.
- Staging Supabase exists and is the non-production data environment for staging validation.
- FastAPI + Next.js are staging-first.
- Next admin score entry remains disabled.

## Status table

| Area | Status | Current implementation | Next action | Risk level |
|---|---|---|---|---|
| Staging Supabase | Done | Staging Supabase project exists and is used for `Test` branch SaaS validation. | Keep environment guards in place and apply schema changes staging-first. | Medium |
| Streamlit production | Done | Streamlit remains the active production admin/runtime surface. | Keep production stable while SaaS surfaces are proven in staging. | Medium |
| FastAPI public API | In progress | FastAPI exists under `services/api` and is being staged as the public API layer. | Complete and validate read-only public endpoints against staging Supabase. | Medium |
| Next.js public web | In progress | Next.js app exists under `apps/web` and is used staging-first for public pages. | Finish read-only public flows and staging smoke coverage. | Medium |
| Next.js admin score entry | Blocked (intended) | Admin rated score entry path is disabled by feature flag and guardrails. | Keep disabled until auth, authorization, audit, and E2E safety criteria are complete. | High |
| Admin auth/JWT | Not started | No production-ready Supabase Auth JWT validation flow is finalized in FastAPI. | Implement JWT validation, session model, and auth enforcement in staging first. | High |
| Club-scoped roles | Not started | Club-scoped authorization is a required guardrail and not yet fully enforced for new admin writes. | Enforce club-scoped role checks on every write path. | High |
| Public leaderboard read model | In progress | Public read model/tables are defined directionally for SaaS read-only consumption. | Finalize read-model contracts and verify endpoint behavior on staging data. | Medium |
| Workers | In progress | Worker CLIs exist and are part of moving jobs out of Streamlit. | Validate worker reliability and club-scoped safety in staging. | Medium |
| Email safety | Blocked (safety gate) | Email worker usage is intentionally constrained pending safe staging email configuration. | Confirm safe non-production email routing before broader worker runs. | High |
| CI/API contract tests | Not started | No complete staging contract suite is yet the enforced gate for SaaS cutover. | Add API contract tests + staging CI gates for public endpoints and auth failures. | Medium |
| Migrations | In progress | `supabase/migrations/` is canonical and staging-first migration flow is defined. | Continue staging-first apply/verify discipline before production SQL changes. | Medium |
| Observability | Not started | No fully defined cross-surface observability baseline is documented as a release gate. | Establish logs, metrics, and alerting minimums for API/web/workers. | Medium |
| Tenant onboarding | Not started | Multi-club onboarding process is not yet standardized for SaaS rollout. | Define repeatable club onboarding checklist and config flow in staging. | Medium |
| Billing/self-serve | Not started (deferred) | Billing and self-serve are intentionally deferred at this phase. | Revisit only after production admin migration is proven. | Low |

## Milestones

1. **Milestone 1:** Staging read-only public SaaS demo.
2. **Milestone 2:** Tenant-safe admin auth and scorekeeper workflow in staging.
3. **Milestone 3:** Second club pilot in staging (see `docs/second_club_staging_pilot.md`).
4. **Milestone 4:** Production public read-only Next.js/FastAPI cutover.
5. **Milestone 5:** Production admin migration from Streamlit, only after proven.

## Explicit non-goals for now

- No self-serve billing.
- No production Next.js admin rated writes.
- No replacement of Streamlit admin.
- No JavaScript rating logic.

## Related docs

- `README.txt`
- `docs/saas_staging_deploy.md`
- `docs/next_admin_auth_design.md`
- `docs/streamlit_to_saas_migration.md`
- `docs/migrations.md`
- `docs/second_club_staging_pilot.md`
