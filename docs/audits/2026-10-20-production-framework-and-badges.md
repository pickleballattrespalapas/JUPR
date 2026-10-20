# Production candidate: framework security and badge contracts

Prepared from production `9cc0ca576dfe6d3a178d99761a9d51cbb3d2e607`. This is a selective port of the framework upgrade already integrated on staging, plus the badge-contract repair in PR 1646. Production deployment requires Joe's separate explicit approval under AGENTS.md.

## Changes

Next 16.4.2, React 19.3.1, Node 24 and matching lint/types replace the production Next 14/React 18 stack. All request APIs are awaited, including production-only root layout and the gated tournament fixture. Existing production homepage, club routes, tournament aliases, workflow checks and production API readiness gate are preserved. Webpack remains explicit. CI runs lint separately because Next 16 no longer includes lint in its build, and RR/recap acceptance shares a single production build against a local fixture API.

New match-correction operations capture the frozen 41-badge v2 contract. Existing v1 operations recover using their saved 29-badge scope and row version. No database migration, historical rewrite, correction-gate activation, or email scheduler change is included. The history-loading limit remains a separate blocker to enabling large-club match corrections.

## Verification on the production candidate

- Full existing component suite, TypeScript and production build pass.
- All 10 RR browser scenario groups pass against a prebuilt Next server and synthetic FastAPI data, including public/admin handoff, token revocation, navigation, scoring and late arrivals.
- Recap browser acceptance passes: generation, independent email ordering, photo/promotion cards, PDF/PNG export, autosave, preview, publish/private revisions, reload, reusable layouts and mobile editing.
- 80 focused Python tests pass, including both badge contract versions, old-job recovery, schema guards and CI safety contracts.
- Actual PostgreSQL claim/apply routines pass synthetic PGlite checks for both versions, wrong-version/scope rejection, manual/unrelated award preservation and idempotency.
- Lint: zero errors, 186 warnings. Newly introduced React Compiler diagnostics remain warnings; React Compiler is not enabled.
- Production npm audit: zero vulnerabilities. Full audit: five high dependency entries in the existing development-only braces lint chain, documented in the original framework report; no upstream fix applied or unsafe downgrade used.
- Lockfile install dry-run passes. Runtime checks used Node 24 and Python 3.12 locally; CI uses Python 3.11.

The original `20261020_web_dependency_upgrade.md` records the staging upgrade history. This document records the production-specific port; staging-only homepage, club directory and intake changes were not imported. No production mutation or email was performed during this preparation.
