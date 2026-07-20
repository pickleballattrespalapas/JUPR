# Tournament Setup and Admin parity evidence (order 26)

This slice makes `tournaments`, `tournament_manager`, and
`tournament_registration_admin` automated-ready and manual-ready. It does
**not** move any parity-matrix row to `Done`; the final staging acceptance and
matrix reconciliation remain order 29 work.

## Authority and boundaries

- FastAPI authenticates the Supabase JWT, resolves the club-scoped
  `manage_tournaments` permission, and calls Python services. Next renders
  state and sends reviewed commands only.
- `/admin/tournament-setup` is the dedicated setup decision: registration
  settings, Python templates, durable drafts, no-write impact review, and the
  exact `PUBLISH SETUP` confirmation live there. Draw creation/scoring remains
  Tournament Ops.
- Setup publish normalizes only approved day/event columns, assigns stable IDs
  for incomplete drafts, forces the authorized tournament ID, and rejects any
  supplied row ID already owned by another tournament before service-role
  writes. Existing rows update through tournament-scoped predicates; new rows
  insert rather than cross-tournament upsert.
- Registration Admin lists/detail/exports, edits single or selected rows, and
  offers a read-only Operations import handoff. It never writes draw teams.
  Tournament Ops must independently recheck draw scope, linked players,
  duplicates, and the existing-games lock with `IMPORT REGISTRATIONS`.
- Imported selections cannot be changed through Registration Admin. Imported
  registration status changes and bulk status changes are refused before the
  first write; draw membership must be repaired in Tournament Ops.
- Formula-like CSV cells beginning with `=`, `+`, `-`, `@`, tab, or carriage
  return remain apostrophe-prefixed by the Python reporting service.

## Staging-only write gates

Keep the read flag independent from every mutation surface:

```text
JUPR_ENV=staging
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENTS=1
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_MUTATIONS=1
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_SETUP_MUTATIONS=1
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_REGISTRATION_MUTATIONS=1
JUPR_ENABLE_NEXT_ADMIN_TOURNAMENT_IMPORT_HANDOFF=1
JUPR_REQUIRE_API_AUDIT_LOG=1
SUPABASE_SERVICE_ROLE_KEY=<FastAPI-only secret>
JUPR_STREAMLIT_FALLBACK_URL=<reviewed fallback>
```

The three Next mutation flags are accepted only when `JUPR_ENV=staging`.
The import-handoff flag is an operational readiness marker for the separately
guarded Tournament Ops exercise; it never enables a write from Registration
Admin. Production refuses these order-26 mutations even if a flag is
accidentally present. `SUPABASE_SERVICE_ROLE_KEY` belongs only on FastAPI;
never put it in a Next/Vercel public variable or browser storage. Apply
`supabase/migrations/20260719203000_tournament_admin_operations.sql` through
the normal reviewed migration process before opening a mutation flag. This
implementation did not apply the migration or mutate staging/live data.

## Durable mutation contract

Every staging mutation requires the version or state fingerprint loaded by the
operator. FastAPI builds a canonical SHA-256 request fingerprint and a second
deterministic 64-character operation key from club, surface, action, entity,
reviewed state, lock scope, and normalized payload. A partial unique index on
`(club_id, lock_scope)` atomically serializes all active order-26 operations for
the same tournament, including setup, shell, and registration surfaces.

1. Completed identical operations replay their stored response without a
   second domain write.
2. A changed payload/state generates a different key and must pass a fresh
   stale check.
3. Deterministic validation runs only after completed/reconcilable replay has
   been checked, but before an operation row or audit is created.
4. State is checked once before and once after the atomic tournament lock is
   acquired. Versioned row mutations retain compare-and-swap predicates. Empty
   draft deletion uses one service-role-only SQL function that locks the parent
   version, rechecks every usage table, and deletes setup children plus the
   parent in the same transaction.
5. A required `*_intent` audit is persisted before the first domain mutation.
6. The result is persisted before the required `*_completion` audit.
7. If completion audit is lost, the identical request reconciles the stored
   result without repeating the mutation.
8. Any exception after intent is treated as an uncertain partial/response-loss
   outcome and stored as `recovery_required`. It is never labelled normally
   retryable. Reload authoritative state and use the fallback if the result
   cannot be verified.
9. A `*_failure` audit is attempted for every post-intent failure. If that
   audit also fails, the API preserves explicit recovery-required language.

The implementation is intentionally isolated in
`admin_tournament_guarded_operation.py`. When order 23 is stacked, its shared
guarded-operation layer should adopt this module's persistence/audit callbacks;
do not retain two generic frameworks. The Tournament Admin request/response and
table contract can remain stable through that adapter.

## Exact confirmation phrases

| Action | Phrase |
|---|---|
| Tournament shell edit | `SAVE TOURNAMENT` |
| Registration edit | `SAVE REGISTRATION` |
| Selection/division edit | `SAVE SELECTION` |
| Bulk registration edit | `BULK UPDATE REGISTRATIONS` |
| Archive / unarchive | `ARCHIVE` / `UNARCHIVE` |
| Empty draft deletion | `DELETE DRAFT` |
| Setup settings | `SAVE SETUP` |
| Setup draft | `SAVE SETUP DRAFT` |
| Setup publish | `PUBLISH SETUP` |
| Ops import (separate surface) | `IMPORT REGISTRATIONS` |

## Automated evidence

```bash
/tmp/jupr-followup-venv/bin/python -m pytest -q \
  tests/test_tournament_admin_guarded_operations.py \
  tests/test_api_contract_admin_tournament_guarded.py \
  tests/test_api_contract_admin_tournament.py \
  tests/test_api_contract_admin_tournament_setup.py \
  tests/test_api_contract_admin_tournament_status_action.py \
  tests/test_api_contract_admin_tournament_delete_draft.py \
  tests/test_admin_tournament_registration_reporting_service.py

make check-next-parity-matrix check-parity-closure-program
cd apps/web && npx tsc --noEmit && npm run build
cd apps/web && npx playwright test e2e/tournament-admin.staging.spec.ts
```

The Playwright spec is compiled/listed locally. Its authenticated and mutation
cases require the documented staging credentials, disposable fixture IDs, and
explicit mutation-evidence environment gate; they were not run against staging
by this implementation branch.

The focused tests include no-write/stale spies, deterministic identity,
intent-before-domain-write ordering, completion replay, partial-mutation
response-loss refusal, Python template/impact dry run, imported-draw refusal,
cross-tournament setup ownership refusal, active-lock/second-stale checks,
service-role/RLS/CORS static guards, explicit-null date clearing, and safe
reporting coverage.

## Manual staging book

Use one disposable DRAFT tournament, two disposable registrations (one with a
selection), and a separate imported-selection fixture. Record request IDs,
operation keys/fingerprints, intent/completion/failure audit IDs, screenshots,
and cleanup evidence.

1. Keep all write flags off. Confirm list/detail/export/impact review work and
   every write fails closed with a visible Streamlit fallback.
2. Open only Setup mutations. Load the Python standard template, save a draft,
   review impact, deliberately change JSON, and prove publish disables until a
   new review. Publish the disposable setup once; resend the exact request and
   verify `idempotent_replay=true` with no second configuration change.
3. In another session change settings, then submit the stale browser state.
   Expect 409, no operation row, no audit row, and no domain write. Reload.
4. Open Tournament mutations. Perform a same-value shell edit, archive,
   unarchive, and delete only an entirely empty disposable draft. Verify every
   intent precedes the domain audit/completion and stale tabs reload safely.
5. Open Registration mutations. Exercise single, selection, and bulk edits.
   Verify a stale member blocks the entire bulk preflight before any row write.
6. Load the imported fixture. Confirm single/bulk registration-status and all
   selection edits refuse with a Tournament Ops handoff. Confirm the handoff
   itself reports `dry_run=true`, `write_count=0`, and has no import button.
7. In Tournament Ops, verify import is refused after games exist. Do not remove
   or rewrite scored draw history merely to satisfy this exercise.
8. Simulate response loss after the persisted result but before completion
   audit. Resend the exact request and verify reconciliation without a second
   domain mutation. Simulate an exception after a partial write and verify the
   operation stays `recovery_required` until authoritative manual review.
9. Export a disposable name/note beginning with each spreadsheet trigger and
   confirm the CSV cell is escaped. No email is sent; broadcast remains preview
   only and staging email mode remains `dry_run` or `staging_redirect`.
10. Return every reversible field to its recorded starting value, delete only
    the empty disposable draft, retain immutable audits/operation evidence, and
    close all four mutation flags.

Stop immediately on wrong-club data, a missing service role/migration, missing
intent audit, a non-409 stale write, a duplicate operation, an imported-draw
bypass, or uncertain state without recovery guidance. Use the linked Streamlit
fallback and do not promote a matrix row to `Done`.
