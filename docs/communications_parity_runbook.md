# Communications parity staging runbook

This runbook covers the Next/FastAPI Weekly Recap Admin and Player Updates Admin surfaces. They remain staging-first and keep Streamlit as the operational fallback until the consolidated manual acceptance session is signed off. This work does not change any parity-matrix row to `Done`.

## Deployment prerequisites

Apply the canonical migration history through `supabase/migrations/20260720123402_baseline_worker_run_log.sql` to the staging Supabase project before deploying the API. This includes `supabase/migrations/20260719182606_communications_outbox_stale_guards.sql`, which adds optimistic row versions, idempotency keys, delivery-attempt metadata, an atomic verified-subscriber replacement RPC, and explicit service-role-only Data API grants. The forward worker-ledger baseline creates the durable pre-run marker if the historical migration is absent, enables RLS, revokes browser-role access, and preserves service-role access.

FastAPI staging requires:

```text
JUPR_ENV=staging
SUPABASE_URL=<staging project URL>
SUPABASE_SERVICE_ROLE_KEY=<staging service role; FastAPI only>
JUPR_ENABLE_NEXT_ADMIN_WEEKLY_RECAP=1
JUPR_ENABLE_NEXT_ADMIN_PLAYER_UPDATES=1
JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS=0
JUPR_STAGING_WRITE_WAVE=none
JUPR_REQUIRE_API_AUDIT_LOG=1
JUPR_REQUIRE_WORKER_RUN_LOG=1
JUPR_EMAIL_MODE=dry_run
JUPR_WEB_BASE_URL=<staging Next preview origin>
JUPR_STAGING_EMAIL_REDIRECT_TO=<approved staging inbox>
JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL=0
```

The two base feature flags expose authenticated reads at rest. They do not
authorize a mutation. Every communications POST/PATCH remains blocked by the
staging wave middleware and an independent application guard until the exact
`communications` wave sets
`JUPR_ENABLE_NEXT_ADMIN_COMMUNICATIONS_MUTATIONS=1`. Restore the mutation flag
to `0` and `JUPR_STAGING_WRITE_WAVE=none` after the bounded batch.
Reload the admin page after either deployment transition so its disabled-state
projection reflects the active release; FastAPI remains authoritative and
rejects a stale browser action.

For redirected delivery testing, use `JUPR_EMAIL_MODE=staging_redirect`, set `JUPR_STAGING_EMAIL_REDIRECT_TO` to the approved staging recipient, keep `JUPR_WEB_BASE_URL` pointed at the staging Next origin, configure SMTP secrets on Fly, and keep the live-delivery flag off. Never put the service-role key or SMTP secrets in Vercel or any `NEXT_PUBLIC_*` variable.

The reconciled staging runtime was at `JUPR_EMAIL_MODE=dry_run` on 2026-07-24.
Keep `dry_run` as the at-rest state. Do not send a real customer message. The
redirected-delivery, password-recovery, unsubscribe, and preferences checks wait
for Joe to name and observe an approved staging inbox; return to `dry_run` after
that bounded session.

## Automated-ready evidence

- Admin APIs require Supabase JWT authorization, club-scoped permissions, and a server-side service role.
- Every recap/subscription/outbox write carries `expected_row_version`; stale conditional writes fail or are returned as stale without mutating the row.
- Queue and replacement requests carry UUID operation keys. Verified subscriber replacement is one Postgres transaction and preserves the prior row as unsubscribed history.
- Selected sends claim `pending` rows as `sending` before delivery. A crash can
  leave uncertain `sending` rows; the retry uses an explicit Yes/No dialog that
  supplies the stronger internal `RETRY UNCERTAIN EMAILS` API safeguard. The
  operator never types the phrase.
- The email worker requires a durable `worker_run_log` pre-run marker in staging; a missing log table fails closed before delivery.
- Next live email is independently blocked by `JUPR_ENABLE_NEXT_PLAYER_UPDATES_LIVE_EMAIL`. Use `dry_run` or `staging_redirect` for acceptance.
- Published and unpublished recap data uses the same full preview surface. Draft print output is watermarked as operator-only.

## Deferred manual acceptance

Use disposable staging rows. The operator captures the requested screenshots;
the evidence runner records audit/provider IDs.

The read-only browser evidence in `apps/web/e2e/communications.staging.spec.ts`
uses a fresh workflow-minted `STAGING_ADMIN_BEARER_TOKEN`, optional
`STAGING_ADMIN_EMAIL`, and the validated disposable draft for week
`2099-01-05`. The fixture contains only `Test` and `staging only` content and
remains unpublished. The workflow discovers these values after authenticating;
it does not depend on a stored bearer token or stored fixture IDs. The suite
intentionally performs no writes; the steps below remain deferred to the
consolidated session.

1. Generate a recap, open its full unpublished preview, print/save it, change a spotlight selection, save, publish, unpublish, and compare with Streamlit.
2. Open the same recap in two browsers. Save in the first; confirm the second receives a stale-state message and cannot overwrite it.
3. Preview one player digest and confirm no digest/outbox row is created.
4. Queue an exact range twice with the same operation key and confirm only one logical outbox row exists.
5. In `staging_redirect`, send selected pending rows and inspect destination, content, unsubscribe link, audit row, `delivery_mode`, and provider ID.
6. Force one controlled delivery error, retry it, and confirm attempt count/history. Review a simulated `sending` row and verify the stronger uncertain-delivery warning before retry.
7. Delete only a pending disposable row. Confirm sent/skipped history cannot be deleted from this surface.
8. Replace a verified subscriber, confirm one active replacement and an unsubscribed predecessor, then deactivate the replacement. Verify future queueing skips it.

## Recovery and rollback

- A stale response is a stop condition: reload; do not copy old values into a fresh form.
- A `sending` row is delivery-uncertain. Check provider logs and recipient inbox before using the uncertain retry control.
- Queue deletion is limited to pending rows. Delivered history is retained for audit/reconciliation.
- Recap publish mistakes are recovered with the audited unpublish action; prior generated JSON remains in the row/audit trail.
- Subscription replacement preserves the predecessor and linkage IDs. Do not manually reactivate both rows.
- If a staging acceptance step is unsafe or ambiguous, disable the individual API flag and use the Streamlit fallback. Schema rollback should be a separately reviewed forward migration after retaining audit/outbox history.
