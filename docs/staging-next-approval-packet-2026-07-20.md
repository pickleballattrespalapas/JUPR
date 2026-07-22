# Next staging write approval packet — data-correction A / dismissal B

This is the complete bounded write plan approved after the 2026-07-20/21 smoke.
It contains exactly two business-data actions against the isolated staging stack:

- **A:** create one disposable data-correction intake and submit one exact retry to
  prove deduplication;
- **B:** dismiss that same request through the authenticated support-request queue.

No other mutation is authorized by this packet. Production is out of scope. Email
mode stays `dry_run` throughout.

## Candidate and infrastructure preflight

Do not run this packet against the old baseline merely because it is listed in the
handoff. After the hardening patch deploys, record the replacement values:

| Field | Required value |
|---|---|
| Canonical branch/ref | `staging` |
| Candidate SHA | `6c27f5de5c04b1d565ba051efb322fff2804ff10` |
| Vercel deployment ID | `dpl_9NtvgLv9GtRbMCUDTHcR2KezobVo` |
| Vercel immutable origin | `https://jupr-mimau97b5-pickleballattrespalapas1.vercel.app` |
| Fly API | `https://juprleagues-api-staging.fly.dev` |
| Fly image for each dispatch | Initial `none`: `registry.fly.io/juprleagues-api-staging:deployment-01KY5M69KJ1DPDFSSRQWNBBSTR` (run `29950338666`, machine `01KY5M7NG6EJBGWXV1TV79978R`); record A, B, and final `none` separately |
| Supabase project ref | `sijpxjxvdtrehmqvirfi` |
| Email mode | `dry_run` |
| Operator | Joe Baumann |
| Witness | Pending — record the distinct witness identity with the witnessed action |

Before A, verify Vercel `/api/environment` and Fly `/health` attest the same final
SHA and isolated staging origins. Begin from `write_wave=none`; never dispatch a
wave from another active write wave.

Run a read-only prestate query before opening A and confirm there is no matching
fixture fingerprint/request from the preceding 24 hours. The intake service
deduplicates identical fingerprints across that window even when the prior row is
dismissed. If a match exists, stop and wait for the window to expire; do not vary
the fixture, delete retained evidence, or claim the first submit is a new row.

### Recorded final-candidate preflight — 2026-07-22

- Vercel deployment `dpl_9NtvgLv9GtRbMCUDTHcR2KezobVo` is `READY`, is sourced
  from `staging`, and reports candidate SHA
  `6c27f5de5c04b1d565ba051efb322fff2804ff10`.
- Fly deploy/verify run `29950338666` succeeded. Live `/health` reported the same
  SHA, `write_wave=none`, `business_data_write_wave_active=false`, email
  `dry_run`, and every controlled write flag false.
- Canonical Staging Smoke run `29950580803` passed public checks, exact
  read-only deployment identity attestation, and strict browser smoke.
- The remote migration ledger includes `public_support_intake_guardrails`.
  `public.public_support_requests` has RLS enabled; `anon` and `authenticated`
  have no DML, while `service_role` has the required DML privileges.
- The prior exact short fixture is retained as dismissed request
  `req_4055accc54b34ab6977d`, created `2026-07-21T14:47:51Z`. At the prestate
  read on `2026-07-22`, it was more than 28 hours old and therefore outside the
  24-hour deduplication window. No matching fixture exists inside the active
  window, so batch A may use the approved values without alteration.

## Short repeatable fixture

Use these exact values. They intentionally honor the owner's request for easy
typing and exact repetition:

| Form field | Value |
|---|---|
| Your name | `Test` |
| Your email | `test@x.invalid` |
| Player name | blank |
| Player ID | blank |
| Match ID | blank |
| Tournament ID | blank |
| Short subject | `smoke` |
| What looks wrong? | `staging only` |
| What should staff change after review? | `nothing` |
| Evidence/screenshot link | blank |
| Staff-contact confirmation | checked |

Equivalent request JSON, for evidence/review only:

```json
{
  "request_type": "data_correction",
  "requester_name": "Test",
  "requester_email": "test@x.invalid",
  "player_name": null,
  "player_id": null,
  "match_id": null,
  "tournament_id": null,
  "subject": "smoke",
  "description": "staging only",
  "requested_action": "nothing",
  "evidence_url": null,
  "consent_to_contact": true,
  "website": "",
  "source": "next_data_corrections_form"
}
```

## A — data-correction intake and exact retry

### Manual dispatch boundary

The GitHub connector cannot dispatch the Fly workflow. The owner/operator must
manually run `Deploy FastAPI staging to Fly` from ref `staging` with:

```text
write_wave = public-intake-auth
```

Record the workflow run URL, candidate SHA, Fly image, machine/release ID, and
`/health` readback. Confirm the selected wave is exactly `public-intake-auth` and
every non-selected controlled write flag is false.

The wave exposes other infrastructure routes, but this packet authorizes only:

```text
POST /clubs/tres-palapas/support/intake
```

Do not submit registration, verified-update, partner-pairing, privacy, support, or
email-preference mutations during A.

### Action and acceptance

1. Reconfirm the 24-hour prestate is clear, then open `/data-corrections` on the
   final Vercel staging candidate.
2. Enter the exact short fixture above and submit once.
3. Record the JSON-bearing 2xx response and new `request.id`.
4. Accept only `accepted=true`, `deduplicated=false`, request type
   `data_correction`, and request status `new`.
5. Re-enter or replay the exact same values without changing whitespace, case, or
   blank fields, then submit exactly once more.
6. Accept only `accepted=true`, `deduplicated=true`, the same request ID, and the
   existing staff-queue message.
7. Verify through a read-only Supabase/admin readback that exactly one matching
   `public_support_requests` row exists. Do not dismiss it during A.

If the first request times out, do not invent a new fixture. Read back the exact
fingerprint/row first; the retry is part of the deduplication proof only after the
first outcome is known.

### A evidence record

| Field | Record |
|---|---|
| Fly run/image | — |
| First response status / artifact | — |
| Request ID | — |
| Exact retry status / artifact | — |
| Deduplicated to same ID | — |
| Exactly-one-row readback | — |
| Operator | — |
| Witness/reviewer | — |

After A is reconciled, manually deploy `write_wave=none`, record that Fly release,
and verify the all-false projection before proceeding to B.

## B — dismiss the same disposable request

### Manual dispatch and admin-token boundary

From the same final `staging` SHA, manually run `Deploy FastAPI staging to Fly`
with:

```text
write_wave = support-requests
```

Record a new workflow run/image. This is a separate release from A. Confirm only
the `support-requests` wave is active.

B requires an authenticated, club-scoped admin identity. Use the existing browser
session at `/admin/support-requests` or a bearer token kept in the operator's
secure environment. Never paste, commit, screenshot, or log the token. A connector
without the real admin session may inspect public/Supabase readback but must stop
at the dismissal boundary and ask the operator to perform it.

### Action and acceptance

1. Load `new` requests and filter type `data_correction`.
2. Select the exact request ID captured in A. Verify subject `smoke`, requester
   `Test`, and email `test@x.invalid` before changing state.
3. Set status to `dismissed`.
4. Use the short admin note `test only`.
5. Click **Save request status**, review the confirmation dialog, and choose
   **Yes, dismiss request**. The UI supplies the internal confirmation value
   `SAVE REQUEST STATUS`; the operator does not type it.
6. Confirm once. The UI sends:

```text
PATCH /admin/clubs/tres_palapas/support-requests/<request-id>
```

with the selected row's `expected_updated_at` and source
`next_admin_support_requests`.
7. Accept only a JSON-bearing 2xx response with the same request ID, status
   `dismissed`, admin note `test only`, non-empty `reviewed_by`, and a new
   `reviewed_at`/`updated_at`.
8. Verify exactly one matching `update_public_support_request_admin` audit event
   for entity `public_support_request`, with before status `new` and after status
   `dismissed`.
9. Retain the dismissed staging-only row and audit event as evidence. Do not delete,
   resolve, reopen, or perform the underlying “nothing” requested action.

On `409`, refresh and reconcile the selected request before any retry. On ambiguous
response, read the request and audit record first. Never submit a blind second PATCH.

### B evidence record

| Field | Record |
|---|---|
| Fly run/image | — |
| Admin actor | — |
| Request ID and captured pre-state timestamp | — |
| PATCH response / artifact | — |
| Dismissed readback | — |
| Audit event ID/artifact | — |
| Retained-evidence owner | — |
| Witness/reviewer | — |

## Required restoration and smoke boundary

After B readback and audit verification:

1. Manually dispatch `Deploy FastAPI staging to Fly` again from the same final
   `staging` SHA with `write_wave=none`.
2. Record the workflow run, Fly image, and machine/release ID as a distinct final
   ledger row.
3. Verify `/health` reports the same candidate SHA, `write_wave=none`,
   `business_data_write_wave_active=false`, email `dry_run`, and every controlled
   write flag false.
4. Verify Vercel `/api/environment` still attests the same final candidate and
   expected Fly/Supabase origins.
5. Only now ask the owner to manually launch canonical `Staging Smoke` from ref
   `staging`. The connector cannot dispatch it. Keep `allow_live_unconfigured` off
   if the live-session route is configured.

`make public-web-smoke` and the legacy `Public Web Smoke` workflow are diagnostic,
not substitutes for the canonical final smoke.

## Stop conditions

Stop without further writes on wrong SHA/project/origin, a wave other than the
named A or B wave, more than one controlled wave enabled, unexpected external
email, response ambiguity, a changed request, a second durable intake row, missing
reviewer/audit attribution, wrong-club visibility, or inability to restore and
attest `none`.
