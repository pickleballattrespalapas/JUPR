# Support intake and staff review

Public support, correction and privacy intake must have an enabled staff review
path. An accepted request becomes a `public_support_requests` record, then an
in-app staff notification. It does not send email, apply a data correction or
change profile visibility. Staff must perform any approved change through the
appropriate authorized workflow and record completion in the queue.

## Runtime contract

`GET /clubs/{slug}/support/intake/status` returns `accepting_requests` and the
validated public club identity. It uses no-store and discloses no queue data or
counts. The POST rechecks the same contract before inserting, including retries.

Acceptance requires all of:

- `JUPR_ENABLE_NEXT_ADMIN_SUPPORT_REQUESTS=1`;
- a write policy permitting both public intake and staff review updates;
- the public-intake environment flag where applicable;
- a successful read of the queue's required review columns.

| Environment/posture | Staff review | New public requests |
| --- | --- | --- |
| Staging `open`, reviewed flags enabled | Available when schema is ready | Available when schema is ready |
| Staging `support-requests` | Available when schema is ready | Closed |
| Staging `public-intake-auth` | Closed | Closed: intake alone cannot operate support |
| Staging `none` | Closed | Closed |
| Production `read_only` | Closed | Closed |
| Production policy `enabled` and support-review flag enabled | Available when schema is ready | Available when schema is ready |

Local/test environments still require the support-review feature flag. A closed
or missing global write policy stays closed. Disabling the review flag also
stops new intake; it does not delete existing requests. A schema/read failure
returns unavailable, never an empty healthy queue. The global boundary may
return 403 before the support route; otherwise unavailable intake returns 503
with a safe contact message and no insert. The admin readiness response exposes
`ready_for_review`, separately from the feature flag's `enabled` value.

The staging environment checker attests both readiness endpoints against the
selected write wave. A boolean flag alone is insufficient evidence of readiness.
The checked-in production and staging Fly configurations are unchanged. In
particular, the production read-only contract must not be loosened just to make
this feature's check pass.

## Local acceptance and isolation

Run `tests/test_api_contract_support_workflow.py` with the API dependencies
installed and `JUPR_CI_REQUIRE_API_TESTS=1`. It uses TestClient and in-memory
database adapters, not a remote database. For all three request types it tests:

1. Validated club intake and exact-retry deduplication.
2. A club-scoped in-app notification linking to the specific request.
3. Authenticated assigned staff listing and moving the request into review.
4. Optimistic-concurrency rejection of a stale update.
5. Completion, audit entries and removal from pending notifications.
6. The additional identity, action and evidence requirements for privacy.

It also checks unknown clubs, cross-club player IDs and request IDs, revoked and
read-only assignments, queue outages, all relevant staging waves and production
read-only policy. SMTP transports are replaced with failing stubs; email mode is
`dry_run`. The test is included by CI's `test_api_contract_*.py` selection.

## Reviewed staging acceptance before a production repair

- Use the canonical staging deployment/handoff for the exact integrated SHA.
  Keep `JUPR_EMAIL_MODE=dry_run`; do not open production overrides.
- Confirm a named staff reviewer has a current club assignment and will monitor
  the queue/notifications. Technical readiness does not prove staffing.
- Perform a controlled request in each staging club; confirm only its assigned
  staff can read/update it. Follow the in-app link, review and complete it, then
  verify the audit entry and notification removal. Verify no email was sent.
- In a controlled staging test, close review and confirm public intake no longer
  accepts writes and public forms offer the organizer/email contact route.
- Review previously accepted requests separately; do not bulk-resolve them or
  assume the source change processed a backlog.

These steps still require staging acceptance; local tests do not certify live
database write privileges or authenticated browser behavior. No deployment,
production configuration change, data repair or email delivery is part of this
source change. A separately authorized production repair must enable only its
reviewed support capability alongside the existing approved write policy, and
verify both readiness endpoints. Do not merge the staging branch's entire
feature/configuration inventory into the production branch.

## Public form club context (separate F15 change)

Generic support/correction/privacy pages require explicit club selection. A
player's privacy link carries its club and player context; the API still
validates the player against that club. Forms do not silently default to Tres
Palapas. An unavailable/missing club cannot fall through to another club's
queue. Details remain in the form after a failed POST.

This UI uses the existing staging public club directory (`getClubDirectory`,
`GET /public/clubs`), which lists only published, active, listed clubs. The
directory was absent from the audited production web/API release. Keep this
change separate from an urgent F05 backport until that directory capability is
independently included and verified; do not import the wider staging club
creation feature just to satisfy it. Direct links with a known club context
work for unlisted clubs without advertising them in the directory.
