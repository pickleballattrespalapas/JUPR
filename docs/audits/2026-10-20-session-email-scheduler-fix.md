# F03: automatic session email eligibility

This candidate is rebased onto production `028752dbc86252ff5596b856d7cbacef9a2eb137`.
It is not deployed. Its 101 focused tests passed again using mocked SMTP.
No live email is sent by this validation.

Joe rejected 12 historical session emails on October 20. A separately authorized,
transactional delete removed exactly those 12 still-pending, zero-attempt,
row-version-1 session entries. Verification found zero remaining target IDs.
Match, rating, and subscription records were not changed. The old backlog must
not be recreated. Future eligible session messages can send after deployment
under the existing production activation flags; deployment requires separate
exact approval under AGENTS.md.

## Branch difference

Staging at `ac0dfdd4` already contains the exact-session producer and sender, but
does not contain `player_update_email_scheduler.py` or its API startup hook.
The defect is in the production-only scheduler. The separate staging feature
worktree remains unchanged; this patch should not be treated as a staging
activation of automatic delivery. Any later integration needs to account for
the branch divergence explicitly.

## Corrected contract

The producer saves a nonempty `digest_snapshot_json` with `update_scope=session`,
exact committed `match_ids`, session evidence, and badge references. Its operation
key is a UUIDv5 of the club and normalized match IDs. The previous scheduler
looked only for an empty snapshot and a legacy date-window key, so it could not
select those rows.

The producer and scheduler now share the existing session-key calculation; the
algorithm and already-saved keys do not change. The scheduler selects pending
session snapshots and checks their exact-game key before passing the saved row
version into the existing worker. Manual preview keys, invalid scopes, and empty
legacy snapshots remain ineligible. The sender still uses the exact saved game
IDs and evidence, waits for the referenced badge jobs, atomically claims rows,
and leaves uncertain deliveries for operator review. No retry, backfill, schema,
SMTP, sender, or activation-flag behavior is changed.

## Historical audit context (superseded by the deletion above)

The audit retained these safe historical details:

| Created at (UTC) | Saved games | Attempts at audit time |
| --- | ---: | ---: |
| 2026-10-14 04:41:46 | 14 | 0 |
| 2026-10-18 05:13:10 | 5 | 0 |

These are historical observations, not a current queue inventory. The later explicit disposition is recorded above.

Before any separately authorized production rollout, inspect the current queue
and decide whether the old sessions should still be delivered. Match each row
using its current club, saved scope, and operation key; do not rely on the old
count, timestamps, or pending status alone. A row that is still pending and has
the valid session contract will become automatically eligible after deployment
when all existing activation gates are enabled and its badge jobs are ready.
Deploying this fix with the scheduler already enabled can therefore send those
historical sessions. Hold automatic delivery until that disposition is reviewed
if historical messages are not intended.

Rows already sent or claimed are not selected. Error or uncertain rows are not
automatically reset. Empty legacy snapshots remain excluded; any explicit future
requeue must recover exact committed IDs from the original session receipt,
never infer scope from a date window. Do not blanket-reset or backfill the queue.

## Local validation

The regression harness runs the real producer, scheduler, worker, badge-readiness
checks, subscription checks, claim/update functions, and sender. An in-memory
database models detached reads and row-version changes; digest computation,
rendering, and SMTP are mocked. Tests cover exact IDs and evidence, two sessions
on the same day, producer and repeated-poll idempotency, manual/legacy/malformed
scope exclusion, delayed badges, competing claims, edits after selection, worker
markers, and dry-run guards. No external send occurs.

The installed Supabase client also executes the scheduler query through an
in-memory HTTP transport, verifying its JSON filter encoding without a server.
A replay of the same real-produced row against the baseline scheduler selected
zero updates; the corrected scheduler sent exactly one through mocked SMTP.

Focused command:

```sh
python -m pytest -q tests/test_player_update_scheduler_pipeline.py tests/test_production_email_activation.py tests/test_player_update_session_scope.py tests/test_player_update_queueing.py tests/test_player_update_email_worker.py tests/test_admin_communications_service.py
```

Result: **101 passed**, with nine existing FastAPI/Starlette deprecation
warnings. `git diff --check` also passed.

The local harness does not validate deployed credentials, an actual PostgREST
server, SMTP deliverability, or the current production backlog. Existing batch
limits and scheduling fairness remain unchanged. Deployment and controlled
operational verification are separate from this local patch.
