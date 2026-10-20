# Badge conflict repair and recovery

This runbook accompanies `20261109004103_badge_conflicts_are_not_serialization_failures.sql`.
Work on this branch is staging-only. The permitted staging Supabase project is
`sijpxjxvdtrehmqvirfi`. Do not apply it or terminate any production backend as
part of staging verification; production promotion is a separately authorized
release action under `AGENTS.md`.

## Why this change is needed

A request with an outdated badge source revision, season revision or tied-result
fingerprint cannot succeed by replaying the same transaction. The old routines
raised SQLSTATE `40001`, which PostgREST treats as a serialization failure.
Supabase documents an infinite retry bug in PostgREST 14 and a fix in version 16:

https://supabase.com/docs/guides/troubleshooting/high-cpu-and-infinite-transaction-retries-when-using-custom-error-codes-in-rpc-functions-77326b

The migration changes only six explicit application-conflict raises in four
reviewed badge routines to `PT409`. It preserves genuine PostgreSQL serialization
failures, row/advisory locks, permissions and all award mutations. If the expected
signatures or raise counts have changed, the whole migration aborts. Reapplying
the corrected definitions is safe. No award, score, rating or revision is
backfilled or reset by this migration.

The Python reconciler handles only the exact stale-program-revision response. It
reads and evaluates a fresh snapshot once, then tries that new payload. A second
conflict remains pending and uses the worker's existing durable one-minute
backoff. Other errors are not retried by this handler. The staging stop switch is
checked before each attempted apply. Admin save conflicts return HTTP 409 for
the operator to refresh; they do not start background work.

## Optional isolated SQL regression

Install `@electric-sql/pglite` into a temporary directory outside this repository,
then pass its absolute module path to the fixture runner, from the repository root:

```sh
PGLITE_MODULE=/path/to/temporary/node_modules/@electric-sql/pglite/dist/index.js \
  node tests/sql/badge_conflicts_pglite.mjs
```

This runs the actual original badge routines, the new migration twice, and both
transaction scripts against a minimal synthetic PostgreSQL schema. It checks
that only the six reviewed raises change, preserving function identity,
privileges and security settings; unrelated `40001` remains intact; all fixture
writes roll back; and unexpected catalog drift aborts the whole migration.
PGlite is an optional test tool, not an application dependency. This does not
replace validation on the full staging schema or prove PostgREST/concurrent
connection behavior.

## Staging deployment and verification

1. Review the migration and test the candidate locally. Use the canonical
   migration directory; do not modify old applied migrations or use
   `scripts/db_migrate.sh`.
2. Verify the exact staging project, then apply the migration through the
   coordinated staging release. Deploy the corresponding Python/API change as
   well: older API code can classify the new SQLSTATE as a generic 503 until it
   receives this update. The error-code change can safely precede the API change.
3. Inspect the four installed functions and execute privileges. Compare to the
   pre-migration definitions: only the reviewed `errcode` literals should differ.
   All six application raises should use `PT409`; no `anon` or `authenticated`
   execution grants should be introduced.
4. On populated staging, use an isolated synthetic club/player. Generate the
   reusable harness with
   `python tests/sql/build_badge_staging_verification.py > /tmp/badge-staging.sql`,
   review it, and execute the **entire** output only on
   `sijpxjxvdtrehmqvirfi`. It embeds the two current SQL regressions, changes only
   their fixture selectors/admin IDs, applies timeouts, and wraps everything in
   `BEGIN ISOLATION LEVEL REPEATABLE READ; ... ROLLBACK;`. The final ten residual
   fixture counts must all be zero. Never run the bare regression scripts on a
   populated database: their default first-player selector can temporarily
   reconcile existing club awards and correctly fail the preservation assertion.
   Do not split the generated SQL into autocommitted statements. Normal internal
   queue/activity sequence increments can remain after rollback; fixture rows
   must not. No existing player, club or award is a test mutation target.
5. Verify a stale season/round-robin request returns HTTP 409 promptly. Verify a
   source revision changing during evaluation causes at most one refreshed
   apply, then the durable backoff. Do not send test player emails.
6. Confirm the exact staging web/API SHA and normal write-wave/email isolation
   according to `AGENTS.md` before marking the release ready.

## Recorded staging verification

On 2026-10-20 at 00:43 UTC, the installed repair was verified on staging project
`sijpxjxvdtrehmqvirfi` using candidate `23a6febe794d41e37d09914f4aff51506ade0e0c`.
Both checked-in regression bodies passed against an isolated synthetic fixture,
with real installed routines and enabled triggers. Assertions covered stale
program/season/tie conflicts, no stale apply side effects, valid replay,
restoration/manual revocation, administrator authorization, live winner writes,
source-dirty revisions, durable backoff and public execution restrictions.
After rollback, all ten residual counts were zero: club, player, award, program
state, season, admin assignment, round-robin receipt, queue, activity log and live
session. The transaction used a 15-second statement timeout, 2-second lock timeout
and 30-second idle transaction timeout. This is SQL verification; PostgREST HTTP
transport and multi-connection races were not exercised.

## Inspect existing retries without interrupting traffic

Use a fixed UTC log window. Count database errors separately from API requests:

```sql
select log_attributes['parsed.sql_state_code'] as sqlstate,
       log_attributes['parsed.context'] as context,
       log_attributes['parsed.process_id'] as backend_pid,
       count(*) as errors
from logs
where source='postgres_logs'
  and log_attributes['parsed.sql_state_code']='40001'
  and log_attributes['parsed.context'] like '%apply_program_badges_v1%'
group by sqlstate,context,backend_pid;
```

In the same authorized database, inspect matching active backends without
printing request arguments:

```sql
select pid,backend_start,application_name,usename,state,wait_event_type,
       wait_event,query_start,pg_blocking_pids(pid) as blocking_pids
from pg_stat_activity
where usename='authenticator' and application_name='postgrest'
  and query like '%"public"."apply_program_badges_v1"%';
```

The function fix does not necessarily stop transactions already inside a retry
loop. If errors persist, an authorized operator must correlate the recent error
PID with the same backend's current identity, start time, application and exact
RPC. Only then stop that one proven looping backend through the SQL editor.
Do not run a blanket termination by database role, kill unrelated scoring
requests or restart the project merely because a connection is active. Backend
PIDs can be reused; recheck immediately before an operator-approved termination.
No automated termination command is included in the migration or tests.

After cleanup, compare another fixed log window: the known repeated business
`40001` signature should disappear, ordinary queries should complete, and badge
state should reach `applied_revision = revision` without modifying it by hand.
Keep the stopped-connection evidence and any interrupted request receipt for
reconciliation. A successful badge revision alone does not prove an old retry
loop is gone.

## Rollback considerations

Do not restore `40001` business-conflict codes as a rollback: that recreates the
incident. If application behavior needs rollback, keep the SQL correction and
pause the badge worker using the established environment controls while
diagnosing. Existing pending revisions remain durable. Do not mark jobs done,
delete awards or acknowledge revisions to clear alerts. Production deployment
contracts must explicitly include this logical migration when a production
release is separately prepared.
