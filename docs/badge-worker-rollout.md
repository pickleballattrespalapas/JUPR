# Badge worker ownership and isolation

This change is additive and opt-in. It does not repair, requeue, or replay any
existing job. Existing processing records without heartbeats remain evidence
for review, not proof of running work.

## Preparation and migration order

This separate worker package was prepared against staging commit
`6dbf8dfe06f7ab1191957e470bb274398754c064`. It is dormant until the migration,
deployment and activation steps below are separately reviewed. No migration
application, worker provisioning or runtime-setting change is part of this PR.

Before applying anything, verify the current remote staging head and review any
changes since that preparation commit. Compare the hosted staging migration
history with the repository at the exact reviewed release SHA. The new migration
is `20261108234006_badge_queue_worker_leases.sql`; its queue-table prerequisite is
`20260718141016_badge_eval_queue_atomic_club_claim.sql`. The preparation base
already contains migration filenames through `20261109011200`, later than the
new file's timestamp. Repository presence does not establish hosted application
order. Record whether those later versions are applied and explicitly review
the out-of-order plan for this one additive migration. Do not blanket-apply all
pending migrations or rewrite applied history to make the versions align.
Confirm the target is staging project `sijpxjxvdtrehmqvirfi` before application.

The lease SQL regression runs in an in-memory PostgreSQL instance using the
repository's pinned PGlite dependency; it needs no hosted database or credentials:

```sh
npm ci --prefix tests/sql --no-audit --no-fund --ignore-scripts
npm run --prefix tests/sql test:badge-worker-leases
```

The migration-source-guard CI workflow runs the same command. It verifies oldest
pending claim order, one unexpired lease per club, wrong-token rejection, expiry,
fenced completion, untouched legacy jobs, no expired-job replay and RPC grants.
This isolated check does not replace hosted migration-history and RLS checks.

The candidate's production migration contract includes the new SQL fingerprint
and requires the logical ledger name `badge_queue_worker_leases`. This describes
what a future reviewed production candidate must have; it does not attest that
production has the migration. Production has not received this migration and
must fail the deployment ledger check while the name is absent. The staging
ledger timestamp `20261109000605` is not a production expected-head value. Leave
production head configuration unchanged until the exact production application
is separately authorized and its resulting ledger is verified.

## Release sequence

1. Apply the new queue-lease migration to staging and verify the SQL behavior
   fixture, role grants, and RLS. Existing historical migrations have later
   timestamps, so check migration history before applying this migration.
2. Deploy the worker-aware API and worker code with the existing topology. The
   default `JUPR_BADGE_QUEUE_LEASES=0` preserves compatibility until the new RPCs
   are verified. Never mix old queue consumers with lease-enabled consumers.
3. In a reviewed release, set `JUPR_BADGE_QUEUE_LEASES=1` for **all** queue
   consumers. Check claimed-job heartbeats and fenced completion in staging.
   Lease-enabled match saves only enqueue; they never run the inline fallback,
   including when enqueue fails or ownership is lost. Failed enqueue returns
   `queue_error` and saved-source recovery references, suppresses dependent
   notifications, and emits `badge.job.enqueue_failed`. It does not pretend a
   missing queue write succeeded. The already-committed matches are the durable
   recovery source; do not submit those matches again.
4. Provision **one** separately supervised badge process for this club, running
   `python -m jupr_app.workers.isolated_badge_worker --club-id tres_palapas`.
   It requires `JUPR_BADGE_WORKER_MODE=external` and leases enabled. Give it its
   own CPU/memory allocation; a second process on the same constrained machine
   does not isolate capacity. Route HTTP only to API processes. The same
   external-mode value disables the API's embedded program-badge loop.
   Common inline queue drains in match-save flows also defer to this process.
   An unrelated explicit badge-recompute/admin operation is not made a queue
   operation by this switch; keep those operations out of peak public traffic.
   The program-badge processor also refuses polling from API contexts in external
   mode. With two API machines and one dedicated worker, only the latter polls.
   Program award application already locks the club revision and matches award
   identity for idempotent application; it does not lease the calculation itself.
   Do not horizontally duplicate the dedicated worker without adding a shared
   program-calculation lease first.
5. Verify queue progress, program-revision progress, heartbeat, CPU, memory and
   public latency before declaring isolation successful. A graceful stop waits
   for the current cycle; a forced stop leaves its claim for review.
6. Repeat the reviewed sequence for production. Promote only reviewed changes;
   staging and production have different feature-policy code. Production's
   explicit badge feature policy remains required by the isolated entrypoint.

## Ownership behavior

Each new claim receives a database-generated token, worker ID, start time,
heartbeat and 120-second lease. A heartbeat runs every 30 seconds. At most one
unexpired **leased** job is claimed per club; the claim transaction is short.
Job computation remains synchronous in the worker and can exceed its drain-loop
budget. Logs report job wall time and worker-thread CPU time separately.

Heartbeat and completion require the current token and an unexpired lease.
Workers verify ownership before incremental facts and award writes. On ownership
loss, processing stops before its next effect boundary and no completion is
acknowledged. An already in-flight database request cannot be canceled by this
guard. This is acknowledgement fencing, not an atomic fence around every award
write. **There is deliberately no automatic expired-job reclaim.**
Existing source edits that intentionally requeue a job can change ownership;
this change does not add transactional fencing to the separate award-write RPCs.

## Existing or expired claims

Use a bounded, read-only query on the indexed club/status fields to inspect job
ID, event type, start/heartbeat/expiry, worker ID, attempt count and finish time.
Do not include player lists, payloads or credentials in diagnostics.

Before any explicit recovery, stop or conclusively identify the former worker,
verify the source event and already-persisted awards/facts, then use a reviewed
idempotent recovery for that exact job. An expired lease alone does not prove
that an in-flight write stopped. Legacy claims have no lease and cannot be
safely recovered just from their age. Never republish recorded matches to repair
a badge job. No automated recovery command is included in this release.

## Rollback

Keep the additive database columns/RPCs. Stop the isolated worker gracefully,
verify no in-flight processing remains, then restore embedded mode with the
worker-aware code. Do not deploy a legacy consumer while leased processing jobs
remain active. Reconciliation of unresolved claims is a separate reviewed action.

## Hosted staging verification (November 9 UTC / November 8 Mazatlan)

The reviewed SQL was applied only to staging project `sijpxjxvdtrehmqvirfi`,
recorded by the management connection as migration `20261109000605`, logical
name `badge_queue_worker_leases`. The source migration keeps its generated
`20261108234006` filename; logical-name and SQL-content matching are necessary
when comparing this hosted ledger. The preceding hosted head was
`20261103141122`; later-dated repository files are historical source copies,
not permission to apply them again.

Verified against the hosted database as `service_role`:

- all three RPCs are invoker functions, callable by service role, not anon or
  authenticated; queue RLS remains enabled;
- claim, heartbeat and completion succeed only for the current unexpired token;
- a second active claim is refused, expired ownership cannot complete, and
  legacy/expired processing rows are not replayed;
- two concurrent service-role claim calls against two isolated synthetic jobs
  returned one claim and zero claims; both synthetic rows were removed;
- the transactional behavior fixture was rolled back;
- security-advisor groups were unchanged and showed no new-function findings.

No worker was provisioned and no activation flags changed. End-to-end worker
progress, graceful shutdown and public latency still need validation with a
separately provisioned staging worker before production isolation approval.
