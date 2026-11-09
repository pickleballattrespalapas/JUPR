# Badge worker ownership and isolation

Queue ownership is additive. The reviewed staging configuration now enables
one dedicated Tres Palapas worker; production remains unchanged. It does not
repair, requeue, or replay any existing job. Existing processing records without heartbeats remain evidence
for review, not proof of running work.

## Preparation and migration order

The worker package is integrated with the current staging calculation reuse,
request timing, and registration changes. The lease migration is already applied
to staging as recorded below. The staging workflow checks its RPCs and columns
and refuses activation while any badge job is processing. It never applies SQL.

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

## Selected staging topology and release sequence

- Keep both DFW API machines at shared-cpu-1x / 1 GB.
- Add exactly one DFW badge worker at shared-cpu-1x / 2 GB, with no HTTP service.
- All processes use external mode scoped to `JUPR_BADGE_WORKER_CLUB_ID=tres_palapas`.
  API processes use `JUPR_BADGE_QUEUE_LEASES=0`; the worker command overrides it
  to `1`. Tres Palapas requests only enqueue. Other clubs keep their embedded
  program processing and legacy queue claims/fallback. Scope validation also
  prevents an accidental global lease flag from changing unrelated clubs.
- Current Fly list pricing checked November 8 gives an approximate $15.86/month
  for this continuously running shared worker in Dallas: (base $2.19 + 1.75 GB
  additional RAM at $6/GB) × Dallas multiplier 1.25. This is additional compute,
  excluding storage, network and tax. It is a starting size to measure, not a
  guarantee of adequate sustained worker CPU. Pricing: https://fly.io/pricing/.

The normal staging deploy has two phases. It verifies the migration and an idle
processing queue, records a sanitized inventory, then stops the previous worker
if present. It deploys only the `app` group first. Exact health must attest the
staging SHA, external pilot scope, API leases off, and dry-run email. Inventory
validation requires both API machines to have their original capacity and the
same immutable image, with any old worker stopped. Only then does it deploy the
`badge_worker` group using that immutable image and `--ha=false`.

Final inventory must contain exactly two API machines and one running worker,
all on the same image; the worker must have 2 GB, the exact scoped lease-enabled
command, and no network services. An extra worker, changed API capacity, foreign
image, or mixed image fails the release. Existing scope/lease settings must not
be overridden manually. Do not activate an old worker or run a second worker
entrypoint from SSH.

The rollout briefly pauses background processing; API traffic continues during
its rolling update. A deployment that fails after stopping the worker needs
operator attention: retain that failure, inspect whether the APIs already defer,
and finish the same reviewed candidate or restore the previous known topology.
Do not treat a healthy HTTP endpoint as proof that badge processing resumed.

Worker logs emit `badge.worker.started`, cycle counts and durations, job lease
heartbeats, and graceful stop. Validate real queue and program progress and
public request timings before claiming capacity recovery. Staging idle cycles
alone do not establish production performance. Keep all emails in dry-run mode.

Production needs a separately reviewed cherry-pick, its migration applied and
verified against its own ledger, preservation of its badge feature policy, and
explicit production release authorization. This staging rollout does not make
any production change or recover stranded historical jobs.

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

## Measurement and acceptance

Record the exact release image and machine IDs with before/after windows. Use
request timing logs to compare the same routes, cache state and representative
traffic, plus CPU throttling/burst credits, memory/restarts, queue age and job
completion, and program revision/applied_revision. Never infer p95 or recovery
from a handful of health probes. The preliminary staging check is a bounded
smoke test, not a load test. Keep API upgrades deferred until these measurements
show their remaining bottleneck.

Graceful shutdown is regression-tested with a live heartbeat: the current job
finishes, the heartbeat joins, and no subsequent phase claims new work. Fly's
300-second shutdown allowance is best effort; a forced stop leaves unresolved
claims for explicit review. There is no automatic expired-job replay.
