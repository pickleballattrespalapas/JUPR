# Generator compute isolation

Schedule planning in the public/admin/shared generator and Hub service paths
runs in one spawned process per API process in staging/production. Database
reads, authorization, durable operation records, optimistic version checks,
and atomic commits stay in the request process. Only pure domain functions
and their plain data inputs cross the process boundary.

No Fly machine, paid worker service, database migration, or CPU allocation is
added. The child process shares its existing machine's CPU and RAM. Isolation
removes Python GIL competition with the API; it cannot fix exhausted Fly CPU
credits or insufficient machine memory. Measure those before changing capacity.

Admission is bounded to one running and one queued calculation. Excess work
returns HTTP 503 with Retry-After: 2 through the generator and Hub route error
handlers. Reuse the existing operation retry/recovery flow. Never automatically
submit a new operation ID after an uncertain HTTP response.

The planner waits at most 60 seconds. On timeout its caller cannot reach the
subsequent session commit. A running calculation may finish later, but has no
DB handle and cannot save. Its admission slot remains occupied until it ends;
queued work is cancelled where possible. A crashed worker is replaced lazily,
without replaying mutations. Domain validation errors retain their normal
rejection semantics. API shutdown cancels queued pure work.

`JUPR_GENERATOR_PROCESS_WORKER=0` is an emergency rollback to in-process planning.
`=1` enables the same worker in local tests. With no override it is enabled for
`JUPR_ENV=staging` and `production`, disabled for local/unit-test environments.
Keep one API process per current machine until memory and concurrency are
measured: each API process owns a separate planner and admission bound.

Structured `jupr.api.compute` events contain operation, queue_ms, compute_ms,
cpu_ms, and elapsed_ms on success; busy/timeout/crash events contain no request
bodies, player names, scores, session tokens, or DB credentials. Compare CPU time
to wall time and Fly CPU/memory metrics during simultaneous RR and ordinary
page requests. Request elapsed_ms logs still measure full backend requests.

Validation: `tests/test_generator_compute.py` uses a real process and a
19-player/21-round calculation while another API endpoint responds. It also
checks domain-error propagation, timeout admission retention, slot recovery,
and stale broken-pool waiters not shutting down a replacement.
