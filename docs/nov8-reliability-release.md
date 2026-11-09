# November 8 PCS reliability repair

Status: implementation and verification; production release has not been performed. Joe selected calculation improvements plus a separate worker, followed by measurement before any API capacity upgrade.

## Core candidate

- Include all historical weekly participants, including inactive players, while preserving current standings and award eligibility rules. Week 4 evidence is 100 matches, two rounds, 50 players per round and 52 distinct participants; 18 participants were omitted from the active-only display.
- Preserve ten five-player courts when edited attendance still fits. A failed saved-session read remains a retryable recovery state rather than Setup.
- Confirm committed League Live matches and save the round before badge evaluation. Read back the exact publish operation after a lost response. Preserve production notification policy.
- Require fresh acknowledgement before publishing a non-today played date. Tres Palapas uses America/Mazatlan for the default date.
- Bound and reuse revision-validated leaderboard statistics, and narrow badge-earner reads to the selected badge. Missing or changed revisions bypass reuse.
- Log request duration, database call wall/CPU observations and calculation phases, with route templates and no player or credential values. These measurements do not partition all network, queue and CPU wait.
- Remove the historical badge-activation command from routine production deployment. Core changes require no schema migration.

## Selected performance approach

Keep the two existing DFW API machines at shared-cpu-1x / 1 GB. The checked-in candidate `fly.toml` now preserves that size; the earlier performance-1x / 2 GB proposal is deferred. This candidate contains calculation/read optimizations and request timing, but does not provision or activate a worker.

Complete the separate worker candidate's staging validation and deploy one worker with its own CPU and memory in a reviewed rollout. Establish its concrete region, machine size and recurring cost before the production release. Compare representative request latency/errors, API CPU throttling and memory, and worker queue progress before deciding whether the API machines still need upgrading.

The incident evidence remains exhausted shared CPU burst credit, approximately 90% quota throttling and one observed 1 GB OOM exit. Neither the optimizations nor worker isolation has yet demonstrated production recovery. Production promotion, migration and worker activation remain separate release actions; selection of this approach does not attest that those actions have occurred.

## Retained incident measurements

All local times below are America/Mazatlan on November 8, 2026. These are historical observations, not a new monitoring run.

| Window | Observation | Interpretation |
| --- | --- | --- |
| About 14:50 | Website /api/environment 200 in about 5.1 s; API /health and /clubs/tres-palapas read timeouts around 19.8 s | Synthetic times include the checking environment; no percentile estimate |
| About 15:09 | One API machine OOM exit 137 and restart | Memory exhaustion confirmed; leak or responsible job unproven |
| 16:00–16:04 | Both machine burst balances zero, about 90% throttling; utilization about 5–14% | Shared quota pressure, not 100% CPU utilization |
| About 16:02 | 55 completed API requests, all 200; 36 over 5 s and 11 over 30 s; leaderboard 10.805–42.164 s; badge earners 62.961 s | Bounded sample, not all traffic; successful status did not mean fast response |
| 16:00–16:05 | 4,257 Python/Fly Supabase requests, all 200; 4,242 timed, mean upstream service 3.73 ms, maximum 59 ms | Database-origin service metric only; excludes full client/network wait |

Vercel runtime logs also showed timeout, socket/connection failures, 503s and requests reaching the 300-second runtime limit. Observability schema was unavailable (404); no metrics is not evidence of health.

## Separate unresolved work

1. **Historical played dates.** Both completed Week 4 rounds contain 100 matches dated November 1 despite publication on November 8. A date-only update is unsafe: replay changes club-wide rating chronology and related projections, while saved round exports and notification evidence can remain stale. Preserve Week 4 tags, exact IDs, players and scores; simulate original/corrected ordering first and prepare an audited, idempotent correction with exact preconditions. Do not rewrite original publish receipts or republish matches.
2. **Worker isolation.** Lease-aware ownership and external worker code are a separate candidate. Test SQL roles/RLS, concurrent claims, heartbeats and fenced completion in staging before activation. Existing or expired processing jobs are never automatically replayed. One separate worker is required; CPU isolation is not achieved by another process on the constrained API machine.
3. **Stranded jobs.** The original Round 1 job remains evidence for exact-job reconciliation, even though both publish operations completed. Verify source effects and former worker state before any recovery. Age alone is not permission to replay.
4. **Corrected results email.** Use exact session/receipt membership and all 52 participants. Match outcomes and points can be reported; withhold corrected rating gains, movement and affected awards until date correction is reconciled. Draft only; sending is not authorized.
5. **Recovery verification.** Match production branch, web deployment and both API machine images. Review comparable passive latency/error, CPU and memory windows with representative traffic. Use at most the bounded lightweight probes authorized by Joe. One fast health response does not establish recovery.

## Release gates

- Stage core changes through a reviewed PR into staging. Require the exact merge SHA's successful Fly deployment and handoff artifact attesting matching Vercel identity and ready_for_manual_testing.
- Production promotion uses only selected fixes on rollback-feb8, never a wholesale staging merge. The existing API image is e5d832fee3803a33f8c040ec7458955e67978aa1; the web/production base is ed899b6baf519f0f86418bfad9a354b8538d475a. Their intervening diff contains no API implementation changes.
- The production migration contract remains unchanged and must pass against the actual ledger. Do not alter its expected head merely to bypass a check.
- Exact approval must cover the production branch merge (website deploy) and API release. This candidate preserves API machine sizing and includes no worker provisioning. A worker production release must separately cover its migration, coordinated activation and concrete recurring cost. Existing runtime workers resume normal activity on restart; there is no historical backfill or explicit email-send step in this candidate.
- Capture the prior immutable API image, configuration and live feature profile before release. API rollback does not roll back the website, schema or business records.
