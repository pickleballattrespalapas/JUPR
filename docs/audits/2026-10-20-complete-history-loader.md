# Complete shared history loading

The shared loader previously made one request per table and requested only the latest 5,000 matches. A server row cap could silently shorten players, league ratings, metadata, badge awards and matches. Match-exclusion reconciliation correctly rejected incomplete match history, but could not operate above the limit.

The loader now reads ordered pages for all six tables, advances by the actual rows received, and checks exact counts before and after reading. Missing payloads, premature empty pages, duplicate/missing primary keys and changing counts fail instead of returning partial data. Match order remains newest ID first and club/soft-delete filters remain in every request. The default match load is complete; explicit preview limits remain supported. Correction preflight and recovery now request full history and retain their final exact-count guard. Schema-degradation behavior and stored player-activity snapshots are preserved.

Verification: 355 focused tests passed. Coverage includes 6,203 matches, 1,202 players/league ratings, 1,002 league records and 1,102 badge awards with a 137-row server cap; cross-club/deleted-row exclusion; actual PostgREST client serialization; page errors, missing counts, repeated pages and membership changes; old-job contracts, recovery and direct-match/generator approval behavior. A real canonical badge evaluation retains a sole win older than the former 5,000-match window and removes that award after excluding the match.

One existing test, test_tracked_replay_fences_every_projection_write_batch, fails on unchanged staging 1afc9947 because it expects five heartbeat calls to equal four RPC calls. It was reproduced on the baseline and excluded from the final targeted run; its runtime and assertions are unchanged.

No database migration, historical replay, email change, or production feature activation is included. Counts do not provide a transactional snapshot: equal-count replacements or concurrent field edits are still governed by existing optimistic-concurrency/replay guards. Full history also uses more memory than a truncated result. Match correction activation still requires the gated full-operation acceptance rehearsal; these tests establish loading and badge-plan behavior, not live mutation acceptance.

The player-specific public match-history endpoint was inspected and already uses complete player-filtered paging; it is not changed here. New tests are included in API CI.
