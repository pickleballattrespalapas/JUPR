# PCS audit: second repair batch

Base: staging `0cc70b4bd920caa02e69f7afe02e7b2f5ba69473`.

## Changes

- F07, results services: league/admin league histories, league ratings, player lookup, and standard tournament draws/teams/games/podium/player lookup now read ordered pages until an empty page. Server-capped short pages do not terminate the read. Club/league/tournament filters are rebuilt on every page. Failed reads propagate rather than publishing a partial history as complete. Stable keys are ID, league name within a club, or player ID within a club and league.
- F12: newly generated weekly report tournament links use `tournament_id`. Both report sanitizers accept that parameter and the older `tournament` parameter. The results page accepts old published links while preferring the canonical parameter. Other URL restrictions remain intact.
- F21: local tied games encountered during interclub rating replay count appearances and activity but neither wins nor losses. They receive no winner bonus and retain a zero match delta. Decisive results retain existing behavior.

## Verification

- 85 focused Python tests passed (public/admin results, API contract, weekly reports, interclub replay).
- 183 related Python tests passed (generator submissions, activity, tournament setup, award projections, highlights, session callers).
- Regression fixtures cover 1,408 league games / 17 weeks and 1,080 tournament games, including server caps of 137 and 1,000, tenant/deletion exclusions, failed subsequent pages, canonical and legacy links, unsafe link suffixes, and tied/decisive games with bonuses.
- Four web component scripts passed: weekly-report, tournament-completed-draws, weekly-recap-builder, weekly-recap-delete.
- Next webpack build and TypeScript no-emit check passed; git diff whitespace check passed.

## Scope and remaining work

No database migration, historical rating replay, live email, or production deployment is part of this staging change. The generic legacy data loader and separate public match-history endpoint still need the remaining F07/F08 work. Offset paging is not a transactional database snapshot; simultaneous history edits remain a potential consistency limitation. The production Next upgrade and versioned match-exclusion badge contract remain separate repairs.

The 12 historical player-update outbox rows rejected by Joe were deleted in a separately authorized production transaction on October 20. Exactly 12 guarded pending/session rows were deleted and zero target IDs remained; no emails were sent. This deletion did not touch match, rating, or subscription records.

The production-only scheduler fix is held in a separate candidate based on production `028752dbc86252ff5596b856d7cbacef9a2eb137`. Its 101 focused tests pass with mocked delivery. Deploying it requires Joe's exact production approval under AGENTS.md and can enable future eligible session emails under existing live flags. Staging email remains dry-run.
