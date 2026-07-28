# Team leagues and league awards — acceptance criteria

These criteria describe the approved staging candidate. Payment remains offline. All new writes are staging-only, permission-gated, audited, idempotent where retries are possible, and disabled in production.

## Fixed-partner team leagues

- A club owner can attach team-league settings to an existing club league, open or close registration, set the venue, start date/time/timezone, registration deadline, and choose whether substitutes are allowed.
- Players can register as a two-player team or join the solo partner waitlist. A team is not confirmed until the invited partner accepts through a single-purpose, expiring link.
- A lost registration response is recoverable after a refresh even when the browser creates a new request key. An exact normalized request returns the original durable receipt; a changed repeat involving an already-active player conflicts instead of creating or hiding a second registration.
- Invitation email remains a dry run by default and redirects to the configured staging recipient when staging delivery is enabled. Secrets are not placed in query strings or retained in browser history.
- The same two partners remain the registered roster for the season. When substitutes are disabled, only those partners may play. When enabled, a substitute must be an active player in the same club and cannot belong to another active team in the league.
- Admins can pair exactly two waiting players into a team or withdraw selected waiting entries. Each action is atomic, version-safe, idempotent, and audited.

## Regular season and playoffs

- The regular-season preview produces a complete round robin: every team meets every opponent exactly once, each team has no more than one match per week, and odd team counts receive one bye per round.
- Weekly local start time is preserved across daylight-saving transitions.
- Publishing requires the exact reviewed preview plus unchanged schedule, standings, roster version, and confirmed-roster fingerprint. A stale preview cannot publish.
- Playoffs are optional: none, top-two final, top-four single elimination, or an all-team single-elimination field. Brackets preserve deterministic seeding, place the top two seeds in opposite halves, and create explicit byes when needed.
- A regular-season result cannot be corrected after playoff seeding has created playoff fixtures.

## Results, ratings, and recovery

- Played results require two valid players per side, nonnegative non-tied scores, and a winner matching the score. They publish one canonical doubles match and then finalize the fixture under a stable fixture-specific key.
- Forfeits update the fixture and standings without creating a rated match.
- Standings rank by record, then head-to-head among tied teams, then point differential, with deterministic final ordering.
- Interrupted score operations remain visible. Recovery inspects the canonical direct-match receipt before allowing finalization or compensation; a committed match can never be compensated.
- Reads and writes are club-scoped. Admin routes require league-management permission. Database tables enforce RLS and expose no direct public table access.

## League analytics and awards

- Analytics include only canonical, nondeleted, nonexcluded, valid, non-tied league matches. Every report returns included/excluded counts, exclusion reasons, rule version, included match IDs/dates, and the expected season length when known.
- Player measures include current rating, rating gain, games, record, win percentage, points, differential, average margin, streaks, close-game record, upset measures, opponent strength, wins above expected, partnership measures, partner variety, weeks played, and attendance.
- Close games use a two-point-or-less margin. Upsets compare the two teams' complete pre-match average ratings and require at least a 0.25 JUPR disadvantage. Opponent strength is the average complete pre-match opposing-team rating.
- Partnership reporting names the partner with the best win percentage, then uses games together and name/ID for deterministic ties. Attendance is distinct league weeks played divided by the configured or scheduled season length; it remains unavailable when season length is unknown.
- Expected wins and over-performance are shown only when all four pre-match ratings are available; missing ratings never become an invented 50/50 expectation.
- Team measures include games, record, win percentage, points for/against, differential, head-to-head score, and standing.
- Before freeze, a club owner can enable any supported player or team award, choose top one/two/three, and set the relevant minimum sample. Choices are version-checked, audited, and locked after freeze.
- Ties at the configured cutoff are deterministic and retained as co-winners with distinct durable award identities, subject to the bounded award-result limit.
- Freeze stores the exact configuration, computed awards, player/team measures, and provenance used for later preview and audit.
- Player-winner overrides require an eligible league player and a reason when changed. Team awards stay bound to the frozen team standings. Durable award result records preserve the computed recipient/metric, the selected recipient's frozen metric, the reason, and the exact source snapshot.
- Player badge minting is verified by readback. Categories without a player-badge definition, including team awards, remain durable award records and do not create an invalid player badge.

## Usability and isolation

- Public registration, partner confirmation, team-league administration, settings, bulk roster management, and award reporting work at normal laptop and mobile widths; wide data tables scroll within their cards.
- Public registration errors use an alert treatment distinct from successful status messages.
- League Manager exposes separate overview, settings, roster, team leagues, live rounds, and awards pages without removing existing workflows.
- No feature accesses the production API or production Supabase project. No staging write window is opened by development or automated verification.
