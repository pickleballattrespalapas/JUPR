# Site reliability repair

This staging candidate addresses the October 29 audit across public pages,
tournament administration, public statistics, adaptive round robins, and API
schedule computation. No schema migration or paid capacity change is required.

## Behavior

- Club layouts and built-in sharing metadata read a compact published shell.
  Publication/display settings remain fresh and fail closed, with a local retry
  panel. Logos load independently and revalidate publication before a 304.
- Tournament status failures retain navigation and show a retry panel; admin
  workspace/auth checks remain in place.
- Leaderboards query badges for visible players and highlights, reusing public
  badge data for 30 seconds with bounded single-flight caching. Failed optional
  badge reads retain rankings with a visible availability notice. Player history
  resolves only participant identities; sharing previews use small identity/title
  endpoints rather than loading profiles or full schedules.
- Active adaptive standalone RRs materialize the current and next round.
  Distant previews are generated when selected. Batched departures use one
  regeneration. Fixed/perfect schedules and Moneyball retain their full schedule;
  the Hub retains its own court/queue planning. Recorded/active games stay fixed.
- Pure generator/Hub planning uses one bounded subprocess on each existing API
  machine. Database reads/writes and concurrency checks remain in the parent.
  See [compute operations](generator-compute-isolation.md) for timeouts,
  admission limits, metrics, and rollback configuration.

## Verification

The new regression suites cover publication failure/unpublish, optional logo
failure, local retry, badge failure/cache expiry/cross-club isolation, lightweight
sharing, deferred navigation in both directions, fairness and perfect rotations,
batch departures, retained score drafts, worker timeout/no session save, and API
responsiveness during real process-based computation. Existing generator/Hub,
public profile/leaderboard, and web component suites are also exercised.

Local replay of the audited round 4/21 roster: three sequential departures went
from 6.2944 seconds to 0.5459 seconds; one batch took 0.1865 seconds. Round 5 matchups
were identical and rounds 1–4 unchanged. These are local measurements, not a
production performance guarantee. No real player/session fixture is committed.

## Staging and production acceptance

Verify the deployed staging `/public/clubs/tres-palapas/site/shell` projection
through PostgREST, logo conditional requests, player/session sharing metadata,
and ordinary page latency during roster changes. Check that the exact deployed
SHA has a successful staging workflow and ready-for-manual-testing handoff.

Production currently differs from canonical staging: its older root layout
still performs a full club-site read. Any eventual production release must also
include staging's existing fetch-free root layout/PublicFooterNav navigation
change, or explicitly port the compact-shell approach there. Cherry-picking
only this candidate without reviewing that difference leaves an old dependency.

Before increasing Fly capacity, compare queue/compute/CPU timings with actual
machine CPU and memory metrics under simultaneous generator and page traffic.
The subprocess shares existing CPU/RAM; it does not provide extra physical
capacity. Production release requires Joe's separate explicit authorization.
