# Selective production repair candidate

This local candidate is based on production branch `rollback-feb8` at
`143c771129f16da0b25f3aceb0fab26e1a9e19e5`. It has not been pushed, merged,
deployed, or applied to a remote database. Production schema preflight and
operational acceptance remain unverified and require separate authorization.

## Selected source changes

| Original commit | Candidate commit | Change |
| --- | --- | --- |
| `70b6a669` | `f54d9702` | F04/F23 player activity column compatibility |
| `134e4c3b` | `be416edd` | Bounded JWKS key caching |
| `a0bf7157` | `d4bbe6bc` | Structured API logs and query credential redaction |
| `7a2124ae` | `67ad1e3a` | F20 tournament commerce publication visibility |
| `68cfe982` | `77aee3b0` | F22 draft playoff outcome visibility |
| `23a6febe` | `1a846093` | F01 nonretryable badge business conflicts |
| `a635291f` | `f4a9ce2f` | F06 unrated participation activity |
| `940815cf` | `5fa55252` | F06 legacy activity concurrency guard |
| `8c74c5b` | `bf85f673` | F05 support intake requires staff review readiness |
| `9e936948` | `ad998956` | F03 saved-session email scheduler eligibility |

F15's club-directory-dependent public form change and the Next 16 upgrade are
excluded. The later legacy team-match function privilege migration is also
excluded pending independent review. This candidate does not resolve the full
audit or attest that every production vulnerability has been repaired.

## Production compatibility resolutions

- Retained the production tournament-write guard import when adding publication
  classification to commerce.
- Retained production club-scoped badge gates and the production worker's
  reviewed-club selection. Each stale-evidence retry rechecks the requested club.
  Tests cover both retry scope and rejection of an unreviewed club.
- Preserved the existing round-robin CI step while adding badge test coverage.
- Updated the production migration inventory fingerprint, required ledger names,
  deployment order and matching inventory assertions for exactly two new SQL
  migrations. Existing profile, duplicate-ledger allowance, additional-ledger
  policy and schema-only exceptions are unchanged.
- Extended only the local PGlite synthetic schema to accommodate the production
  branch's already-isolated badge transaction fixture.

Both Fly configuration files and production feature/tournament policy modules
are byte-for-byte unchanged from the base. No source change enables support
review or changes the existing email activation policy.

## Required rollout decisions

The new migrations are `20261109004103_badge_conflicts_are_not_serialization_failures`
and `20261109004104_unrated_match_activity`. They inspect installed function
definitions and fail on unrecognized prerequisites. Local SQL tests cannot
prove the current production catalog matches. Apply the reviewed migrations
before the application begins issuing activity-only projections.

The existing production support-review flag remains disabled. The F05 guard
therefore closes new public intake instead of silently accepting unreviewable
requests. Opening intake requires separately reviewed staffing, authorization,
schema readiness and support-review enablement. Existing requests are untouched.

The preserved production configuration has live email and automatic session
delivery enabled. F03 can make previously pending valid saved-session rows
eligible after a future deployment. Review their current disposition before
deployment; do not infer approval to send from this local test or from an old
queue inventory. No SMTP or external send occurred in these checks.

## Local verification

The focused Python selection covered 593 tests, including all selected repairs
and existing production release, feature-gate, email, schema-window and deployment
guards. Its only initial failures were two inventory assertions for the old
migration count/tail. Those expectations were updated without weakening guards;
all 76 tests in the affected compatibility subset then passed.

Both badge SQL transaction suites passed on isolated PGlite PostgreSQL, including
rollback restoration, exactly six changed business-conflict raises, preserved
routine signatures/OIDs/ACLs/security settings, idempotent reapplication,
unchanged unrelated serialization errors, and atomic rejection of catalog drift.
The unrated activity SQL suite passed doubles/singles activity, rating invariants,
compare-and-swap, atomic replay, old-writer compatibility and repeat migration
application. These fixtures do not model full Supabase/PostgREST or concurrent
connections.

Migration source/version guards, TypeScript checking, the tournament-completed
draws component script and support notification-destination component script
passed. The Next production build, including lint/type checks, also passed with
existing CSS/React warning classes. Frontend dependencies remain Next 14/React 18.
Build verification used localhost-only API/Supabase configuration; production
services were not queried.

## Follow-up verification fixtures

Test-only commits `c50858d5` and `ebec0698` were also selected. The activity SQL
fixture now uses int32-compatible player IDs, non-null singles counters and the
installed `Overall` league contract. The badge rollback generator was adapted
to recognize the production branch's existing isolated program fixture, without
changing its assertions or accepting unknown fixture changes.

The updated activity SQL suite passed locally. The generated badge harness also
executed against isolated PGlite, passed both regression bodies, restored the
fixture state and returned zero for all ten residual counts. Thirteen focused
Python fixture/activity tests passed. These follow-ups change tests and documents
only; migration SQL, production configuration and runtime policies are unchanged.
