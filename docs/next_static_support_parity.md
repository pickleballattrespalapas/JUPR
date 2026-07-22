# Next static, policy, and support parity

This slice closes the automated-ready gap for Rating Rules, FAQ, Privacy, Terms,
Contact/Support, Data Corrections, and Profile Privacy. It does not mark those
matrix rows `Done`; copy/legal approval, real inbox handling, and a staging queue
review remain manual gates.

## Public behavior

- `/how-ratings-work` is the canonical rating-rules route. It covers official and
  unrated sources, Python-authoritative movement, corrections/replay, badges,
  differences from external ratings, and admin integrity. `/faq` cross-links it.
- `/support` retains a populated email fallback and adds a durable
  `general_support` form backed by the same staff-review queue as data correction
  and profile privacy requests.
- `/privacy` discloses the operator, public-alias/private-identity boundary, and
  direct correction, privacy, email-preference, and support routes.
- `/terms` retains the first-party scope, ratings, registration, dispute,
  availability/limitation, correction, and contact sections.

## Intake safety

The browser never reads or writes Supabase. FastAPI validates the club, email,
optional club-owned player ID, and http/https evidence URL. A database query
deduplicates exact 24-hour retries and enforces a per-email, per-club hourly cap
(`JUPR_PUBLIC_SUPPORT_RATE_LIMIT_PER_HOUR`, default `5`). The honeypot remains a
generic success. Public responses never expose requester contact details.

Apply `supabase/migrations/20260719171000_public_support_intake_guardrails.sql`
after the canonical server-only table migration and before enabling public intake.
The table remains RLS-enabled, revoked from `anon`/`authenticated`, and accessible
only to the service role through FastAPI.

## Profile privacy fulfillment

The review queue records identity verification, fulfillment progress, approved
resolution action, and non-sensitive fulfillment evidence. A profile-privacy item
cannot be resolved until identity is `verified`, fulfillment is `completed`, an
action is recorded and non-sensitive fulfillment evidence exists. An admin note
is optional context rather than a terminal-state requirement. The update is
compare-and-swap guarded by `expected_updated_at` and audit-attributed. This queue
does not itself mutate public player data.

Manual staging acceptance must verify identity using an approved off-platform
method, apply the action through the authorized player workflow, inspect player,
leaderboard, match, roster, badge, recap, and live projections, and only then
resolve the queue item. Do not upload identity documents into the evidence field.
