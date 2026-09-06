# PCS interclub implementation plan

Status: design foundation; no interclub rating writes enabled.

## First season
Southern BCS (La Paz and south), January–March. Eight participating clubs,
with four clubs at each meet and two simultaneous meets at different hosts.
Primary divisions 3.5 and 4.0; optional 4.5/Open. Four-player teams. Meets fit
within three hours. Preserve the agreed host-club encounters and multiple
opponents per trip. Match format: three games to 11, win by two, no cap.
These are starting season settings, not hardcoded platform limits.

## Account boundaries
One login can have separate club roles. Club player records and ratings remain
islands. A league is owned by an organizing club; participating clubs retain
control of their own rosters. Free participation includes one administrator and
no operators. Either organizer or host administrators can assign meet access
within their club's staff allowance. Host access never authorizes league rules,
other clubs' directories, published-result corrections, or rating approval.

## Data model to implement
- interclub_seasons: organizing club, name, dates, timezone, state, version,
  eligibility rules, scoring rules, divisions and roster deadline.
- interclub_participations: season, participating club, acceptance state.
- interclub_teams: participation, division, team name.
- interclub_entries: team, represented club, source club player ID, immutable
  starting league rating, submitted name and eligibility facts.
- interclub_meets: season, host participation, scheduled time, court inventory,
  state, version and rating approval state.
- interclub_meet_teams: meet/team association and submitted roster revision.
- interclub_meet_staff: explicit staff assignment referencing an eligible club
  staff membership. Effective access expires/revokes with either assignment.
- interclub_games: meet, division, teams, player entry IDs, individual game
  scores and non-play result. Store represented club at time of play.
- interclub_rating_batches: meet, approved result revision, organizer actor,
  calculation version, idempotency key and processing status.
- interclub_rating_effects: batch, source game, entry, destination club/player,
  rating stream and before/after values. Unique on game/player/stream/revision.
- interclub_audit: actor, action, version and before/after changes.

Do not copy email, phone or whole club directories into league records. Resolve
submitted source player IDs under the submitting club's authorization, never
from user-supplied destination club IDs. Enforce source player/club ownership
with database constraints as well as API checks.

## Operator workflow
1. Organizer creates a draft season, divisions, eligibility and scoring rules.
2. Organizer invites clubs; each club administrator accepts and submits teams.
3. Organizer schedules meets and hosts. Host or organizer assigns eligible staff.
4. Club administrators may update their own roster after the deadline, but
   eligibility violations require organizer-administrator approval. Never
   rewrite participants in already played games when changing a roster.
5. Meet staff run check-in, courts and scores using established league rules.
6. Results can be published separately from rating approval.
7. Organizer administrator reviews the meet and approves it for ratings.
8. Approved corrections automatically recalculate affected rating histories.

## Rating pipeline
Seed each league entry from its represented club's rating at entry time.
Reuse the existing separate league and overall calculation behavior. League
rating changes are not simply copied as deltas into overall club ratings.
The overall destination is only the represented club; other islands do not
change. Participants from Free clubs still receive the approved rating update.

Approval snapshots the complete meet revision. Reject stale approval requests.
Process each game once per player and rating stream; duplicate submissions or
worker retries cannot double-apply ratings. Preserve chronological ordering,
including local club games between interclub games. Later corrections replay
all dependent effects consistently across both streams. Use a durable job with
visible pending/failed/completed status and retry-safe checkpoints. No partially
updated meet may be labeled fully rated. Retain audit of original and corrected
results. Existing club rating algorithms and replay semantics must be inspected
and characterized with fixtures before enabling this pipeline.

## Delivery sequence and gates
1. Club staff management and enforced scope/expiry boundaries (current work).
2. Draft season editor, club participation, teams/rosters and host scheduling.
3. Day-scoped meet operations using shared tournament/live primitives.
4. Organizer review and dual-stream rating pipeline; correction/retry tests.
5. Public schedule/results, privacy audit and pilot acceptance in staging.

Required tests include cross-club roster injection, host escalation to league
rules, expired staff, invalid eligibility, post-deadline valid substitutions,
unauthorized publication corrections, duplicate approval, simultaneous approval
and correction, failures between rating streams, and historical replay with
interleaved club games. Production release requires separate authorization.

## Agreed plans and billing (future enforcement)
Free: participation only, 1 admin / 0 operators.
Basic: $19/month or $190/year; generators/live play; 2 admins / 3 operators.
Middle: $39/month or $390/year; Basic plus leagues, challenge ladders and club
rating reports/history; 3 admins / 5 operators.
Top: $69/month or $690/year; Middle plus tournaments/interclub organizing;
unlimited staff. Basic/Middle extra slots: $3/month or $30/year for either role.
Personal rating access stays free. No staff add-ons on Free.

One month trial per club starts on activation of chosen tier; tier changes keep
original trial end. Card upfront once billing launches; automatic billing unless
canceled. Upgrades/add-ons immediate and prorated. Downgrades/cancellations and
slot removals at period end, no partial refund. Select retained staff before
confirming downgrade. Unsupported programs remain viewable but pause. Failed
renewal: three days grace, then paid operations pause while admins retain billing
and records. Pricing/usage review reminder: November 30, 2026.

## Platform onboarding increment
The `/admin/platform` dashboard lists club accounts, creates a draft club plus
its first administrator atomically, and tracks draft/in-progress/ready-for-review
onboarding. Only identities in `pcs_platform_admins` can access these endpoints.
The migration bootstraps the existing Tres Super Admin through its Auth user ID;
subsequent club role grants do not grant platform authority. There is no public
endpoint to appoint platform administrators. Every onboarding write is audited.

This increment does not launch a club, activate trials, charge cards, send email,
or implement public club signup. Draft status is onboarding/public-visibility
metadata, not a complete entitlement gate. Plan enforcement and invitations are
required before opening self-service onboarding. The dashboard displays existing
plan status rather than implying subscription enforcement is already complete.
