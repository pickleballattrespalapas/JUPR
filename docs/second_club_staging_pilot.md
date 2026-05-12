# Second Club Staging Pilot Checklist

## Purpose

Validate JUPR multi-club behavior in staging using a non-production second club before any production multi-club rollout.

## Prerequisites

- Staging Supabase is verified and currently used for `Test` branch validation.
- `clubs` table migration is applied in staging.
- Club-scoped admin roles are implemented and enforceable.
- Public leaderboards are working for club-scoped reads.
- Staging email safety is enabled (non-production routing/allowlist guardrails).
- Next admin score entry remains disabled unless explicitly testing auth behavior in staging.

## Setup checklist

- [ ] Create a second club row in staging (non-production tenant only).
- [ ] Create a small set of staging-only players for the second club.
- [ ] Create league metadata for the second club.
- [ ] Assign `club_owner`, `organizer`, and `scorekeeper` roles scoped to the second club.
- [ ] Confirm Tres Palapas staging data remains separate after setup.

## Validation checklist

### Public

- [ ] Validate `/clubs/{second-club}` resolves correct club public profile in staging.
- [ ] Validate `/clubs/{second-club}/leaderboards` returns only second-club data.
- [ ] Confirm no private fields are exposed in public responses.

### Admin

- [ ] Verify second-club scorekeeper cannot access Tres Palapas write operations.
- [ ] Verify Tres Palapas scorekeeper cannot write second-club matches.
- [ ] Verify admin activity log records the correct `club_id` for every write attempt.
- [ ] Verify worker runs are scoped to the selected club.

### Data isolation

- [ ] Confirm matches are club-scoped.
- [ ] Confirm leaderboards are club-scoped.
- [ ] Confirm roles are club-scoped.
- [ ] Confirm email subscriptions/outbox are club-scoped.

## Rollback

- Disable second club in staging.
- Remove staging-only test data if needed.
- Do not touch production systems, data, or configuration.

## Non-goals

- No billing.
- No public signup.
- No production multi-club launch.
