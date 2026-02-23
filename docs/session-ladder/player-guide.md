# Session Ladder Engine MVP — Player Guide

This guide explains what players can expect during Session Ladder nights.

## How a session works
1. You are added to the session roster (expected/check-in/walk-in).
2. Courts are seeded by current rating snapshot.
3. Games are played and scored court-by-court.
4. Standings are computed automatically from game results.
5. Movement occurs between rounds (max 1 court up/down).
6. Session is completed and then published with recap + updated leaderboard.

## Roster intake experience
Managers can add players using:
- existing-player search/select,
- manual new-player entry,
- copy/paste lists,
- CSV upload.

As a player, the key statuses are:
- `EXPECTED`
- `CHECKED_IN`
- `NO_SHOW`
- `WALK_IN`

## Seeding and standby
- Seeding is rating-based (highest ratings first).
- Courts are uniform size (4 or 5 per court).
- If attendance exceeds court capacity, overflow players are standby until assigned.

## Scoring and stats
Scores are entered per game. The app auto-computes:
- Wins/Losses
- Points For / Points Against
- Point Differential

Tie-breakers are applied in this order:
**Wins → PD → PF → H2H → Playoff**.

## Movement between rounds
- Players move at most one court each round.
- Top performers move up; bottom performers move down.
- Movers-per-court can be 1 or 2, depending on manager settings.

## Ratings and publish
At session completion:
- ratings are updated once for the session,
- rating history is stored with session reference.

When published:
- recap and updated leaderboard are shown,
- results are locked,
- attendance contributes toward awards eligibility.

## Awards eligibility
- Eligibility is based on attended sessions in league/season context.
- Minimum sessions required is configured by the club (default target is typically 4).

## Future live scoring
`FEATURE_LIVE_SCORING` is planned for future UX enhancements.
The current MVP already stores game-level records needed to support real-time scoring views later.
