# Adaptive doubles rotation quality

Standard standalone Round Robins preserve their initial fixed previews, including
non-perfect schedules that may later join a hub. The improved picker starts when
attendance requires adaptation or play extends beyond the initial fixed cycle.
It uses all recorded
partner/opponent history, including players who have since left. Absences do not
create catch-up bye debt, and available courts are filled before matchup scoring.

## Priority order

1. Select eligible playing groups using the existing attendance/bye fairness rules.
2. Minimize repeated partner pairs and previous partnership counts in the next round.
3. Limit concentrated partnerships and consecutive partnership streaks.
4. Limit concentrated opponent encounters and consecutive opponent streaks.
5. Within those limits, improve partner coverage and repeat spacing across the next two rounds.
6. Use aggregate opponent counts, exact matchups, and repeated court groups to break ties.

These are preferences within legal playing groups, not promises that every repeat
can be avoided. Small groups and longer sessions necessarily repeat pairings.

## Bounded planning

For four or eight seated players, evaluate all 3 or 315 distinct court/team
layouts for each eligible playing group. For larger groups, evaluate 64 seeded
candidates per eligible playing group. Keep at most four candidates and evaluate
one additional round, with at most five eligible playing groups in that second
round. The search is deterministic and does not optimize the full session.

Lookahead uses the next round's actual roster boundaries. It is disabled across
committed, skipped, or partially scored future rounds. Planning cache keys include
the next-round roster and committed state. Played, active, scored, skipped, and
otherwise protected rounds retain their assignments under existing controls.

Partner/opponent meeting rounds are reconstructed from saved state, including
explicit round numbers for games entered out of order. Cancelled and non-played
matches do not count. Skipped rounds do not extend a consecutive-round streak.

Moneyball fixed lineups, ladder mini-games, mixed singles/doubles allocation, and
hub dispatch retain their existing schedulers.

## Regression example

An anonymous fixture reproduces an eight-player start, a ninth player joining in
Round 2, and one departure before Round 5. The old Round 7 repeated an opponent
pair for the third consecutive round. The updated picker keeps all four fresh
partnerships, seats all eight remaining players, and avoids transferring that
three-round streak to another pair. A second fixture verifies that unavoidable
Round 8 partnership repeats can be spaced away from Round 7.
