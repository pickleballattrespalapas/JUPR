# Round-Robin scoring modes

The Round-Robin Generator supports two session modes.

## Scored

Each round accepts scores. After the round result summary, the organizer opens the cumulative standings and continues to the next round from that page. Standings use the ranking method selected during setup.

## Unscored

Rounds display matchups without score fields or standings. The organizer selects **Round Played** to record that the scheduled matchups occurred and continue directly to the next round. A round may still be skipped. Played rounds remain part of matchup history so later adaptive roster changes continue to minimize repeats.

The Ladder Generator remains scored because its next-round court movement depends on results.

## Durability and completion

**Round Played** is one durable, idempotent operation: it records the round and
advances to the next round, or completes the session on the final round.
Retrying the same operation cannot advance twice. Skipped rounds remain distinct
from played rounds.

Scored sessions always move from the saved round results to the full cumulative
standings before the organizer can continue. The final standings lead to a clear
session-complete state. Official match publishing is available only for scored
staff sessions.
