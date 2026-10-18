# Moneyball vs Expectations

Choose **Event format → Moneyball vs Expectations** in either the public or staff Round-Robin Generator. Set the complete starting lineup using different players from the club directory. Choose six players for the balanced six-round rotation, or any supported perfect doubles RR count: 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32, 33, 36, 37 or 40. Each must have a valid doubles club rating before setup. Use all required courts and the full fixed rotation, with games to 11 and win by two. For the perfect rotations, each player plays N−1 games. Multiples of four use N−1 rounds; counts one above a multiple of four use N rounds with one bye each. Every player partners once with every other player and faces each twice. There are no handicap starting scores.

**Six-player exception:** one court, six rounds, four games and two byes per player. Each player has four distinct partners, faces every opponent once or twice, and sits once in each three-round block with no consecutive byes. This is balanced but is not a perfect round robin: everyone misses one potential partner and opponent counts vary. It uses the same frozen rating expectations and independent balance audit, with distributions based on four games per player and a 600 MXN pool. Standard six-player round robins keep their existing adaptive scheduling.

Each player contributes 100 MXN. The highest Moneyball score takes the 100 × starting-player-count MXN pool; tied winners split it. The application shows settlement amounts for the organizer to pay manually. It does not collect or transfer money.

## Scoring

The versioned model and algorithm match the offline Fair Play prototype. Frozen team ratings predict possible point margins for each game. Each player's total point differential is converted to a midpoint percentile of their predicted distribution, then adjusted using logit offsets calibrated from simulated whole events before play. Every simulated game has one outcome shared by all four players. Preparation uses 500,000 calibration events and an independent 1,000,000-event audit; a setup is rejected if a player's modeled first-place share deviates from 100/N percent by more than 0.5 percentage points.

Wins and expected wins are displayed for context. There is no separate win bonus: ranking uses the adjusted point-performance percentile, as in the prototype. The model was fitted to earlier Tres Palapas games and is marked as experimental. Equal chances under that model are not a guarantee of equal real-world winning chances; accurate starting ratings matter.

## Frozen inputs and live results

Before starting, both the public and staff matchup previews show each player's scheduled games, expected wins, expected losses, and expected point differential for the **full event** above the round-by-round matchups. These totals come from the same frozen game probabilities used for scoring, without exposing individual ratings. Byes are excluded (six-player rotations show four games per player). Decimal expectations are averages: three wins beats an expectation of 2.6 wins, but wins do not award a separate bonus or guarantee the prize. An older saved preview without these totals prompts the organizer to generate a new preview.

The server resolves ratings from club-scoped player IDs; clients cannot supply ratings or balancing offsets. Preview fingerprints include the frozen setup, so a rating change between preview and start requires a fresh preview. The persisted session stores its private lock, model version, distributions, and offsets. Starting another event recalculates expectations. Score changes never recalibrate an existing event.

Player ratings, distributions, and offsets are removed from preview/session payloads, public exports, and shared viewing/handoff links. Standings show wins, expected wins, point differential, expected point differential, and Moneyball score. Expected figures cover saved games only. Interim scores are provisional; the full-event balancing audit does not claim equal winning chances for an incomplete event.

All scheduled games must be saved before the session can finish or display a final settlement. Selecting pending rounds in a different order is supported. Adding, removing or reordering the starting lineup, skipping rounds, playoffs, early completion and extra rounds are rejected. Substitutions may fill the same scheduled slot for the current unscored round or all remaining unplayed games. Both the original player and every substitute in that slot are permanently ineligible to win. Their individual games and stats remain visible, marked DQ; they receive no Moneyball score or payout. Everyone else keeps their original expectations. Substitutions do not add to the pool. If no eligible players remain, the pool is returned to the starting players. Saved scores can be corrected before submission; corrections immediately recalculate standings and settlement using the same frozen inputs. Exact ties use unrounded scores; displaying two decimals does not create a tie. Any leftover payout cents are allocated in standings order.

Rated/unrated match publication remains the organizer's pre-event choice and goes through the existing submission/approval workflow. Moneyball expectations do not change that workflow or silently update club ratings. Existing standard round robins retain their flexible roster, open-ended play, scoring modes, and playoffs.

No schema migration is needed: the private lock lives in the existing session JSON state. Persisted locks keep their included model, so later model versions cannot alter an event already played.

Version 2 prepares calibration and audit simulations in bounded-memory batches for larger lineups. Previously frozen version 1 eight-player sessions retain their original model, distributions, offsets and payout.

### New players during setup

Club staff with player-management permission can use **Create new player** in either Moneyball setup page, enter a name and doubles starting level (1–7), then choose **Create and select player**. The existing guarded player-creation service creates the club profile and adds it to the current lineup before expectations are calculated. Existing matching profiles are reused without changing their ratings. Interrupted creation can be checked after refreshing, and preview/start wait until creation is resolved. Signed-out organizers see a staff sign-in link and keep their saved lineup.
