# Moneyball vs Expectations

Choose **Event format → Moneyball vs Expectations** in either the public or staff Round-Robin Generator. Select eight different players from the club directory. Each must have a valid doubles club rating before setup. This first version uses two courts, seven fixed rounds, games to 11 and win by two. Every player partners once with every other player and faces each twice. There are no handicap starting scores.

Each player contributes 100 MXN. The highest Moneyball score takes the 800 MXN pool; tied winners split it. The application shows settlement amounts for the organizer to pay manually. It does not collect or transfer money.

## Scoring

The versioned model and algorithm match the offline Fair Play prototype. Frozen team ratings predict possible point margins for each game. Each player's total point differential is converted to a midpoint percentile of their predicted distribution, then adjusted using logit offsets calibrated from simulated whole events before play. Every simulated game has one outcome shared by all four players. Preparation uses 500,000 calibration events and an independent 1,000,000-event audit; a setup is rejected if a player's modeled first-place share deviates from 12.5% by more than 0.5 percentage points.

Wins and expected wins are displayed for context. There is no separate win bonus: ranking uses the adjusted point-performance percentile, as in the prototype. The model was fitted to earlier Tres Palapas games and is marked as experimental. Equal chances under that model are not a guarantee of equal real-world winning chances; accurate starting ratings matter.

## Frozen inputs and live results

The server resolves ratings from club-scoped player IDs; clients cannot supply ratings or balancing offsets. Preview fingerprints include the frozen setup, so a rating change between preview and start requires a fresh preview. The persisted session stores its private lock, model version, distributions, and offsets. Starting another event recalculates expectations. Score changes never recalibrate an existing event.

Player ratings, distributions, and offsets are removed from preview/session payloads, public exports, and shared viewing/handoff links. Standings show wins, expected wins, point differential, expected point differential, and Moneyball score. Expected figures cover saved games only. Interim scores are provisional; the full-event balancing audit does not claim equal winning chances for an incomplete event.

All fourteen games must be saved before the session can finish or display a final settlement. Selecting pending rounds in a different order is supported. Roster changes, skipped rounds, playoffs, early completion, and extra rounds are rejected for this format. Start a fresh event if the lineup changes. Saved scores can be corrected before submission; corrections immediately recalculate standings and settlement using the same frozen inputs. Exact ties use unrounded scores; displaying two decimals does not create a tie. Any leftover payout cents are allocated in standings order.

Rated/unrated match publication remains the organizer's pre-event choice and goes through the existing submission/approval workflow. Moneyball expectations do not change that workflow or silently update club ratings. Existing standard round robins retain their flexible roster, open-ended play, scoring modes, and playoffs.

No schema migration is needed: the private lock lives in the existing session JSON state. Persisted locks keep their included model, so later model versions cannot alter an event already played.
