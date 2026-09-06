# JUPR chronological predictive backtest

This backtest freezes the current player-facing JUPR policy before any formula experiment:

- one in-club rating per format;
- the same K factor from the first rated match;
- a winner always gains;
- a loser may gain by outperforming the score expectation;
- no provisional, reliability, uncertainty, or recency adjustment.

## Method

Matches are sorted by `date`, then stable match ID. For each match, the prediction is calculated from the simulated ratings that exist immediately before that result. Only after the prediction is scored does the result update the ratings. A newly encountered player is seeded from that player's stored pre-match snapshot, falling back to 1200 Elo (3.000 JUPR) when no snapshot exists.

The report includes Brier score, log loss, outcome accuracy, score-share mean absolute error, calibration buckets, monthly slices, singles/doubles slices, a 50% baseline, and explicit checks of the winner-gain rule. Lower Brier score, log loss, and score-share error are better.

This is a walk-forward predictive evaluation, not a random train/test split. Randomly shuffling matches would leak future rating information into earlier predictions.

## Input

Export rated and unrated match rows with these columns:

`id,club_id,date,deleted_at,rating_scope,match_format,match_type,t1_p1,t1_p2,t2_p1,t2_p2,score_t1,score_t2,t1_p1_r,t1_p2_r,t2_p1_r,t2_p2_r,rating_bonus_elo`

JSON, JSON Lines, and CSV are supported. State is isolated by `club_id`, so club ratings never bleed into one another. An export without `club_id` is treated as one club. Unrated, deleted, tied, and malformed rows are reported under `skipped` and do not update simulated ratings.

Run:

```bash
python scripts/backtest_ratings.py --input matches.json --output rating-backtest.json
```

or:

```bash
make rating-backtest MATCH_EXPORT=matches.csv
```

Future shadow models should use the same ordered input and metrics. A candidate is not eligible for player-facing use unless `winner_gain_violations` remains zero and its update policy preserves loser gains for genuine score outperformance.

The first aggregate staging run is retained in
`docs/rating_backtest_baseline_staging_2026-09-06.json` only as a pipeline smoke
test. Staging contains randomly entered acceptance data, so its predictive
metrics must not be used to select or change a rating formula.

The decision baseline is
`docs/rating_backtest_baseline_production_2026-09-06.json`. It was calculated
with read-only queries over real production match history, retains aggregate
metrics only, and includes both an as-recorded snapshot cross-check and a
sensitivity run that excludes the explicitly named `Test League`. No production
rows, schema, ratings, or settings were changed.
