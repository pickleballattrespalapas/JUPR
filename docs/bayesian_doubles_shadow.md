# Bayesian doubles shadow benchmark

## Product boundary

Current JUPR remains the only official and player-visible rating. The Bayesian
model is a private research benchmark. It does not write official ratings,
create matchups, expose uncertainty, introduce provisional status, or change
the rule that an official JUPR winner always gains while a losing team may gain
for beating its score expectation.

## Model

The shadow keeps a private Gaussian belief for each player's doubles skill. A
player's first stored pre-match JUPR seeds the belief's average; 1200 is the
fallback when that seed is absent. This preserves the club's real starting
information while the shadow model independently learns from later results. A
team is the average of its two players. Before each result is consumed, the
model predicts the chance that team 1's performance gap will be positive.

After the prediction is recorded, the score is converted to an Elo-scale gap:

`400 × log10((team 1 score + smoothing) / (team 2 score + smoothing))`

The factorised Gaussian update assigns more of the new evidence to internally
uncertain players. It uses changing partner and opponent networks to separate
individual estimates over time. It uses no time decay or recency drift.

This is a deliberately small score-aware Bayesian team model, not a claim to be
Microsoft TrueSkill. The design follows established Bayesian team-rating ideas
that individual Gaussian skills can be inferred from team results, and that
score-aware likelihoods may improve predictions when data is limited:

- Herbrich, Minka, and Graepel, *TrueSkill: A Bayesian Skill Rating System*
  (2007): https://www.microsoft.com/en-us/research/publication/trueskilltm-a-bayesian-skill-rating-system/
- Graepel, *Score-based Bayesian Skill Learning* (2012):
  https://www.microsoft.com/en-us/research/publication/score-based-bayesian-skill-learning-2/

## Historical selection protocol

Parameters are selected using a predeclared grid and the lowest Brier score in
the validation window. Log loss breaks exact ties. The selection run stops at
the end of validation and therefore cannot inspect the holdout. Only the chosen
configuration is then evaluated on the untouched holdout.

For the first production study:

- history through February 2026 trains the shadow state;
- March 2026 selects the private Bayesian parameters;
- April and May 2026 form the historical holdout;
- the upcoming season is the stronger prospective holdout.

The aggregate, player-anonymous result is checked in at
`docs/rating_shadow_comparison_production_2026-09-06.json`. On the untouched
April-May holdout, the Bayesian shadow improved winner accuracy from 60.8% to
63.5%, Brier score from 0.2308 to 0.2197, and score-share mean absolute error
from 0.1619 to 0.1562. This is promising historical evidence, not a promotion
decision.

Run a local export without retaining player-level output:

```bash
python scripts/compare_rating_models.py \
  --input matches.json \
  --validation-start 2026-03-01 \
  --validation-end 2026-04-01 \
  --holdout-start 2026-04-01 \
  --exclude-league "Test League" \
  --output rating-shadow-comparison.json
```

The report contains aggregate model metrics and parameter metadata only. It
does not contain player names, player ratings, or match receipts.

The deployed server also includes `jupr_app.workers.rating_shadow_worker`. It
is enabled only in staging and records aggregate results in the existing
server-only `worker_run_log`. The checked-in production configuration keeps
both the general worker flag and the separate production-shadow flag off. A
production run therefore fails closed until a later, explicit approval changes
that exact deployment boundary.

## Promotion rule

No Bayesian result automatically changes JUPR. A future player-facing change
requires a clear prospective improvement in Brier score, calibration, and score
prediction; preserved official winner/loser rules; a simple match receipt; and
separate approval for the exact versioned formula and production deployment.
