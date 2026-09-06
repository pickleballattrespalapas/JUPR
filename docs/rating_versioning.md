# JUPR rating calculation versioning

Rating versioning is backend audit metadata. It does not add a second player-facing score, a provisional label, confidence, reliability, or recency weighting.

Every `matches` row stores:

- `rating_algorithm_version`: the mathematical update rule;
- `rating_parameter_version`: the named default parameter policy;
- `rating_parameters`: the effective numeric settings, including a managed league's K override when one applies.

The current frozen policy is:

- algorithm: `jupr-hybrid-score-share-v1`;
- parameters: `flat-k32-floor1-loser-cap16-v1`;
- overall K: 32 Elo;
- winner floor: +1 Elo;
- positive loser-gain cap: +16 Elo;
- scale: 400 Elo per 1.000 JUPR.

The registry table is server-only. Match inserts are stamped even when they pass through an older atomic RPC whose explicit column list does not yet mention the metadata. Rating replay updates are stamped by the same database trigger.

## Required process for a formula change

1. Run and retain the chronological baseline report for the currently deployed version.
2. Add a new algorithm or parameter ID; never reuse an existing ID for changed behavior.
3. Add the new pair and its complete parameters to `rating_calculation_versions` in a migration.
4. Update the match defaults and insert/replay trigger in that migration.
5. Shadow-test the candidate on the same chronological rows.
6. Reject any candidate that violates winner-and-gain or removes legitimate loser gains for score outperformance.
7. Deploy only after predictive metrics and policy checks are reviewed.

Historical rows keep their original version. Replaying history intentionally stamps recalculated snapshots with the version that performed the replay.
