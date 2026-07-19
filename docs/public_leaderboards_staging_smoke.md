# Public Leaderboards Staging Smoke

This slice makes the Next leaderboard automated-ready. It does not mark the
Streamlit parity row `Done`; manual comparison remains part of the consolidated
acceptance session.

## Contract

- `OVERALL` reads the club's overall player rating and record projection.
- Named league tabs read only that league's rating projection and its active
  `min_games` qualification rule.
- `status=active` is the default. `all` and `inactive` are explicit public views.
- Search does not change a player's canonical rank. A unique name match or a
  `player_id` deep link adds the Player Snapshot.
- Gain is `(rating - starting_rating) / 400`; gap is the preceding ranked
  player's Elo difference divided by 400.
- At most three public badge definitions are embedded per player. The total
  badge count supplies the overflow label.
- FastAPI returns only allowlisted player, scope, summary, badge, highlight, and
  pagination fields. Supabase credentials and raw table rows stay server-side.

## Automated staging check

With the staging URL and Vercel bypass secret configured, run:

```bash
cd apps/web
npm run test:e2e:staging -- e2e/leaderboards.staging.spec.ts
```

The route-specific suite checks a populated Active/Overall view, parity columns,
snapshot links, search, a one-row pagination fixture, league qualification,
filtered-empty behavior, and the unknown-club API error state.

## Deferred manual acceptance

During the final all-page session, compare a representative Overall tab and a
league tab against Streamlit (`page=leaderboards&public=1`):

1. Confirm ordering, rating, gain, gap, W-L, games, win percentage, qualification,
   badges, and active/inactive labels.
2. Open a player snapshot and both its profile and badge deep links.
3. Exercise every status and sort option, search for one and multiple names, and
   traverse pagination without losing filters.
4. Compare populated, filtered-empty, and mobile horizontal-table layouts.

The page is read-only. If staging data or layout is wrong, keep the Streamlit
public leaderboard available and roll back the Next deployment; no data recovery
step is required.
