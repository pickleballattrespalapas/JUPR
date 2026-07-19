# Next League Results parity contract

The public Next.js route `/clubs/{clubSlug}/league-results` remains read-only
and FastAPI-backed. Streamlit remains the operational fallback until the final
manual parity session approves this page.

## Public contract

`GET /clubs/{clubSlug}/league-results` accepts `league_name`, `week`, `player`,
and `weekly_min_games`. FastAPI validates every selector against the club-scoped
league data and returns the effective selections. The response contains:

- current `league_ratings` standings and season rating deltas;
- configured league weeks, including scheduled weeks with no results;
- weekly games, win percentage, rating delta, end-of-week rank, and rank delta;
- distinct selected-week and season highlight sets with explicit minimum-game
  qualification thresholds;
- the complete public player selector, selected-player summary and trends; and
- up to 15 public-safe recent match projections with partner, opponents, result,
  score, rating delta, and stable match/player links.

Private player/contact fields and raw match rows are never returned. The API
selects only the columns used to build the public projection.

## Automated acceptance

- Domain tests cover week/player selection, qualification, dense weekly ranks,
  rank changes, season-vs-week highlight scope, and recent-match projections.
- API tests cover the public denylist and selector validation.
- The Next production build and a staging browser test cover deterministic
  selector hydration plus the all-sections print view.

## Manual acceptance held for the final session

Compare a representative league with Streamlit for standings, every configured
week, rank movement, season and weekly highlights, a long player selector,
recent matches, deep links, responsive layout, and printed pagination. Do not
change the parity matrix row to `Done` until that session is signed off.
