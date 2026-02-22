# AUDIT: Leagues Manager (branch `rebuild`)

## Overview + scope
This audit covered the Streamlit **League Manager** page, the league domain logic it relies on, and the replay/recompute paths that keep league state and standings coherent.

Scoped code paths:
- UI entry + routing: `streamlit_app.py`, `jupr_app/ui/pages/league_manager.py`
- League domain/status + awards: `jupr_app/domain/leagues.py`
- League data loading/persistence touchpoints: `jupr_app/data/load.py`
- Replay path used by Admin/league maintenance: `jupr_app/domain/replay_history.py`, `jupr_app/ui/pages/admin_tools.py`
- Regression test coverage: `tests/test_replay_history_rpc.py`

## Architecture map (short)
1. Request/UI navigation:
   - Streamlit routes "🏟️ League Manager" to `league_manager.render(ctx)`.
2. Handler/UI workflow:
   - `league_manager.render` initializes ladder state, reads `ctx.df_leagues/df_meta`, and runs setup + roster + live event flow.
3. Service/domain:
   - League metadata/status/awards helpers from `jupr_app/domain/leagues.py`.
4. Persistence/data:
   - `jupr_app/data/load.py` loads `league_ratings`, `leagues_metadata`, and `matches` from Supabase.
5. Maintenance/rebuild path:
   - Admin tools trigger replay via `jupr_app/domain/replay_history.py`, which recomputes and writes league ratings through `replace_league_ratings` RPC.

Flow summary: `Streamlit page -> league_manager.render -> domain helpers + Supabase reads/writes -> replay_history (when maintenance/replay is run) -> updated league_ratings/matches/players`.

## Expected behavior (source of truth)
- The League Manager page should be reachable from app navigation and render for admin users.
- Replay history should recompute league ratings and invoke `replace_league_ratings` RPC for league resets.
- Replay logic should be robust to Supabase query implementations used in tests/mocks.

Primary source of truth used:
- `tests/test_replay_history_rpc.py` (explicitly validates replay uses `replace_league_ratings` RPC)
- `streamlit_app.py` + `league_manager.py` routing/render wiring
- `admin_tools.py` replay controls and dispatch

## Reproducing original failure
### Minimal failing command (pre-fix)
```bash
pytest -q tests -k "league_manager or leagues or league"
```

### Observed failure
- `tests/test_replay_history_rpc.py::test_replay_history_uses_replace_league_ratings_rpc` failed.
- Error: `_Query.select() got an unexpected keyword argument 'count'` from `replay_history._count_rows`.

This blocked league replay maintenance flow in compatibility contexts using a minimal/mock Supabase query object.

## Issue list

| Severity | Area | File(s) | Evidence | Root cause | Fix | Test coverage |
|---|---|---|---|---|---|---|
| High | League replay maintenance | `jupr_app/domain/replay_history.py`, `tests/test_replay_history_rpc.py` | Failing test shows TypeError on `select(..., count="exact")` with mock query impl | `_count_rows` assumed a specific Supabase client signature (`count` kwarg + `limit`) and no fallback path | Added compatibility fallback: try `select(..., count="exact")`, fallback to `select("id")`; apply `.limit(1)` only when available; fallback to `len(data)` when response has no `.count` | `tests/test_replay_history_rpc.py` passes; league-focused test subset passes |

## What changed (high level)
- Updated replay row counting to be client-signature tolerant and resilient when count metadata is absent.
- Verified league replay RPC path tests and league-focused suite now pass.

## Commands run + outcomes
- `git branch --show-current && git status --short && rg -n "LeaguesManager|LeagueManager|leagues manager|/leagues|league_|league-" .` (inventory/search)
- `pytest -q tests/test_end_league_top_performers.py tests/test_load_data_leagues_metadata.py tests/test_navigation_determinism.py` (initial targeted baseline, passed)
- `pytest -q tests -k "league_manager or leagues or league"` (pre-fix repro, failed on replay_history count signature)
- `pytest -q tests/test_replay_history_rpc.py tests/test_end_league_top_performers.py tests/test_load_data_leagues_metadata.py tests/test_navigation_determinism.py` (post-fix targeted validation, passed)
- `pytest -q tests -k "league_manager or leagues or league or replay_history"` (post-fix broader league subset, passed)
- `python -m compileall .` (CI-aligned compile check, passed)

## Remaining risks / follow-ups
- Replay behavior correctness against a real Supabase backend is not fully end-to-end covered here (only unit-level + mock-level replay test).
- Existing uncommitted `jupr_court_board/frontend/node_modules/.package-lock.json` workspace change was pre-existing and intentionally left untouched.
