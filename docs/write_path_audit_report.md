# Write-Path Audit Report

## Scope
Tables audited:
- `matches`
- `tournament_games`
- `league_ratings`
- related write paths participating in match creation / tournament generation

Search patterns used:
- `.insert(`
- `.upsert(`
- `.update(`
- `.delete(`
- `supabase.table("matches")`
- `supabase.table("tournament_games")`

## Structured summary

| File | Function | Table written | Bypasses `record_match()`? | `idempotency_key` used | `club_id` included | `on_conflict` defined |
|---|---|---|---|---|---|---|
| `jupr_app/domain/match_pipeline.py` | `record_match` | `matches` (`sb_insert`) | No (canonical pipeline) | Yes (required) | Yes | N/A (`insert`) |
| `jupr_app/domain/match_pipeline.py` | `update_match` | `matches` (`sb_update`) | No (pipeline mutation) | N/A | Yes | N/A |
| `jupr_app/domain/match_pipeline.py` | `delete_match` | `matches` (`sb_delete`) | No (pipeline mutation) | N/A | Yes | N/A |
| `jupr_app/domain/replay_history.py` | `replay_history` | `matches` (`sb_update` snapshots) | No raw insert; post-processing rewrite path | N/A | Yes | N/A |
| `jupr_app/domain/replay_history.py` | `replay_history` | `league_ratings` (`delete` + `replace_league_ratings` RPC) | N/A | N/A | Yes | N/A |
| `jupr_app/domain/player_merge.py` | `merge_player_into` | `matches` (`update` player slots) | Not match creation; maintenance path | N/A | Yes | N/A |
| `jupr_app/domain/player_merge.py` | `merge_player_into` | `league_ratings` (`update`) | N/A | N/A | Yes | N/A |
| `jupr_app/ui/pages/tournaments.py` | tournament generation handlers | `tournament_games` (`upsert`) | N/A | N/A | Yes | Yes (`on_conflict` present) |
| `jupr_app/ui/pages/tournaments.py` | tournament reset/cleanup handlers | `tournament_games` (`delete`) | N/A | N/A | Yes | N/A |
| `jupr_app/ui/pages/tournaments.py` | bracket result handlers | `tournament_games` (`update`) | N/A | N/A | Yes | N/A |
| `jupr_app/ui/pages/record_match.py` | tournament result sync handlers | `tournament_games` (`update`) | N/A | N/A | Yes | N/A |

## Violations

### Raw `.insert()` to `matches`
- **None found** in repository write paths. `matches` creation flows through `jupr_app.domain.match_pipeline.record_match()`.

### Writes bypassing `record_match()` (for match creation)
- **None found** for direct `matches` creation paths.
- Non-creation maintenance writes (`update`/`delete`/snapshot rewrites/merge operations) are present and expected.

## Notes
- `tournament_games` generation paths already use `upsert(..., on_conflict=...)` in tournament generation handlers.
- `league_ratings` replay paths rely on deterministic rebuild logic via `replace_league_ratings` RPC.
